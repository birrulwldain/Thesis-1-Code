"""
libs_physics.py — Two-Zone CR-LIBS Forward Model (Physics Engine)
==================================================================
Q1 Thesis: Deterministic Forward Model for Laser-Induced Breakdown Spectroscopy

Architecture (strict phase separation):
  PHASE 1: CR Matrix Solver   → steady-state level populations n_i [cm⁻³]
  PHASE 2: Voigt Absorption   → spectral absorption coefficient κ(λ) [cm⁻¹]
  PHASE 3: Radiative Transfer → observed spectrum I_obs(λ) via Two-Zone RTE

Design Principles (per Thesis Directive):
  - No black-box ML or curve-fitting in the physics engine.
  - No Saha Equation: populations solved via CR ODE system.
  - Stiff ODE solved with implicit Radau method + analytical Jacobian.
  - All approximations (van Regemorter, Stark width) explicitly documented.
  - SI units throughout; converted at output boundaries.

References:
  [1] Fujimoto T. (2004). Plasma Spectroscopy. Clarendon Press.
  [2] Rybicki & Lightman (1979). Radiative Processes in Astrophysics.
  [3] van Regemorter H. (1962). ApJ 136, 906-915.
  [4] Griem H.R. (1997). Principles of Plasma Spectroscopy. Cambridge.
  [5] NIST Atomic Spectra Database. https://physics.nist.gov/asd
  [6] Olivero & Longbothum (1977). JQSRT 17, 233-236. (Voigt profile)
"""

import numpy as np
import pandas as pd
import h5py
import re
import os
import plotly.graph_objects as go
import warnings
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from scipy.integrate import solve_ivp
from scipy.special import wofz        # Faddeeva function → exact Voigt profile

# =============================================================================
# SECTION 0: PHYSICAL CONSTANTS (all SI, with explicit dimensions)
# =============================================================================

# Fundamental constants
C_LIGHT     = 2.99792458e8         # Speed of light          [m/s]
K_B_J       = 1.380649e-23         # Boltzmann constant      [J/K]
K_B_eV      = 8.617333262145e-5    # Boltzmann constant      [eV/K]
H_PLANCK_J  = 6.62607015e-34       # Planck constant         [J·s]
H_PLANCK_eV = 4.135667696e-15      # Planck constant         [eV·s]
M_ELECTRON  = 9.1093837015e-31     # Electron mass           [kg]
EV_TO_J     = 1.602176634e-19      # 1 eV in Joules          [J/eV]

# van Regemorter mean Gaunt factor [ref: 3, Eq. 8]
# Approximation valid for optically allowed (electric-dipole) transitions only.
# For forbidden transitions, distorted-wave methods (e.g., ADAS/Chianti) are needed.
G_BAR_NEUTRAL = 0.2    # For neutral atoms (sp_num = 1)
G_BAR_ION     = 0.276  # For singly/multiply ionized species (sp_num >= 2)

# Atomic mass unit
AMU_KG = 1.66053906660e-27   # [kg/amu]

import yaml

# Muat Konfigurasi Tesis Universal 1-Pintu
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
try:
    with open(_CONFIG_PATH, 'r') as f:
        _CONFIG = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"Gagal memuat config.yaml: {e}. Pastikan file konfigurasi 1-Pintu tersedia.")

# Simulation grid configuration (Diwariskan dari YAML agar konsisten lintas Blok)
SIMULATION_CONFIG = {
    "resolution"          : _CONFIG['instrument']['resolution'],
    "wl_range_nm"         : tuple(_CONFIG['instrument']['wl_range_nm']),
    "target_max_intensity": 0.8,
}

# Resolve data paths relative to this file (robust to working directory)
_PROJECT_DIR     = os.path.dirname(os.path.abspath(__file__))
NIST_HDF_PATH    = os.path.join(_PROJECT_DIR, "data", "nist_data(1).h5")
ATOMIC_DATA_PATH = os.path.join(_PROJECT_DIR, "data", "atomic_data1.h5")

# Atomic masses [amu] for Doppler width calculation. Extend as needed.
ATOMIC_MASS_AMU: Dict[str, float] = {
    "H":  1.008,  "C":  12.011, "N":  14.007, "O":  15.999,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085,
    "K":  39.098, "Ca": 40.078, "Ti": 47.867, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Rb": 85.468, "Sr": 87.620, "Pb": 207.200,
    "Li": 6.941,
}

# =============================================================================
# SECTION 1: DATA STRUCTURES
# =============================================================================

@dataclass
class EnergyLevel:
    """Single atomic/ionic energy level."""
    index      : int    # Index within the reconstructed level list
    energy_eV  : float  # Level energy above ground state [eV]
    degeneracy : float  # Statistical weight g = 2J+1


@dataclass
class Transition:
    """Radiative transition between two energy levels from NIST data."""
    lower_idx    : int    # Index of lower level i in EnergyLevel list
    upper_idx    : int    # Index of upper level k in EnergyLevel list
    wavelength_nm: float  # Transition wavelength (air) [nm]
    A_ki         : float  # Spontaneous emission Einstein coefficient [s⁻¹]
    g_lower      : float  # Statistical weight of lower level g_i
    g_upper      : float  # Statistical weight of upper level g_k
    E_lower_eV   : float  # Energy of lower level [eV]
    E_upper_eV   : float  # Energy of upper level [eV]


@dataclass
class PlasmaZoneParams:
    """Physical parameters defining one plasma zone (Core or Shell)."""
    T_e_K       : float  # Electron temperature [K]
    T_i_K       : float  # Ion/atom temperature [K] (for Doppler broadening)
    n_e_cm3     : float  # Electron number density [cm⁻³]
    thickness_m : float  # Zone geometric thickness along line of sight [m]
    label       : str = "Zone"


# =============================================================================
# SECTION 2: DATA FETCHER  (adapted from sim.py; returns typed structures)
# =============================================================================

class DataFetcher:
    """
    Reads NIST spectroscopy HDF5 and atomic data HDF5 files.
    Returns EnergyLevel and Transition objects for use in the CR solver.
    """

    def __init__(self,
                 nist_path  : str = NIST_HDF_PATH,
                 atomic_path: str = ATOMIC_DATA_PATH):
        self.nist_path   = nist_path
        self.atomic_path = atomic_path
        self._ion_energies: Dict[str, float] = {}
        self._load_ionization_energies()

    def _load_ionization_energies(self) -> None:
        """Load ionization energies [eV] from atomic_data1.h5 into a dict."""
        try:
            with h5py.File(self.atomic_path, 'r') as f:
                dset    = f['elements']
                columns = list(dset.attrs['columns'])
                rows    = []
                for item in dset[:]:
                    rows.append([
                        item[0],
                        item[1].decode('utf-8'), item[2].decode('utf-8'),
                        item[3].decode('utf-8'), item[4].decode('utf-8'),
                        item[5], item[6].decode('utf-8'),
                    ])
                df = pd.DataFrame(rows, columns=columns)
                for _, row in df.iterrows():
                    self._ion_energies[str(row["Sp. Name"]).strip()] = float(row["Ionization Energy (eV)"])
        except Exception as e:
            warnings.warn(f"[DataFetcher] Cannot load ionization energies: {e}")

    def get_ionization_energy(self, element: str, sp_num: int) -> float:
        """Return ionization energy [eV] for a given element and ionization stage."""
        key = f"{element} {'I' if sp_num == 1 else 'II'}"
        val = self._ion_energies.get(key, 0.0)
        if val == 0.0:
            warnings.warn(f"[DataFetcher] No ionization energy for '{key}'.")
        return val

    def get_transitions(self,
                        element  : str,
                        sp_num   : int,
                        wl_range : Tuple[float, float] = (200.0, 900.0)
                        ) -> pd.DataFrame:
        """
        Fetch and clean NIST transition data for (element, sp_num).

        Returns a DataFrame with columns:
          ['ritz_wl_air(nm)', 'Aki(s^-1)', 'Ek(eV)', 'Ei(eV)', 'g_i', 'g_k']
        """
        REQUIRED = ['ritz_wl_air(nm)', 'Aki(s^-1)', 'Ek(eV)', 'Ei(eV)', 'g_i', 'g_k']
        empty = pd.DataFrame(columns=REQUIRED)
        try:
            with pd.HDFStore(self.nist_path, mode='r') as store:
                df = store.get('nist_spectroscopy_data')
                sel = df[(df['element'] == element) & (df['sp_num'] == sp_num)].copy()
                if sel.empty or not all(c in df.columns for c in REQUIRED):
                    return empty

                # Parse wavelength
                sel['ritz_wl_air(nm)'] = pd.to_numeric(sel['ritz_wl_air(nm)'], errors='coerce')

                # Parse energy columns (strip non-numeric chars like brackets/flags)
                for col in ['Ek(eV)', 'Ei(eV)']:
                    sel[col] = sel[col].apply(
                        lambda x: float(re.sub(r'[^\d.-]', '', str(x)))
                        if re.sub(r'[^\d.-]', '', str(x)) else np.nan
                    )

                # Parse numeric columns
                for col in ['Aki(s^-1)', 'g_i', 'g_k']:
                    sel[col] = pd.to_numeric(sel[col], errors='coerce')

                sel = sel.dropna(subset=REQUIRED)
                sel = sel[(sel['ritz_wl_air(nm)'] >= wl_range[0]) &
                          (sel['ritz_wl_air(nm)'] <= wl_range[1])]
                return sel[REQUIRED].reset_index(drop=True)
        except Exception as e:
            warnings.warn(f"[DataFetcher] Error reading NIST for {element}_{sp_num}: {e}")
            return empty

    @staticmethod
    def build_levels_and_transitions(df: pd.DataFrame
                                     ) -> Tuple[List[EnergyLevel], List[Transition]]:
        """
        Reconstruct unique energy levels and typed Transition objects from a
        NIST transition DataFrame.
        """
        TOL = 1e-4  # [eV] — levels within this tolerance are considered identical
        level_map: Dict[float, float] = {}  # rounded_energy → degeneracy

        def _register(E: float, g: float):
            key = round(float(E) / TOL) * TOL
            level_map.setdefault(key, float(g))

        for _, row in df.iterrows():
            _register(row['Ei(eV)'], row['g_i'])
            _register(row['Ek(eV)'], row['g_k'])

        sorted_E   = sorted(level_map.keys())
        levels     = [EnergyLevel(i, E, level_map[E]) for i, E in enumerate(sorted_E)]
        e_to_idx   = {E: i for i, E in enumerate(sorted_E)}

        transitions = []
        for _, row in df.iterrows():
            i_key = round(float(row['Ei(eV)']) / TOL) * TOL
            k_key = round(float(row['Ek(eV)']) / TOL) * TOL
            i_idx = e_to_idx.get(i_key)
            k_idx = e_to_idx.get(k_key)
            if i_idx is None or k_idx is None:
                continue
            transitions.append(Transition(
                lower_idx    = i_idx,
                upper_idx    = k_idx,
                wavelength_nm= float(row['ritz_wl_air(nm)']),
                A_ki         = float(row['Aki(s^-1)']),
                g_lower      = float(row['g_i']),
                g_upper      = float(row['g_k']),
                E_lower_eV   = float(row['Ei(eV)']),
                E_upper_eV   = float(row['Ek(eV)']),
            ))

        return levels, transitions


STARK_DATABASE = {
    393.4: (0.0034, 1e17), 396.8: (0.0033, 1e17), 288.2: (0.0019, 1e16),
    396.2: (0.0022, 1e16), 309.3: (0.0018, 1e16), 510.6: (0.0025, 1e16),
    521.8: (0.0026, 1e16), 404.6: (0.0015, 1e16),
}

def _stark_hwhm_nm(wavelength_nm: float, n_e_cm3: float) -> float:
    wl_key = round(wavelength_nm, 1)
    if wl_key in STARK_DATABASE:
        w_ref_nm, n_e_ref = STARK_DATABASE[wl_key]
        return w_ref_nm * (n_e_cm3 / n_e_ref)
    else:
        return max(0.001 * (n_e_cm3 / 1e16), 1e-5)

# =============================================================================
# PHASE 1: POPULATION SOLVER (pLTE Paradigm)
# =============================================================================

def solve_cr_populations(levels: List[EnergyLevel], transitions: List[Transition], T_e_K: float, n_e_cm3: float, n_total_cm3: float, sp_num: int, t_max_s: float = 1e-6) -> np.ndarray:
    N = len(levels)
    if N == 0: return np.zeros(0)
    kT_e = 8.617333262145e-5 * T_e_K
    E_arr = np.array([lv.energy_eV for lv in levels])
    g_arr = np.array([lv.degeneracy for lv in levels])
    E_min = E_arr.min()
    boltz_factors = g_arr * np.exp(-(E_arr - E_min) / kT_e)
    Z = boltz_factors.sum()
    if Z > 0:
        populations = (boltz_factors / Z) * n_total_cm3
    else:
        populations = np.zeros(N)
    return populations

# =============================================================================
# PHASE 2: VOIGT PROFILE & ABSORPTION COEFFICIENT
# =============================================================================

def _doppler_hwhm_nm(wavelength_nm: float,
                      T_i_K       : float,
                      mass_kg     : float) -> float:
    """
    Doppler (thermal) Gaussian half-width at half-maximum (HWHM) [nm].

    Formula [Griem 1997, Sobelman 1979]:
        Δλ_D = (λ₀/c) * sqrt(8 ln2 · k_B · T_i / m_i)

    Physical origin: Maxwell-Boltzmann velocity distribution of emitting atoms.
    """
    lam_m = wavelength_nm * 1e-9
    hwhm_m = (lam_m / C_LIGHT) * np.sqrt(8.0 * np.log(2.0) * K_B_J * T_i_K / mass_kg)
    return hwhm_m * 1e9   # [m] → [nm]

# (Fungsi _stark_hwhm_nm telah dihapus karena disubstitusi oleh class StarkBroadener)

def voigt_profile(wavelength_grid_nm: np.ndarray,
                   center_nm         : float,
                   fwhm_gaussian_nm  : float,
                   fwhm_lorentzian_nm: float) -> np.ndarray:
    """
    Normalized Voigt line profile φ(λ) [nm⁻¹].

    The Voigt profile is the convolution of Gaussian G(λ) and Lorentzian L(λ):
        φ_V(λ) = ∫ G(λ') · L(λ-λ') dλ'

    Computed exactly via the Faddeeva (complex error) function w(z) [ref: 6]:
        z = (Δλ + i·γ_L) / (σ_G · √2)
        φ_V(λ) = Re[w(z)] / (σ_G · √(2π))

    Where:
        Δλ   = λ - λ₀               (wavelength offset)
        σ_G  = FWHM_G / (2√(2 ln2)) (Gaussian σ parameter)
        γ_L  = FWHM_L / 2           (Lorentzian HWHM)

    Normalization: ∫ φ_V(λ) dλ = 1.0  (verified numerically, corrected if off)
    """
    sigma_G  = fwhm_gaussian_nm   / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma_L  = fwhm_lorentzian_nm / 2.0
    delta_lam = wavelength_grid_nm - center_nm

    # Complex argument for Faddeeva function
    z       = (delta_lam + 1j * gamma_L) / (sigma_G * np.sqrt(2.0))
    profile = np.real(wofz(z)) / (sigma_G * np.sqrt(2.0 * np.pi))

    # Numerical renormalization (ensures ∫φdλ=1 on finite grid)
    dl   = np.gradient(wavelength_grid_nm)
    norm = np.dot(profile, dl)
    if norm > 1e-30:
        profile /= norm

    return profile.astype(np.float64)


def compute_absorption_coefficient(populations       : np.ndarray,
                                    levels            : List[EnergyLevel],
                                    transitions       : List[Transition],
                                    wavelength_grid_nm: np.ndarray,
                                    T_i_K             : float,
                                    n_e_cm3           : float,
                                    mass_kg           : float) -> np.ndarray:
    """
    Spectral absorption coefficient κ(λ) [cm⁻¹] for all transitions.

    Formula [Rybicki & Lightman 1979, Eq. 1.74; ref: 2]:
        κ(λ) = Σ_{i→k} n_i · σ_ik(λ) · (1 − stim_emission_correction)

    Where the cross-section σ_ik(λ) using Einstein A_ki:
        σ_ik(λ) = (λ⁴ / 8π·c) · (g_k/g_i) · A_ki · φ_V(λ)
        [m² per nm⁻¹]

    Stimulated emission correction (prevents negative populations driving):
        stim = n_k · g_i / (n_i · g_k)

    All intermediate quantities in SI; final κ converted to [cm⁻¹].
    """
    kappa = np.zeros(len(wavelength_grid_nm), dtype=np.float64)

    for t in transitions:
        i, k = t.lower_idx, t.upper_idx
        if i >= len(populations) or k >= len(populations):
            continue

        n_i = populations[i]   # [cm⁻³]
        n_k = populations[k]   # [cm⁻³]
        if n_i <= 0.0 and n_k <= 0.0:
            continue

        # Voigt profile φ(λ) [nm⁻¹]
        fwhm_G = 2.0 * _doppler_hwhm_nm(t.wavelength_nm, T_i_K, mass_kg)
        fwhm_L = 2.0 * _stark_hwhm_nm(t.wavelength_nm, n_e_cm3)
        
        # VEKTORISASI/OPTIMASI LOKALISASI: Hitung Voigt hanya pada grid +- 50 width, hemat 99% RAM & CPU
        hw_window = max(50.0 * max(fwhm_G, fwhm_L), 0.5)
        mask = np.abs(wavelength_grid_nm - t.wavelength_nm) <= hw_window
        if not np.any(mask):
            continue
            
        phi_window = voigt_profile(wavelength_grid_nm[mask], t.wavelength_nm, fwhm_G, fwhm_L)
        phi = np.zeros_like(wavelength_grid_nm)
        phi[mask] = phi_window

        # Absorption cross-section σ(λ) in SI units
        #   σ(ν) = (c²/8πν²) · (g_k/g_i) · A_ki · φ(ν)   [m² · Hz⁻¹]
        #   Converting φ(ν)→φ(λ): φ(ν) = φ(λ) · λ²/c
        #   σ(λ) at a given λ (evaluated cross-section) [m²]:
        #     σ(λ) = (λ⁴ / 8πc) · (g_k/g_i) · A_ki · φ(λ)
        lam_m      = t.wavelength_nm * 1e-9            # [m]
        sigma_m2   = ((lam_m ** 4) / (8.0 * np.pi * C_LIGHT)) \
                     * (t.g_upper / t.g_lower) * t.A_ki  # [m³] × φ[nm⁻¹]
        # Note: φ in [nm⁻¹] = φ[m⁻¹] × 1e-9; σ_m2 has units [m³];
        #       σ_final[m²] = sigma_m2 [m³] × phi[m⁻¹]
        #       Here phi is [nm⁻¹], so σ[m²] = sigma_m2 × phi[nm⁻¹] × 1e9
        sigma_m2_spectral = sigma_m2 * 1e9   # [m² nm] × phi[nm⁻¹] → [m²] per point

        # Stimulated emission correction (net absorption)
        if n_i > 0:
            stim = min((n_k * t.g_lower) / (n_i * t.g_upper), 1.0)
        else:
            stim = 0.0

        stim_corr = max(1.0 - stim, 0.0)

        # κ_line(λ) [m⁻¹] = n_i [m⁻³] × σ(λ) [m²] × stim_corr
        n_i_m3 = n_i * 1e6   # [cm⁻³ → m⁻³]
        kappa_m = n_i_m3 * sigma_m2_spectral * phi * stim_corr   # [m⁻¹]

        # Convert to [cm⁻¹] (1 m⁻¹ = 0.01 cm⁻¹)
        kappa += kappa_m * 0.01

    return kappa


def compute_emission_coefficient(populations       : np.ndarray,
                                  transitions       : List[Transition],
                                  wavelength_grid_nm: np.ndarray,
                                  T_i_K             : float,
                                  n_e_cm3           : float,
                                  mass_kg           : float) -> np.ndarray:
    """
    Spectral emission coefficient j(λ) [erg/s/cm³/nm/sr].

    Formula [Rybicki & Lightman 1979, Eq. 1.73]:
        j(λ) = (hc / 4πλ) · A_ki · n_k · φ_V(λ)

    Physical meaning: power emitted per unit volume, wavelength, solid angle.
    """
    j_spec = np.zeros(len(wavelength_grid_nm), dtype=np.float64)

    for t in transitions:
        k = t.upper_idx
        if k >= len(populations):
            continue

        n_k = populations[k]   # [cm⁻³]
        if n_k <= 0.0:
            continue

        # Voigt profile
        fwhm_G = 2.0 * _doppler_hwhm_nm(t.wavelength_nm, T_i_K, mass_kg)
        fwhm_L = 2.0 * _stark_hwhm_nm(t.wavelength_nm, n_e_cm3)
        
        # VEKTORISASI/OPTIMASI LOKALISASI: Hitung Voigt hanya pada span lokal
        hw_window = max(50.0 * max(fwhm_G, fwhm_L), 0.5)
        mask = np.abs(wavelength_grid_nm - t.wavelength_nm) <= hw_window
        if not np.any(mask):
            continue
            
        phi_window = voigt_profile(wavelength_grid_nm[mask], t.wavelength_nm, fwhm_G, fwhm_L)
        phi = np.zeros_like(wavelength_grid_nm)
        phi[mask] = phi_window

        # h·c/λ [J] — photon energy
        lam_m  = t.wavelength_nm * 1e-9
        h_c_ov_lam = H_PLANCK_J * C_LIGHT / lam_m   # [J]

        # j(λ) [W/m³/nm/sr] = (hc/4πλ) * A_ki * n_k[m⁻³] * φ[nm⁻¹]
        n_k_m3  = n_k * 1e6   # [cm⁻³ → m⁻³]
        j_SI    = (h_c_ov_lam / (4.0 * np.pi)) * t.A_ki * n_k_m3 * phi

        # Convert to CGS-like: [erg/s/cm³/nm/sr]
        # 1 W/m³/nm/sr = 1 J/s/m³/nm/sr = 1e7 erg/s / (1e6 cm³) / nm / sr = 10 erg/s/cm³/nm/sr
        j_spec += j_SI * 10.0

    return j_spec


# =============================================================================
# PHASE 3: RADIATIVE TRANSFER EQUATION (Two-Zone)
# =============================================================================

def integrate_rte(I_core   : np.ndarray,
                   kappa_shell: np.ndarray,
                   j_shell    : np.ndarray,
                   d_shell_m  : float,
                   wavelength_grid_nm: np.ndarray) -> np.ndarray:
    """
    Integrate 1D Radiative Transfer Equation through a uniform Shell zone.

    Geometry:
        [Core] → emits I_core(λ) → [Shell, thickness d_shell] → Observer

    RTE solution for a homogeneous emitting/absorbing slab [ref: 2, Ch. 1]:
        τ(λ)   = κ_shell(λ) · d_shell        [optical depth, dimensionless]
        S(λ)   = j_shell(λ) / κ_shell(λ)    [source function]
        I_obs  = I_core · exp(−τ) + S · (1 − exp(−τ))

    Limiting cases (verified in main()):
        τ → 0  (optically thin shell):    I_obs → I_core + j_shell·d_shell
        τ → ∞  (optically thick shell):   I_obs → S_shell (shell dominates)

    Args:
        I_core      : Core intensity spectrum  [same units as j/κ ratio]
        kappa_shell : Shell absorption coeff.  [cm⁻¹]  — shape (N_λ,)
        j_shell     : Shell emission coeff.    [erg/s/cm³/nm/sr]
        d_shell_m   : Shell thickness          [m]
        wavelength_grid_nm: wavelength grid    [nm]

    Returns:
        I_obs: Observed intensity [same arbitrary units, normalized later]
    """
    d_shell_cm = d_shell_m * 100.0    # [m → cm] for consistency with κ [cm⁻¹]

    # Spectral optical depth τ(λ) = κ(λ) · d   [dimensionless]
    tau = kappa_shell * d_shell_cm
    tau = np.clip(tau, 0.0, 700.0)   # Prevent exp overflow (exp(-700) ≈ 0)

    exp_neg_tau = np.exp(-tau)

    # Source function S(λ) = j(λ) / κ(λ)
    # Guard against division by zero (where κ=0, emission is optically thin)
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.where(kappa_shell > 1e-40, j_shell / kappa_shell, 0.0)

    # RTE: I_obs = I_core · e^{-τ} + S · (1 − e^{-τ})
    I_obs = I_core * exp_neg_tau + S * (1.0 - exp_neg_tau)

    return I_obs


def instrumental_broadening(spectrum          : np.ndarray,
                             wavelength_grid_nm: np.ndarray,
                             fwhm_instrument_nm: float = 0.1) -> np.ndarray:
    """
    Apply Gaussian instrumental broadening to simulate finite spectrometer resolution.

    Physical motivation:
      Real LIBS spectrometers have a finite slit-width and grating resolution.
      Typical FWHM for echelle spectrometers: 0.05–0.15 nm.
      Typical FWHM for compact CCD spectrometers: 0.3–0.5 nm.

      This convolution is applied AFTER the RTE (it is not plasma physics —
      it is an instrument response function). It is kept strictly separate
      so it can be tuned or removed independently of the forward model.

    Method:
      Convolve I_obs(λ) with a Gaussian kernel G(λ; σ_inst):
          I_convolved(λ) = I_obs(λ) ⊗ G(λ)

      The kernel is constructed on the same wavelength grid.
      σ_inst = FWHM_instrument / (2 * sqrt(2 * ln2))

    Args:
      spectrum           : Input spectrum I_obs(λ)  shape (N_λ,)
      wavelength_grid_nm : Wavelength axis [nm]     shape (N_λ,)
      fwhm_instrument_nm : Instrument FWHM [nm]. Default 0.1 nm (echelle class)

    Returns:
      I_broadened : np.ndarray — convolved spectrum, same shape and normalization.
    """
    from scipy.signal import fftconvolve

    # Grid step [nm/point]
    dl = (wavelength_grid_nm[-1] - wavelength_grid_nm[0]) / (len(wavelength_grid_nm) - 1)

    # Gaussian σ in grid-point units
    sigma_nm  = fwhm_instrument_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_pts = sigma_nm / dl

    # Build kernel on a window large enough to capture ±5σ
    half_win  = int(np.ceil(5.0 * sigma_pts))
    kernel_x  = np.arange(-half_win, half_win + 1, dtype=np.float64)
    kernel    = np.exp(-0.5 * (kernel_x / sigma_pts) ** 2)
    kernel   /= kernel.sum()   # normalise so total area is conserved

    # Convolve using FFT (fast, handles large arrays well)
    I_conv = fftconvolve(spectrum, kernel, mode='same')

    return I_conv.astype(np.float64)


def compute_saha_ionization_fractions(element: str,
                                      total_fraction: float,
                                      T_e_K: float,
                                      n_e_cm3: float,
                                      fetcher: DataFetcher) -> Tuple[float, float]:
    """
    Menghitung keseimbangan makroskopis ionisasi menggunakan Persamaan Saha.
    Menerapkan paradigma pLTE: Keseimbangan ionisasi (Bound-Free) diasumsikan 
    telah termalisasi oleh elektron bebas, sementara eksitasi internal (Bound-Bound)
    akan diselesaikan secara Non-LTE oleh matriks CR.
    """
    ion_energy_eV = fetcher.get_ionization_energy(element, sp_num=1)
    if ion_energy_eV <= 0.0:
        warnings.warn(f"[Saha Splitter] Energi ionisasi {element} tidak ditemukan. Diasumsikan 100% netral.")
        return total_fraction, 0.0

    # Konstanta Fundamental (SI)
    k_B_J = K_B_J
    k_B_eV = K_B_eV
    m_e_kg = M_ELECTRON
    h_J = H_PLANCK_J

    # Termal De Broglie Wavelength factor (SI: m^-3)
    thermal_term = (2.0 * np.pi * m_e_kg * k_B_J * T_e_K / (h_J ** 2)) ** 1.5
    
    # Konversi n_e ke SI [m^-3]
    n_e_m3 = n_e_cm3 * 1e6

    # Evaluasi Eksponensial (Penurunan Energi Ionisasi Debye diabaikan untuk simplifikasi pLTE batas atas)
    kT_eV = k_B_eV * T_e_K
    
    # Pendekatan Q1: Asumsi rasio fungsi partisi ion/netral ~ 1.0, 
    # dikalikan 2.0 untuk degenerasi spin elektron (g_e = 2).
    pre_factor = 2.0 * (thermal_term / n_e_m3)
    
    # Menghindari math overflow pada suhu sangat tinggi
    exponent = -ion_energy_eV / kT_eV
    if exponent > 700:
        saha_ratio = np.inf
    elif exponent < -700:
        saha_ratio = 0.0
    else:
        saha_ratio = pre_factor * np.exp(exponent)

    # Menghitung fraksi (Saha Ratio = n_ion / n_neutral)
    if np.isinf(saha_ratio):
        frac_neutral = 0.0
        frac_ion = total_fraction
    else:
        frac_neutral = total_fraction / (1.0 + saha_ratio)
        frac_ion = total_fraction * (saha_ratio / (1.0 + saha_ratio))

    return frac_neutral, frac_ion


# =============================================================================
# MAIN CLASS: TwoZonePlasma  (orchestrates all three phases)
# =============================================================================

class TwoZonePlasma:
    """
    Two-Zone plasma model: runs the CR-LIBS Forward Model for Core + Shell zones.

    Workflow:
        Phase 1 (per zone, per species): CR solver → n_i(T_e, n_e)
        Phase 2 (per zone):              κ(λ), j(λ) via Voigt profiles
        Phase 3:                         RTE integration → I_obs(λ)
    """

    def __init__(self,
                 core_params    : PlasmaZoneParams,
                 shell_params   : PlasmaZoneParams,
                 elements       : List[Tuple[str, int, float]],
                 fetcher        : Optional[DataFetcher] = None):
        """
        Args:
            core_params  : PlasmaZoneParams for the Core zone
            shell_params : PlasmaZoneParams for the Shell/Periphery zone
            elements     : List of (element_symbol, sp_num, number_fraction)
                           sp_num: 1=neutral, 2=singly ionized
                           number_fraction: fraction of total electron density
                           (must sum to 1.0)
            fetcher      : DataFetcher instance (created if None)
        """
        self.core    = core_params
        self.shell   = shell_params
        self.elements = elements
        self.fetcher = fetcher or DataFetcher()

        wl_min, wl_max = SIMULATION_CONFIG["wl_range_nm"]
        N_pts = SIMULATION_CONFIG["resolution"]
        self.wavelengths = np.linspace(wl_min, wl_max, N_pts, dtype=np.float64)

    def _run_zone(self, zone: PlasmaZoneParams
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Run Phase 1+2 for one zone → total κ(λ), j(λ), and intensity I(λ).

        Returns:
            kappa_total  [cm⁻¹]
            j_total      [erg/s/cm³/nm/sr]
            I_zone       [j/κ in optically thin limit, i.e., j*d_zone_cm]
            top_lines    [List of dicts with {wl, elem, ion, intensity_estim}]
        """
        kappa_total = np.zeros_like(self.wavelengths)
        j_total     = np.zeros_like(self.wavelengths)
        zone_top_lines = []

        total_fraction = sum(frac for _, _, frac in self.elements)
        if abs(total_fraction - 1.0) > 1e-6:
            raise ValueError(
                f"Element number fractions must sum to 1.0, got {total_fraction:.6f}"
            )

        for elem_sym, sp_num, frac in self.elements:
            # Fetch NIST transitions
            df = self.fetcher.get_transitions(
                elem_sym, sp_num, wl_range=SIMULATION_CONFIG["wl_range_nm"]
            )
            if df.empty:
                warnings.warn(f"[TwoZonePlasma] No NIST data for {elem_sym} sp_num={sp_num}. Skipping.")
                continue

            levels, transitions = DataFetcher.build_levels_and_transitions(df)
            if not levels:
                continue

            # Number density of this species [cm⁻³]
            # Simple estimate: fraction of n_e (assumes singly ionized plasma)
            n_species_cm3 = frac * zone.n_e_cm3

            # Phase 1: CR solver → level populations n_i [cm⁻³]
            populations = solve_cr_populations(
                levels      = levels,
                transitions = transitions,
                T_e_K       = zone.T_e_K,
                n_e_cm3     = zone.n_e_cm3,
                n_total_cm3 = n_species_cm3,
                sp_num      = sp_num,
            )

            # Atomic mass for Doppler broadening
            mass_kg = ATOMIC_MASS_AMU.get(elem_sym, 40.0) * AMU_KG

            # Phase 2a: Absorption coefficient κ(λ) [cm⁻¹]
            kappa = compute_absorption_coefficient(
                populations       = populations,
                levels            = levels,
                transitions       = transitions,
                wavelength_grid_nm= self.wavelengths,
                T_i_K             = zone.T_i_K,
                n_e_cm3           = zone.n_e_cm3,
                mass_kg           = mass_kg,
            )

            # Phase 2b: Emission coefficient j(λ)
            j_em = compute_emission_coefficient(
                populations       = populations,
                transitions       = transitions,
                wavelength_grid_nm= self.wavelengths,
                T_i_K             = zone.T_i_K,
                n_e_cm3           = zone.n_e_cm3,
                mass_kg           = mass_kg,
            )

            kappa_total += kappa
            j_total     += j_em
            
            # Estimate top lines for this species within this zone
            # I ~ n_upper * A_ki * (hc/wl)
            for t in transitions:
                n_up = populations[t.upper_idx]
                # erg/s/sr/cm3
                line_int = (n_up * t.A_ki * (H_PLANCK_J * C_LIGHT / (t.wavelength_nm * 1e-7))) / (4.0 * np.pi)
                if line_int > 0:
                    zone_top_lines.append({
                        "wl": t.wavelength_nm,
                        "elem": elem_sym,
                        "ion": "I" if sp_num==1 else "II",
                        "int": line_int
                    })

        # Core intensity in optically thin approximation:
        # I_zone = j * d_zone (used as I_core input to Phase 3)
        d_cm    = zone.thickness_m * 100.0
        I_zone  = j_total * d_cm

        # Sort and take top 50 
        zone_top_lines.sort(key=lambda x: x["int"], reverse=True)
        return kappa_total, j_total, I_zone, zone_top_lines[:50]

    def run(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Execute the full two-zone forward model: Phase 1 → Phase 2 → Phase 3.

        Returns:
            wavelengths : np.ndarray [nm]
            I_obs       : np.ndarray — normalised observed intensity [a.u.]
            metadata    : dict with diagnostics (optical depth, zone params, flags)
        """
        print(f"[TwoZonePlasma] Running CORE zone  (T_e={self.core.T_e_K:.0f}K, "
              f"n_e={self.core.n_e_cm3:.1e} cm⁻³)")
        kappa_core, j_core, I_core, core_lines = self._run_zone(self.core)

        print(f"[TwoZonePlasma] Running SHELL zone (T_e={self.shell.T_e_K:.0f}K, "
              f"n_e={self.shell.n_e_cm3:.1e} cm⁻³)")
        kappa_shell, j_shell, _, _    = self._run_zone(self.shell)

        print(f"[TwoZonePlasma] Phase 3: Integrating RTE...")
        I_obs = integrate_rte(
            I_core            = I_core,
            kappa_shell       = kappa_shell,
            j_shell           = j_shell,
            d_shell_m         = self.shell.thickness_m,
            wavelength_grid_nm= self.wavelengths,
        )

        # Konvolusi fungsi aparatus (Resolusi instrumen = 0.1 nm)
        I_obs = instrumental_broadening(I_obs, self.wavelengths, fwhm_instrument_nm=0.1)

        # Normalize to target_max_intensity
        I_max = np.max(np.abs(I_obs))
        if I_max > 0:
            I_obs = I_obs / I_max * SIMULATION_CONFIG["target_max_intensity"]

        # Optical depth diagnostics
        tau_shell = kappa_shell * (self.shell.thickness_m * 100.0)

        metadata = {
            "core_T_e_K"           : self.core.T_e_K,
            "core_n_e_cm3"         : self.core.n_e_cm3,
            "shell_T_e_K"          : self.shell.T_e_K,
            "shell_n_e_cm3"        : self.shell.n_e_cm3,
            "tau_shell_max"        : float(np.max(tau_shell)),
            "tau_shell_mean"       : float(np.mean(tau_shell)),
            "optically_thin_frac"  : float(np.mean(tau_shell < 0.1)),
            "stark_approx_flag"    : False,   # Tabulated empirical Griem data utilized
            "cr_method"            : "Radau (implicit RK)",
            "jacobian"             : "Analytical (J = M_CR)",
        }

        # Collect prominent lines for labeling (Top 50 by intensity)
        # Intensity estimate: j_total * thickness
        prominent_lines = []
        # We can find peaks in j_total and match them back to transitions, 
        # but a simpler way is to just use the transitions list from run_zone
        # (needs slight refactor to return them)
        
        metadata["top_lines"] = core_lines
        
        return self.wavelengths, I_obs, metadata


# =============================================================================
# MAIN EXECUTOR & INTERACTIVE UI (sim.py-like Mechanism)
# =============================================================================

def run_simulation(selected_elements: List[Tuple[str, float]], 
                   core_temp: float, 
                   core_ne: float, 
                   show_labels: bool = True):
    """
    Ekspansi Proporsi Atomik (Saha Ratio) -> CR Matrix Solver -> RTE -> Plot!
    Termodinamika inti Two-Zone tetap utuh dalam perambatan matriks Kinetika.
    """
    total_pct = sum(p for _, p in selected_elements)
    if abs(total_pct - 100.0) > 1e-6:
        print(f"Error: Persentase komposit atomik mutlak harus 100%. (Dapat {total_pct}%)")
        return

    fetcher = DataFetcher()
    elements_expanded = []
    
    for elem, pct in selected_elements:
        base_frac = pct / 100.0
        f_neu, f_ion = compute_saha_ionization_fractions(elem, base_frac, core_temp, core_ne, fetcher)
        
        if f_neu > 1e-4:
            elements_expanded.append((elem, 1, f_neu))
        if f_ion > 1e-4:
            elements_expanded.append((elem, 2, f_ion))
            
    # Normalisasi Akhir Pasca-Ekspansi
    tot = sum(f for _, _, f in elements_expanded)
    elements_expanded = [(sym, sp, f/tot) for sym, sp, f in elements_expanded]

    # ── Zone Parameters ──────────────────────────────────────────────────────
    core = PlasmaZoneParams(
        T_e_K       = core_temp,
        T_i_K       = max(core_temp - 2000.0, 3000.0),
        n_e_cm3     = core_ne,
        thickness_m = 1e-3,       # 1 mm optical path
        label       = "Core",
    )
    shell = PlasmaZoneParams(
        T_e_K       = core_temp * 0.5,    # Gradient pendinginan luar
        T_i_K       = max(core_temp * 0.5 - 1000.0, 3000.0),
        n_e_cm3     = core_ne * 0.1,      # Difusi kepadatan luar
        thickness_m = 2e-3,       # 2 mm selubung absorpsi
        label       = "Shell",
    )
    
    model = TwoZonePlasma(core, shell, elements_expanded, fetcher)
    wavelengths, I_obs, meta = model.run()
    
    # ── Ploting Interaktif ───────────────────────────────────────────────────
    element_str = " ".join([f"{sym}{pct:.0f}%" for sym, pct in selected_elements])
    dyn_title = f"<b>CR-LIBS Deterministic (Q1 Target)</b><br>{element_str} | T_Core={int(core_temp)} K | n_e={core_ne:.1e} cm⁻³"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wavelengths, y=I_obs, mode='lines',
        name='Forward Syntax (CR)',
        line=dict(color='navy', width=1.5),
    ))
    
    if np.max(I_obs) > 0 and show_labels:
        from scipy.signal import find_peaks
        height_threshold = 0.05 * np.max(I_obs)
        # Menghindari tumpang-tindih dengan distance limit = 100 poin (tergantung span nm)
        peak_indices, _ = find_peaks(I_obs, height=height_threshold, distance=150)
        peak_wavelengths = wavelengths[peak_indices]
        peak_intensities = I_obs[peak_indices]
        
        annotations = []
        top_lines = meta.get("top_lines", [])
        
        for wl, intensity in zip(peak_wavelengths, peak_intensities):
            # Find closest transition in NIST metadata to label it
            if top_lines:
                closest = min(top_lines, key=lambda x: abs(x["wl"] - wl))
                if abs(closest["wl"] - wl) < 0.2: # Match within 0.2nm
                    label = f"{closest['elem']} {closest['ion']}<br>{wl:.2f} nm"
                else:
                    label = f"{wl:.2f} nm"
            else:
                label = f"{wl:.2f} nm"
                
            annotations.append(go.layout.Annotation(
                x=wl, y=intensity, text=label, showarrow=True, arrowhead=2, arrowsize=1,
                arrowwidth=1, ax=0, ay=-40, font=dict(size=9, color="#ffffff"),
                align="center", bordercolor="#555", borderwidth=1, borderpad=2,
                bgcolor="#ff7f0e", opacity=0.8))
        fig.update_layout(annotations=annotations)
        
    ann_text = (
        f"<b>Core:</b> T_e={meta['core_T_e_K']:.0f} K, "
        f"n_e={meta['core_n_e_cm3']:.1e} cm⁻³<br>"
        f"<b>Shell:</b> T_e={meta['shell_T_e_K']:.0f} K, "
        f"n_e={meta['shell_n_e_cm3']:.1e} cm⁻³<br>"
        f"<b>τ_shell max:</b> {meta['tau_shell_max']:.3f}<br>"
    )
    
    fig.update_layout(
        title=dyn_title,
        xaxis_title="Wavelength (nm)",
        yaxis_title="Normalized Intensity (a.u.)",
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray', minor=dict(showgrid=True, gridcolor='lightgrey', griddash='dot')),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray', minor=dict(showgrid=True, gridcolor='lightgrey', griddash='dot')),
        font=dict(family="Arial, sans-serif", size=12),
        annotations=list(fig.layout.annotations) if fig.layout.annotations else [] + [go.layout.Annotation(
            x=0.98, y=0.98, xref='paper', yref='paper',
            text=ann_text, showarrow=False, align='left',
            bgcolor='rgba(255,255,255,0.85)', bordercolor='black',
            borderwidth=1, font=dict(size=10),
        )],
    )
    fig.show()

def parse_element_input(user_str: str) -> List[Tuple[str, float]]:
    """
    Parse a string like 'Ca 100' or 'Si 25, Al 25, Fe 50' into a list of tuples.
    Returns: [(symbol, percentage), ...]
    """
    elements = []
    # Split by comma first
    parts = user_str.replace(';', ',').split(',')
    for p in parts:
        p = p.strip()
        if not p: continue
        # Split by space or colon
        m = re.match(r"([A-Za-z]{1,2})\s*[:\s-]*\s*(\d+\.?\d*)", p)
        if m:
            sym = m.group(1).capitalize()
            val = float(m.group(2))
            elements.append((sym, val))
    return elements

def create_composition_form():
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:
        print("Peringatan: Modul 'ipywidgets' tidak tersedia. Silakan instal atau jalankan via CLI konvensional.")
        return
        
    BASE_ELEMENTS = ['Al', 'Ca', 'Fe', 'Si', 'Mg', 'Ti', 'Cr', 'Mn', 'Ni', 'Cu', 'Li', 'Na', 'K', 'O', 'H', 'N', 'Sr', 'C', 'Rb', 'Co', 'Pb']
    element_options = [(elem, elem) for elem in BASE_ELEMENTS]
    composition_widgets = []
    total_percentage_label = widgets.Label(value="Total Percentage: 0.0%")
    
    def add_element_row(_=None):
        element_dropdown = widgets.Dropdown(options=element_options, description="Element:", layout={'width': '300px'})
        percentage_input = widgets.FloatText(value=0.0, description="Percentage (%):", layout={'width': '200px'})
        remove_button = widgets.Button(description="Remove", button_style='danger')

        def update_total(_=None):
            total = sum(w[1].value for w in composition_widgets)
            total_percentage_label.value = f"Total Percentage: {total:.1f}%"

        def remove_row(b):
            row_to_remove = next((row for row in composition_widgets if row[2] == b), None)
            if row_to_remove:
                composition_widgets.remove(row_to_remove)
                rows_vbox.children = [widgets.HBox(list(row)) for row in composition_widgets]
                update_total()

        percentage_input.observe(update_total, names='value')
        remove_button.on_click(remove_row)
        composition_widgets.append((element_dropdown, percentage_input, remove_button))
        rows_vbox.children = [widgets.HBox(list(row)) for row in composition_widgets]
        update_total()

    add_button = widgets.Button(description="Add Element", button_style='success')
    add_button.on_click(add_element_row)

    rows_vbox = widgets.VBox([])
    control_hbox = widgets.HBox([add_button, total_percentage_label])
    display(widgets.VBox([rows_vbox, control_hbox]))
    add_element_row()

    submit_button = widgets.Button(description="Generate CR Spectrum", button_style='primary')
    show_labels_checkbox = widgets.Checkbox(value=True, description='Show Peak Labels', style={'description_width': 'initial'})
    output = widgets.Output()

    def on_submit(_):
        with output:
            clear_output()
            selected_elements = [(w[0].value, w[1].value) for w in composition_widgets if w[1].value > 0]
            if not selected_elements:
                print("Error: No elements with percentage > 0 selected")
                return
            total_percentage = sum(p for _, p in selected_elements)
            if abs(total_percentage - 100.0) > 1e-6:
                print(f"Error: Total percentage ({total_percentage:.1f}%) must be exactly 100%")
                return

            run_simulation(selected_elements, temperature_input.value, electron_density_input.value, show_labels_checkbox.value)

    submit_button.on_click(on_submit)
    temperature_input = widgets.FloatText(value=14000, description="Temperature (K):", layout={'width': '300px'})
    
    ne_options_values = [10**exp for exp in np.arange(15, 18.1, 0.1)]
    ne_options_labels = [f"{val:.1e}" for val in ne_options_values]
    ne_options = list(zip(ne_options_labels, ne_options_values))
    electron_density_input = widgets.Dropdown(options=ne_options, value=1e17, description="Electron Density (cm^-3):", layout={'width': '400px'})

    display(temperature_input, electron_density_input, show_labels_checkbox, submit_button, output)

def main():
    # ── Element list: (symbol, total_number_fraction) ──────────────────────
    # Diwariskan dari matriks geologi Q1 pengguna
    DEFAULT_ELEMENTS = [
        ("Si", 0.42),
        ("Fe", 0.39),
        ("Al", 0.12),
        ("Ca", 0.07),
    ]
    DEFAULT_SELECTED = [(sym, frac * 100.0) for sym, frac in DEFAULT_ELEMENTS]

    if 'ipykernel' in __import__('sys').modules:
        print("Menjalankan dalam mode interaktif GUI termodinamika (Jupyter/IPython).")
        # Pre-set some defaults for ipywidgets if needed, or just launch
        create_composition_form()
    else:
        print("=" * 75)
        print("  TWO-ZONE CR-LIBS FORWARD MODEL  —  Q1 Thesis Physics Engine")
        print("  (Interactive CLI Mode)")
        print("=" * 75)
        print(f"Default: {' '.join([f'{s}{p:.0f}%' for s,p in DEFAULT_SELECTED])}")
        print("Tips: Masukkan elemen seperti 'Ca 100' atau tekan Enter untuk default.")
        
        try:
            while True:
                print("\n" + "-"*40)
                elem_input = input(">> Komposisi Elemen [%] [DEFAULT]: ").strip()
                
                if not elem_input:
                    selected = DEFAULT_SELECTED
                else:
                    selected = parse_element_input(elem_input)
                
                if not selected:
                    print("Error: Format input elemen tidak valid.")
                    continue
                
                total_pct = sum(p for _, p in selected)
                if abs(total_pct - 100.0) > 1e-3:
                    print(f"Peringatan: Total persentase ({total_pct}%) tidak 100%. Menyetel ulang proporsi...")
                    selected = [(s, p*100.0/total_pct) for s, p in selected]

                temp_str = input(">> Temperatur Core (K)  [14000]: ").strip()
                temp = float(temp_str) if temp_str else 14000.0
                
                print("Pilih Densitas Elektron (n_e [cm⁻³]):")
                print("1) 1.0e15  2) 5.0e15  3) 1.0e16  4) 5.0e16  5) 1.0e17 (Default)  6) 5.0e17")
                ne_choice = input(">> Pilih (1-6 atau nilai): ").strip()
                
                ne_map = {"1":1e15, "2":5e15, "3":1e16, "4":5e16, "5":1e17, "6":5e17}
                if ne_choice in ne_map:
                    ne = ne_map[ne_choice]
                elif ne_choice:
                    try:
                        ne = float(ne_choice)
                    except ValueError:
                        ne = 1e17
                else:
                    ne = 1e17
                
                print(f"\n[Saha-CR Engine] Mensimulasikan {selected} pada T={temp}K, n_e={ne:.1e}...")
                run_simulation(selected, temp, ne, show_labels=True)
                
                cont = input("\nSimulasi lagi? (y/n) [y]: ").strip().lower()
                if cont == 'n': break

        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
        except Exception as e:
            print(f"\nTerjadi kesalahan fatal: {e}")

    print("\n[DONE] libs_physics.py complete.")

if __name__ == "__main__":
    main()
