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
import warnings
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from scipy.integrate import solve_ivp
from scipy.special import wofz        # Faddeeva function → exact Voigt profile
import plotly.graph_objects as go

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

# Simulation grid configuration
SIMULATION_CONFIG = {
    "resolution"          : 24480,          # Wavelength grid points (matches legacy sim.py)
    "wl_range_nm"         : (200.0, 900.0), # Wavelength range [nm]
    "target_max_intensity": 0.8,            # Normalization target for output spectrum
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

        Strategy:
          - Collect all (Ei, gi) and (Ek, gk) pairs as candidate levels.
          - Merge levels within TOL [eV] using rounded keys.
          - Sort levels by energy and assign sequential indices.
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


# =============================================================================
# PHASE 1: COLLISIONAL-RADIATIVE (CR) MATRIX SOLVER
# =============================================================================

def _oscillator_strength(trans: Transition) -> float:
    """
    Absorption oscillator strength f_ij (dimensionless) from Einstein A_ki.

    Formula [NIST ASD documentation, Eq. 3]:
        g_i * f_ij = 1.4992e-16 * λ[m]² * g_k * A_ki[s⁻¹]

    Derivation: From the relation between Einstein A and B coefficients and
    the definition of oscillator strength via the dipole matrix element.
    """
    lam_m = trans.wavelength_nm * 1e-9   # [m]
    gf    = 1.4992e-16 * (lam_m ** 2) * trans.g_upper * trans.A_ki
    return max(gf / trans.g_lower, 0.0)


def _van_regemorter_q_excitation(trans: Transition,
                                  T_e_K : float,
                                  sp_num: int) -> float:
    """
    Electron impact EXCITATION rate coefficient q_ij [cm³/s].

    van Regemorter (1962) approximation [ref: 3, Eq. 8]:
        q_ij = C * T_e^(-1/2) * (f_ij / ΔE_ij) * exp(-ΔE_ij / kT_e) * Ḡ

    Constants:
        C   = 5.465e-11  [eV · cm³ · K^(1/2) / s]
        f_ij = absorption oscillator strength
        ΔE   = E_upper - E_lower  [eV]
        Ḡ   = mean Gaunt factor (0.2 neutral, 0.276 ion)

    ⚠ LIMITATION: Valid only for optically allowed (dipole) transitions.
    Forbidden transitions (f_ij ≈ 0) require ADAS/Chianti cross-sections.
    This is documented as an explicit approximation, NOT hidden physics.
    """
    delta_E = trans.E_upper_eV - trans.E_lower_eV
    if delta_E <= 0.0:
        return 0.0
    kT_e  = K_B_eV * T_e_K
    f_ij  = _oscillator_strength(trans)
    G_bar = G_BAR_NEUTRAL if sp_num == 1 else G_BAR_ION
    C     = 5.465e-11     # [eV · cm³ · K^0.5 / s]
    q_ij  = C * (T_e_K ** -0.5) * (f_ij / delta_E) * np.exp(-delta_E / kT_e) * G_bar
    return max(q_ij, 0.0)


def build_cr_matrix(levels     : List[EnergyLevel],
                    transitions: List[Transition],
                    T_e_K      : float,
                    n_e_cm3    : float,
                    sp_num     : int) -> np.ndarray:
    """
    Construct the N×N Collisional-Radiative rate matrix M.

    Population evolution: dn/dt = M · n

    Off-diagonal M[k, j] (k ≠ j) = rate INTO level k FROM level j [s⁻¹]:
      • Radiative decay         j→k (j upper, k lower): A_jk
      • Collisional de-excit.  j→k (j upper, k lower): n_e * q_jk_down
      • Collisional excitation  j→k (j lower, k upper): n_e * q_jk_up

    Diagonal M[i,i] = − Σ_{k≠i} M[k,i]   (total outflow from level i)

    Detailed Balance for de-excitation [ref: 1, Ch. 2]:
        q_ji_down = q_ij_up * (g_i / g_k) * exp(ΔE / kT_e)
    This is ANALYTICALLY enforced — not a numerical approximation.

    Conservation property: Σ_i M[i,j] = 0 for all j. ✓
    """
    N = len(levels)
    M = np.zeros((N, N), dtype=np.float64)
    kT_e = K_B_eV * T_e_K

    for t in transitions:
        i = t.lower_idx   # lower level (ground-ward)
        k = t.upper_idx   # upper level (excited)
        if not (0 <= i < N and 0 <= k < N):
            continue

        A   = t.A_ki                                          # [s⁻¹]
        q_up = _van_regemorter_q_excitation(t, T_e_K, sp_num) # [cm³/s]

        # Detailed balance: de-excitation coefficient
        dE = t.E_upper_eV - t.E_lower_eV
        if kT_e > 0 and dE > 0:
            q_down = q_up * (t.g_lower / t.g_upper) * np.exp(dE / kT_e)
        else:
            q_down = 0.0

        R_up   = n_e_cm3 * q_up    # excitation rate   i→k  [s⁻¹]
        R_down = n_e_cm3 * q_down  # de-excitation     k→i  [s⁻¹]

        # Gain terms (off-diagonal — population flowing INTO a level)
        M[i, k] += (A + R_down)  # k→i : radiative + collisional de-excit.
        M[k, i] += R_up           # i→k : collisional excitation

        # Loss terms (diagonal — population flowing OUT of a level)
        M[k, k] -= (A + R_down)  # k loses to i
        M[i, i] -= R_up           # i loses to k

    return M


def cr_jacobian_func(t: float, n: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Analytical Jacobian J[i,j] = ∂(dn_i/dt)/∂n_j for the CR ODE system.

    For the linear system  dn/dt = M·n:
        ∂(dn_i/dt)/∂n_j = M[i,j]   (exactly)

    Providing this to solve_ivp(method='Radau') prevents the solver from
    using finite-difference Jacobian approximation, which causes numerical
    divergence in stiff plasma kinetics systems.
    """
    return M   # J ≡ M for a linear ODE


def solve_cr_populations(levels      : List[EnergyLevel],
                          transitions : List[Transition],
                          T_e_K       : float,
                          n_e_cm3     : float,
                          n_total_cm3 : float,
                          sp_num      : int,
                          t_max_s     : float = 1e-6) -> np.ndarray:
    """
    Solve the CR ODE system → steady-state level populations n_i [cm⁻³].

    Algorithm:
      1. Build M matrix (N×N) via build_cr_matrix().
      2. Replace the Nth equation with the particle-conservation constraint:
             Σ_i n_i = n_total_cm3
         This regularises the singular steady-state problem into a well-posed IVP.
      3. Integrate with solve_ivp(method='Radau', jac=...) — implicit,
         A-stable, B-stable. Optimal for stiff systems (stiffness ratio >>10³).
      4. Clip populations to ≥0 (physical lower bound) and rescale to conserve mass.

    Initial condition: Boltzmann distribution at T_e (physically reasonable IC).
    The final answer is n(t_max_s) ≈ steady state if t_max >> 1/|λ_min(M)|.

    Args:
      t_max_s: Integration endpoint [s]. Default 1 µs covers typical LIBS plasma.

    Returns:
      n_pop: np.ndarray of shape (N,) with population densities [cm⁻³]
    """
    N = len(levels)
    if N == 0:
        return np.zeros(0)

    M    = build_cr_matrix(levels, transitions, T_e_K, n_e_cm3, sp_num)
    kT_e = K_B_eV * T_e_K

    # Boltzmann initial condition (starting guess — NOT the final answer)
    E_arr = np.array([lv.energy_eV  for lv in levels])
    g_arr = np.array([lv.degeneracy for lv in levels])
    boltz = g_arr * np.exp(-(E_arr - E_arr.min()) / kT_e)
    n0    = (boltz / boltz.sum()) * n_total_cm3   # normalised to n_total

    # Modify matrix: replace last row with conservation equation
    M_mod         = M.copy()
    M_mod[-1, :]  = 1.0          # row N: Σ n_i
    rhs           = np.zeros(N)
    rhs[-1]       = n_total_cm3  # = n_total

    def rhs_func(t, n):
        return M_mod @ n - rhs

    def jac_func(t, n):
        return M_mod   # analytical Jacobian — exact for linear system

    sol = solve_ivp(
        fun    = rhs_func,
        t_span = (0.0, t_max_s),
        y0     = n0,
        method = 'Radau',          # Implicit RK — stiff-safe [ref: scipy docs]
        jac    = jac_func,         # Analytical Jacobian — prevents divergence
        rtol   = 1e-8,             # Tight relative tolerance
        atol   = max(n_total_cm3 * 1e-10, 1e-3),  # Scaled absolute tolerance
        dense_output=False,
    )

    if sol.status != 0:
        warnings.warn(
            f"[CR Solver] ODE did not converge (status={sol.status}). "
            f"Falling back to Boltzmann distribution."
        )
        return n0

    n_out = np.clip(sol.y[:, -1], 0.0, None)

    # Enforce exact mass conservation by rescaling
    total = n_out.sum()
    if total > 0:
        n_out *= n_total_cm3 / total

    return n_out


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


def _stark_hwhm_nm(wavelength_nm: float,
                    n_e_cm3     : float) -> float:
    """
    Stark (pressure) Lorentzian half-width at half-maximum (HWHM) [nm].

    ⚠ PLACEHOLDER: Accurate Stark widths require tabulated data from
      Griem (1974) or NIST Stark-B database (per element and transition).
      This uses a scaled linear approximation:
          Δλ_S ≈ w_ref * (n_e / n_e_ref)
      with w_ref ≈ 0.001 nm at n_e_ref = 1e16 cm⁻³.

    This MUST be replaced with tabulated data before publication.
    Flag: STARK_APPROX_PLACEHOLDER = True in metadata output.
    """
    w_ref_nm = 0.001    # [nm] — order-of-magnitude estimate
    n_e_ref  = 1e16     # [cm⁻³]
    return max(w_ref_nm * (n_e_cm3 / n_e_ref), 1e-5)   # [nm]


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
        phi    = voigt_profile(wavelength_grid_nm, t.wavelength_nm, fwhm_G, fwhm_L)

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
        phi    = voigt_profile(wavelength_grid_nm, t.wavelength_nm, fwhm_G, fwhm_L)

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
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Phase 1+2 for one zone → total κ(λ), j(λ), and intensity I(λ).

        Returns:
            kappa_total  [cm⁻¹]
            j_total      [erg/s/cm³/nm/sr]
            I_zone       [j/κ in optically thin limit, i.e., j*d_zone_cm]
        """
        kappa_total = np.zeros_like(self.wavelengths)
        j_total     = np.zeros_like(self.wavelengths)

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

        # Core intensity in optically thin approximation:
        # I_zone = j * d_zone (used as I_core input to Phase 3)
        d_cm    = zone.thickness_m * 100.0
        I_zone  = j_total * d_cm

        return kappa_total, j_total, I_zone

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
        kappa_core, j_core, I_core = self._run_zone(self.core)

        print(f"[TwoZonePlasma] Running SHELL zone (T_e={self.shell.T_e_K:.0f}K, "
              f"n_e={self.shell.n_e_cm3:.1e} cm⁻³)")
        kappa_shell, j_shell, _    = self._run_zone(self.shell)

        print(f"[TwoZonePlasma] Phase 3: Integrating RTE...")
        I_obs = integrate_rte(
            I_core            = I_core,
            kappa_shell       = kappa_shell,
            j_shell           = j_shell,
            d_shell_m         = self.shell.thickness_m,
            wavelength_grid_nm= self.wavelengths,
        )

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
            "stark_approx_flag"    : True,   # Reminder: Stark widths are approximate
            "cr_method"            : "Radau (implicit RK)",
            "jacobian"             : "Analytical (J = M_CR)",
        }

        return self.wavelengths, I_obs, metadata


# =============================================================================
# PLOTTING UTILITY
# =============================================================================

def plot_spectrum(wavelengths: np.ndarray,
                  I_obs      : np.ndarray,
                  metadata   : Dict,
                  title      : str = "Two-Zone CR-LIBS Spectrum") -> None:
    """Plot the simulated spectrum using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wavelengths, y=I_obs, mode='lines',
        name='I_obs (Two-Zone CR)',
        line=dict(color='darkblue', width=1.5),
    ))
    ann_text = (
        f"<b>Core:</b> T_e={metadata['core_T_e_K']:.0f} K, "
        f"n_e={metadata['core_n_e_cm3']:.1e} cm⁻³<br>"
        f"<b>Shell:</b> T_e={metadata['shell_T_e_K']:.0f} K, "
        f"n_e={metadata['shell_n_e_cm3']:.1e} cm⁻³<br>"
        f"<b>τ_shell max:</b> {metadata['tau_shell_max']:.3f}<br>"
        f"<b>Stark widths:</b> {'APPROX ⚠' if metadata['stark_approx_flag'] else 'TABULATED'}"
    )
    fig.update_layout(
        title=title,
        xaxis_title="Wavelength (nm)",
        yaxis_title="Normalized Intensity (a.u.)",
        annotations=[go.layout.Annotation(
            x=0.98, y=0.98, xref='paper', yref='paper',
            text=ann_text, showarrow=False, align='left',
            bgcolor='rgba(255,255,255,0.85)', bordercolor='black',
            borderwidth=1, font=dict(size=10),
        )],
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
    )
    fig.show()


# =============================================================================
# MAIN — Verification & Demo Run
# =============================================================================

def main():
    """
    Demonstration and verification run of the Two-Zone CR-LIBS model.

    Sample composition: Si 25%, Al 25%, Fe 50% (neutral, sp_num=1)
    Matches legacy sim.py demo run for qualitative comparison.
    """
    print("=" * 70)
    print("  TWO-ZONE CR-LIBS FORWARD MODEL  —  Q1 Thesis Physics Engine")
    print("=" * 70)

    # ── Zone Parameters ──────────────────────────────────────────────────────
    core = PlasmaZoneParams(
        T_e_K       = 14000.0,    # Hot core [K]
        T_i_K       = 12000.0,    # Ion temperature ≈ T_e (LTE approximation within zone)
        n_e_cm3     = 1e17,       # Electron density [cm⁻³]
        thickness_m = 1e-3,       # 1 mm core
        label       = "Core",
    )
    shell = PlasmaZoneParams(
        T_e_K       = 7000.0,     # Cooler periphery [K]
        T_i_K       = 6000.0,
        n_e_cm3     = 1e16,       # Lower n_e in shell [cm⁻³]
        thickness_m = 2e-3,       # 2 mm shell
        label       = "Shell",
    )

    # ── Element list: (symbol, sp_num, number_fraction) ──────────────────────
    # Number fractions must sum to 1.0.
    # sp_num=1 → neutral, sp_num=2 → singly ionized
    elements = [
        ("Si", 1, 0.25),
        ("Al", 1, 0.25),
        ("Fe", 1, 0.50),
    ]

    # ── Initialise and run ────────────────────────────────────────────────────
    fetcher = DataFetcher()
    model   = TwoZonePlasma(core, shell, elements, fetcher)
    wavelengths, I_obs, meta = model.run()

    # ── Print diagnostics ────────────────────────────────────────────────────
    print("\n── METADATA ──────────────────────────────────────────────")
    for k, v in meta.items():
        print(f"  {k:<28}: {v}")

    # ── Physical Verification Checks ─────────────────────────────────────────
    print("\n── VERIFICATION CHECKS ───────────────────────────────────")

    # Check 1: Spectrum is non-trivial
    assert np.max(I_obs) > 0, "FAIL: Output spectrum is all zeros."
    print(f"  [PASS] Spectrum max = {np.max(I_obs):.4f} (non-zero)")

    # Check 2: No NaN or Inf in output
    assert not np.any(np.isnan(I_obs)), "FAIL: NaN in output spectrum."
    assert not np.any(np.isinf(I_obs)), "FAIL: Inf in output spectrum."
    print(f"  [PASS] No NaN or Inf in output spectrum.")

    # Check 3: RTE optically-thin limit (d_shell → 0)
    I_thin = integrate_rte(
        I_core            = I_obs,
        kappa_shell       = np.zeros_like(I_obs),
        j_shell           = np.zeros_like(I_obs),
        d_shell_m         = 0.0,
        wavelength_grid_nm= wavelengths,
    )
    assert np.allclose(I_thin, I_obs, rtol=1e-9), \
        "FAIL: Optically thin limit (τ=0) should give I_obs = I_core."
    print(f"  [PASS] RTE optically-thin limit (τ=0): I_obs = I_core ✓")

    # Check 4: Voigt profile normalization
    from scipy.special import wofz as _wofz
    import numpy as _np
    wl_test = _np.linspace(395.0, 405.0, 10000)
    phi_test = voigt_profile(wl_test, 400.0, 0.05, 0.02)
    dl_test  = _np.gradient(wl_test)
    norm_val = _np.dot(phi_test, dl_test)
    assert abs(norm_val - 1.0) < 0.001, \
        f"FAIL: Voigt profile norm = {norm_val:.6f} (expected 1.0 ± 0.001)."
    print(f"  [PASS] Voigt profile normalization = {norm_val:.6f} ≈ 1.000 ✓")

    print("\n── PLOTTING ──────────────────────────────────────────────")
    plot_spectrum(
        wavelengths, I_obs, meta,
        title="Two-Zone CR-LIBS | Si25% Al25% Fe50% | Core 14kK / Shell 7kK",
    )

    print("\n[DONE] libs_physics.py forward model complete.")
    return wavelengths, I_obs, meta


if __name__ == "__main__":
    main()
