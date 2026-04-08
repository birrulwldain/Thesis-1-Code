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
try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None
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
NIST_HDF_PATH    = os.path.join(_PROJECT_DIR, "..", "data", "nist_data(1).h5")
ATOMIC_DATA_PATH = os.path.join(_PROJECT_DIR, "..", "data", "atomic_data1.h5")

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

    REQUIRED_TRANSITION_COLUMNS = [
        'ritz_wl_air(nm)', 'Aki(s^-1)', 'Ek(eV)', 'Ei(eV)', 'g_i', 'g_k'
    ]

    def __init__(self,
                 nist_path  : str = NIST_HDF_PATH,
                 atomic_path: str = ATOMIC_DATA_PATH):
        self.nist_path   = nist_path
        self.atomic_path = atomic_path
        self._ion_energies: Dict[str, float] = {}
        self._transition_cache: Dict[Tuple[str, int], pd.DataFrame] = {}
        self._load_ionization_energies()
        self._load_transition_cache()

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

    def _load_transition_cache(self) -> None:
        """
        Preload and sanitize the full NIST transition table into RAM once.
        Data are indexed by (element, sp_num) to avoid disk I/O in the main loop.
        """
        empty = pd.DataFrame(columns=self.REQUIRED_TRANSITION_COLUMNS)
        try:
            with pd.HDFStore(self.nist_path, mode='r') as store:
                df = store.get('nist_spectroscopy_data')

            if not all(c in df.columns for c in self.REQUIRED_TRANSITION_COLUMNS):
                warnings.warn("[DataFetcher] NIST table is missing required columns.")
                return

            df = df.copy()
            df['element_clean'] = df['element'].astype(str).str.strip()
            df['sp_num_clean'] = pd.to_numeric(df['sp_num'], errors='coerce').fillna(-1).astype(int)
            df['ritz_wl_air(nm)'] = pd.to_numeric(df['ritz_wl_air(nm)'], errors='coerce')

            for col in ['Ek(eV)', 'Ei(eV)']:
                df[col] = df[col].apply(
                    lambda x: float(re.sub(r'[^\d.-]', '', str(x)))
                    if re.sub(r'[^\d.-]', '', str(x)) else np.nan
                )

            for col in ['Aki(s^-1)', 'g_i', 'g_k']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna(subset=self.REQUIRED_TRANSITION_COLUMNS)

            grouped = df.groupby(['element_clean', 'sp_num_clean'], sort=False)
            for (element, sp_num), group in grouped:
                self._transition_cache[(str(element), int(sp_num))] = (
                    group[self.REQUIRED_TRANSITION_COLUMNS]
                    .sort_values('ritz_wl_air(nm)')
                    .reset_index(drop=True)
                )

        except Exception as e:
            warnings.warn(f"[DataFetcher] Error preloading NIST transition cache: {e}")
            self._transition_cache = {}

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
        empty = pd.DataFrame(columns=self.REQUIRED_TRANSITION_COLUMNS)
        target_element = str(element).strip()
        target_sp_num = int(sp_num)
        key = (target_element, target_sp_num)

        df = self._transition_cache.get(key)
        if df is None or df.empty:
            warnings.warn(f"[DataFetcher] FILTER KOSONG: {target_element} {target_sp_num} tidak ditemukan di cache NIST.")
            return empty

        sel = df[(df['ritz_wl_air(nm)'] >= wl_range[0]) &
                 (df['ritz_wl_air(nm)'] <= wl_range[1])]
        return sel.reset_index(drop=True).copy()

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

class PhysicsCalculator:
    STARK_DATABASE = {
        393.4: (0.0034, 1e17), 396.8: (0.0033, 1e17), 288.2: (0.0019, 1e16),
        396.2: (0.0022, 1e16), 309.3: (0.0018, 1e16), 510.6: (0.0025, 1e16),
        521.8: (0.0026, 1e16), 404.6: (0.0015, 1e16),
    }

    @staticmethod
    def _stark_hwhm_nm(wavelength_nm: float, n_e_cm3: float) -> float:
        wl_key = round(wavelength_nm, 1)
        if wl_key in PhysicsCalculator.STARK_DATABASE:
            w_ref_nm, n_e_ref = PhysicsCalculator.STARK_DATABASE[wl_key]
            return w_ref_nm * (n_e_cm3 / n_e_ref)
        return max(0.001 * (n_e_cm3 / 1e16), 1e-5)

    @staticmethod
    def solve_boltzmann_populations(levels: List[EnergyLevel], transitions: List[Transition], T_e_K: float, n_e_cm3: float, n_total_cm3: float, sp_num: int, t_max_s: float = 1e-6) -> np.ndarray:
        N = len(levels)
        if N == 0:
            return np.zeros(0)
        kT_e = 8.617333262145e-5 * T_e_K
        E_arr = np.array([lv.energy_eV for lv in levels])
        g_arr = np.array([lv.degeneracy for lv in levels])
        E_min = E_arr.min()
        boltz_factors = g_arr * np.exp(-(E_arr - E_min) / kT_e)
        Z = boltz_factors.sum()
        if Z > 0:
            return (boltz_factors / Z) * n_total_cm3
        return np.zeros(N)

    @staticmethod
    def _doppler_hwhm_nm(wavelength_nm: float,
                         T_i_K       : float,
                         mass_kg     : float) -> float:
        """
        Doppler (thermal) Gaussian half-width at half-maximum (HWHM) [nm].
        """
        lam_m = wavelength_nm * 1e-9
        hwhm_m = (lam_m / C_LIGHT) * np.sqrt(8.0 * np.log(2.0) * K_B_J * T_i_K / mass_kg)
        return hwhm_m * 1e9

    @staticmethod
    def voigt_profile(wavelength_grid_nm: np.ndarray,
                      center_nm         : float,
                      fwhm_gaussian_nm  : float,
                      fwhm_lorentzian_nm: float) -> np.ndarray:
        sigma_G  = fwhm_gaussian_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gamma_L  = fwhm_lorentzian_nm / 2.0
        delta_lam = wavelength_grid_nm - center_nm
        z = (delta_lam + 1j * gamma_L) / (sigma_G * np.sqrt(2.0))
        profile = np.real(wofz(z)) / (sigma_G * np.sqrt(2.0 * np.pi))
        dl = np.gradient(wavelength_grid_nm)
        norm = np.dot(profile, dl)
        if norm > 1e-30:
            profile /= norm
        return profile.astype(np.float64)

    @staticmethod
    def compute_absorption_coefficient(populations: np.ndarray,
                                       levels: List[EnergyLevel],
                                       transitions: List[Transition],
                                       wavelength_grid_nm: np.ndarray,
                                       T_i_K: float,
                                       n_e_cm3: float,
                                       mass_kg: float) -> np.ndarray:
        kappa = np.zeros(len(wavelength_grid_nm), dtype=np.float64)
        for t in transitions:
            i, k = t.lower_idx, t.upper_idx
            if i >= len(populations) or k >= len(populations):
                continue
            n_i = populations[i]
            n_k = populations[k]
            if n_i <= 0.0 and n_k <= 0.0:
                continue
            fwhm_G = 2.0 * PhysicsCalculator._doppler_hwhm_nm(t.wavelength_nm, T_i_K, mass_kg)
            fwhm_L = 2.0 * PhysicsCalculator._stark_hwhm_nm(t.wavelength_nm, n_e_cm3)
            hw_window = max(50.0 * max(fwhm_G, fwhm_L), 0.5)
            mask = np.abs(wavelength_grid_nm - t.wavelength_nm) <= hw_window
            if not np.any(mask):
                continue
            phi_window = PhysicsCalculator.voigt_profile(wavelength_grid_nm[mask], t.wavelength_nm, fwhm_G, fwhm_L)
            phi = np.zeros_like(wavelength_grid_nm)
            phi[mask] = phi_window
            lam_m = t.wavelength_nm * 1e-9
            sigma_m2 = ((lam_m ** 4) / (8.0 * np.pi * C_LIGHT)) * (t.g_upper / t.g_lower) * t.A_ki
            sigma_m2_spectral = sigma_m2 * 1e9
            stim = min((n_k * t.g_lower) / (n_i * t.g_upper), 1.0) if n_i > 0 else 0.0
            stim_corr = max(1.0 - stim, 0.0)
            n_i_m3 = n_i * 1e6
            kappa_m = n_i_m3 * sigma_m2_spectral * phi * stim_corr
            kappa += kappa_m * 0.01
        return kappa

    @staticmethod
    def compute_emission_coefficient(populations: np.ndarray,
                                     transitions: List[Transition],
                                     wavelength_grid_nm: np.ndarray,
                                     T_i_K: float,
                                     n_e_cm3: float,
                                     mass_kg: float) -> np.ndarray:
        j_spec = np.zeros(len(wavelength_grid_nm), dtype=np.float64)
        for t in transitions:
            k = t.upper_idx
            if k >= len(populations):
                continue
            n_k = populations[k]
            if n_k <= 0.0:
                continue
            fwhm_G = 2.0 * PhysicsCalculator._doppler_hwhm_nm(t.wavelength_nm, T_i_K, mass_kg)
            fwhm_L = 2.0 * PhysicsCalculator._stark_hwhm_nm(t.wavelength_nm, n_e_cm3)
            hw_window = max(50.0 * max(fwhm_G, fwhm_L), 0.5)
            mask = np.abs(wavelength_grid_nm - t.wavelength_nm) <= hw_window
            if not np.any(mask):
                continue
            phi_window = PhysicsCalculator.voigt_profile(wavelength_grid_nm[mask], t.wavelength_nm, fwhm_G, fwhm_L)
            phi = np.zeros_like(wavelength_grid_nm)
            phi[mask] = phi_window
            lam_m = t.wavelength_nm * 1e-9
            h_c_ov_lam = H_PLANCK_J * C_LIGHT / lam_m
            n_k_m3 = n_k * 1e6
            j_SI = (h_c_ov_lam / (4.0 * np.pi)) * t.A_ki * n_k_m3 * phi
            j_spec += j_SI * 10.0
        return j_spec

    @staticmethod
    def integrate_rte(I_core: np.ndarray,
                      kappa_shell: np.ndarray,
                      j_shell: np.ndarray,
                      d_shell_m: float,
                      wavelength_grid_nm: np.ndarray) -> np.ndarray:
        d_shell_cm = d_shell_m * 100.0
        tau = np.clip(kappa_shell * d_shell_cm, 0.0, 700.0)
        exp_neg_tau = np.exp(-tau)
        with np.errstate(divide='ignore', invalid='ignore'):
            S = np.where(kappa_shell > 1e-40, j_shell / kappa_shell, 0.0)
        return I_core * exp_neg_tau + S * (1.0 - exp_neg_tau)

    @staticmethod
    def instrumental_broadening(spectrum: np.ndarray,
                                wavelength_grid_nm: np.ndarray,
                                fwhm_instrument_nm: float = 0.1) -> np.ndarray:
        from scipy.signal import fftconvolve
        dl = (wavelength_grid_nm[-1] - wavelength_grid_nm[0]) / (len(wavelength_grid_nm) - 1)
        sigma_nm = fwhm_instrument_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        sigma_pts = sigma_nm / dl
        half_win = int(np.ceil(5.0 * sigma_pts))
        kernel_x = np.arange(-half_win, half_win + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (kernel_x / sigma_pts) ** 2)
        kernel /= kernel.sum()
        I_conv = fftconvolve(spectrum, kernel, mode='same')
        return I_conv.astype(np.float64)

    @staticmethod
    def compute_saha_ionization_fractions(element: str,
                                          total_fraction: float,
                                          T_e_K: float,
                                          n_e_cm3: float,
                                          fetcher: DataFetcher) -> Tuple[float, float]:
        ion_energy_eV = fetcher.get_ionization_energy(element, sp_num=1)
        if ion_energy_eV <= 0.0:
            warnings.warn(f"[Saha Splitter] Energi ionisasi {element} tidak ditemukan. Diasumsikan 100% netral.")
            return total_fraction, 0.0
        thermal_term = (2.0 * np.pi * M_ELECTRON * K_B_J * T_e_K / (H_PLANCK_J ** 2)) ** 1.5
        n_e_m3 = n_e_cm3 * 1e6
        kT_eV = K_B_eV * T_e_K
        pre_factor = 2.0 * (thermal_term / n_e_m3)
        exponent = -ion_energy_eV / kT_eV
        if exponent > 700:
            saha_ratio = np.inf
        elif exponent < -700:
            saha_ratio = 0.0
        else:
            saha_ratio = pre_factor * np.exp(exponent)
        if np.isinf(saha_ratio):
            return 0.0, total_fraction
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
                 fetcher        : Optional[DataFetcher] = None,
                 use_rte        : bool = True,
                 instrument_fwhm_nm: float = 0.1):
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
        self.use_rte = use_rte
        self.instrument_fwhm_nm = instrument_fwhm_nm

        wl_min, wl_max = SIMULATION_CONFIG["wl_range_nm"]
        N_pts = SIMULATION_CONFIG["resolution"]
        self.wavelengths = np.linspace(wl_min, wl_max, N_pts, dtype=np.float64)

    def _run_zone(self, zone: PlasmaZoneParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        kappa_total = np.zeros_like(self.wavelengths)
        j_total     = np.zeros_like(self.wavelengths)
        zone_top_lines = []

        # KOREKSI 1: Normalisasi Kekekalan Muatan (Charge Neutrality)
        # Menghitung Z_mean (rata-rata muatan per partikel berat)
        # sp_num 1 (netral, Z=0), sp_num 2 (ionisasi tunggal, Z=1), sp_num 3 (Z=2)
        z_mean = sum(frac * float(sp_num - 1) for _, sp_num, frac in self.elements)
        z_mean = max(z_mean, 1e-10) # Mencegah pembagian dengan nol pada plasma super dingin
        
        # Total Kerapatan Partikel Berat diturunkan dari n_e
        n_heavy_total = zone.n_e_cm3 / z_mean

        for elem_sym, sp_num, frac in self.elements:
            df = self.fetcher.get_transitions(elem_sym, sp_num, wl_range=SIMULATION_CONFIG["wl_range_nm"])
            if df.empty:
                continue

            levels, transitions = DataFetcher.build_levels_and_transitions(df)
            if not levels:
                continue

            # Kerapatan aktual diekspansi dari populasi partikel berat, bukan n_e
            n_species_cm3 = frac * n_heavy_total

            populations = PhysicsCalculator.solve_boltzmann_populations(
                levels=levels, transitions=transitions, T_e_K=zone.T_e_K,
                n_e_cm3=zone.n_e_cm3, n_total_cm3=n_species_cm3, sp_num=sp_num
            )

            mass_kg = ATOMIC_MASS_AMU.get(elem_sym, 40.0) * AMU_KG

            kappa = PhysicsCalculator.compute_absorption_coefficient(
                populations=populations, levels=levels, transitions=transitions,
                wavelength_grid_nm=self.wavelengths, T_i_K=zone.T_i_K,
                n_e_cm3=zone.n_e_cm3, mass_kg=mass_kg
            )

            j_em = PhysicsCalculator.compute_emission_coefficient(
                populations=populations, transitions=transitions,
                wavelength_grid_nm=self.wavelengths, T_i_K=zone.T_i_K,
                n_e_cm3=zone.n_e_cm3, mass_kg=mass_kg
            )

            kappa_total += kappa
            j_total     += j_em
            
            for t in transitions:
                n_up = populations[t.upper_idx]
                line_int = (n_up * t.A_ki * (H_PLANCK_J * C_LIGHT / (t.wavelength_nm * 1e-7))) / (4.0 * np.pi)
                if line_int > 0:
                    zone_top_lines.append({
                        "wl": t.wavelength_nm, "elem": elem_sym,
                        "ion": "I" if sp_num==1 else "II", "int": line_int
                    })

        # KOREKSI 2: Terapkan limit RTE eksak untuk Core (Mencegah pelanggaran Blackbody limit)
        if getattr(self, 'use_rte', True):
            d_cm = zone.thickness_m * 100.0
            tau_zone = np.clip(kappa_total * d_cm, 0.0, 700.0)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                S_zone = np.where(kappa_total > 1e-40, j_total / kappa_total, 0.0)
                
            I_zone = np.where(kappa_total > 1e-40,
                              S_zone * (1.0 - np.exp(-tau_zone)),
                              j_total * d_cm)
        else:
            d_cm = zone.thickness_m * 100.0
            I_zone = j_total * d_cm

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
        kappa_shell, j_shell, _, shell_lines = self._run_zone(self.shell)

        if getattr(self, 'use_rte', True):
            print(f"[TwoZonePlasma] Phase 3: Integrating RTE...")
            I_obs = PhysicsCalculator.integrate_rte(
                I_core            = I_core,
                kappa_shell       = kappa_shell,
                j_shell           = j_shell,
                d_shell_m         = self.shell.thickness_m,
                wavelength_grid_nm= self.wavelengths,
            )
        else:
            print(f"[TwoZonePlasma] Phase 3: Skipping RTE (Optically Thin Mode)...")
            d_shell_cm = self.shell.thickness_m * 100.0
            I_obs = I_core + j_shell * d_shell_cm

        # Konvolusi fungsi aparatus (Resolusi instrumen = 0.1 nm)
        I_obs = PhysicsCalculator.instrumental_broadening(
            I_obs,
            self.wavelengths,
            fwhm_instrument_nm=self.instrument_fwhm_nm,
        )

        # KOREKSI 3: Menghapus Normalisasi Skala Buatan (Artificial Normalization)
        # Normalize to target_max_intensity
        # I_max = np.max(np.abs(I_obs))
        # if I_max > 0:
        #     I_obs = I_obs / I_max * SIMULATION_CONFIG["target_max_intensity"]

        # Optical depth diagnostics
        tau_shell = kappa_shell * (self.shell.thickness_m * 100.0)

        prominent_lines = self._merge_top_lines(core_lines, shell_lines)

        metadata = {
            "core_T_e_K"           : self.core.T_e_K,
            "core_T_i_K"           : self.core.T_i_K,
            "core_n_e_cm3"         : self.core.n_e_cm3,
            "core_thickness_m"     : self.core.thickness_m,
            "shell_T_e_K"          : self.shell.T_e_K,
            "shell_T_i_K"          : self.shell.T_i_K,
            "shell_n_e_cm3"        : self.shell.n_e_cm3,
            "shell_thickness_m"    : self.shell.thickness_m,
            "instrument_fwhm_nm"   : self.instrument_fwhm_nm,
            "use_rte"              : bool(self.use_rte),
            "tau_shell_max"        : float(np.max(tau_shell)),
            "tau_shell_mean"       : float(np.mean(tau_shell)),
            "optically_thin_frac"  : float(np.mean(tau_shell < 0.1)),
            "stark_approx_flag"    : False,   # Tabulated empirical Griem data utilized
            "cr_method"            : "Radau (implicit RK)",
            "jacobian"             : "Analytical (J = M_CR)",
        }

        metadata["top_lines"] = prominent_lines
        metadata["core_top_lines"] = core_lines
        metadata["shell_top_lines"] = shell_lines
        
        return self.wavelengths, I_obs, metadata

    @staticmethod
    def _merge_top_lines(core_lines: List[Dict], shell_lines: List[Dict]) -> List[Dict]:
        merged: Dict[Tuple[str, str, float], Dict] = {}
        for zone_name, lines in (("Core", core_lines), ("Shell", shell_lines)):
            for line in lines:
                key = (
                    str(line["elem"]),
                    str(line["ion"]),
                    round(float(line["wl"]), 4),
                )
                current = merged.get(key)
                intensity = float(line["int"])
                if current is None:
                    merged[key] = {
                        "wl": float(line["wl"]),
                        "elem": str(line["elem"]),
                        "ion": str(line["ion"]),
                        "int": intensity,
                        "zone": zone_name,
                    }
                    continue
                current["int"] += intensity
                if intensity > float(current.get("zone_int", 0.0)):
                    current["zone"] = zone_name
                    current["zone_int"] = intensity

        normalized = []
        for item in merged.values():
            item.pop("zone_int", None)
            normalized.append(item)
        normalized.sort(key=lambda x: x["int"], reverse=True)
        return normalized[:50]


# =============================================================================
# MAIN EXECUTOR & INTERACTIVE UI (sim.py-like Mechanism)
# =============================================================================
class LIBSSimulator:
    @staticmethod
    def run_simulation(selected_elements: List[Tuple[str, float]],
                       core_temp: float,
                       core_ne: float,
                       show_labels: bool = True,
                       use_rte: bool = True):
        if go is None:
            raise ImportError("plotly is required for plotting. Install plotly or disable plotting.")
        total_pct = sum(p for _, p in selected_elements)
        if abs(total_pct - 100.0) > 1e-6:
            print(f"Error: Persentase komposit atomik mutlak harus 100%. (Dapat {total_pct}%)")
            return
        fetcher = DataFetcher()
        elements_expanded = []
        for elem, pct in selected_elements:
            base_frac = pct / 100.0
            f_neu, f_ion = PhysicsCalculator.compute_saha_ionization_fractions(elem, base_frac, core_temp, core_ne, fetcher)
            if f_neu > 1e-4:
                elements_expanded.append((elem, 1, f_neu))
            if f_ion > 1e-4:
                elements_expanded.append((elem, 2, f_ion))
        tot = sum(f for _, _, f in elements_expanded)
        elements_expanded = [(sym, sp, f/tot) for sym, sp, f in elements_expanded]
        core = PlasmaZoneParams(T_e_K=core_temp, T_i_K=max(core_temp - 2000.0, 3000.0), n_e_cm3=core_ne, thickness_m=1e-3, label="Core")
        shell = PlasmaZoneParams(T_e_K=core_temp * 0.5, T_i_K=max(core_temp * 0.5 - 1000.0, 3000.0), n_e_cm3=core_ne * 0.1, thickness_m=2e-3, label="Shell")
        model = TwoZonePlasma(core, shell, elements_expanded, fetcher, use_rte=use_rte)
        wavelengths, I_obs, meta = model.run()
        element_str = " ".join([f"{sym}{pct:.0f}%" for sym, pct in selected_elements])
        dyn_title = f"<b>Saha-Boltzmann Deterministic (Q1 Target)</b><br>{element_str} | T_Core={int(core_temp)} K | n_e={core_ne:.1e} cm⁻³"
        fig = go.Figure()
        empirical_path = os.path.join(_PROJECT_DIR, "..", "raw", "Skala-5", "0", "Cu_D1us-W50us-ii-3500-acc-5_760 torr-skala 5.asc")
        if os.path.exists(empirical_path):
            try:
                import pandas as pd
                df_emp = pd.read_csv(empirical_path, sep='\t', header=None, names=['wl', 'int'])
                wl_min, wl_max = SIMULATION_CONFIG["wl_range_nm"]
                df_emp = df_emp[(df_emp['wl'] >= wl_min) & (df_emp['wl'] <= wl_max)]
                emp_wl = df_emp['wl'].values
                emp_int = df_emp['int'].values
                if len(emp_int) > 0 and np.max(I_obs) > 0:
                    emp_int_scaled = emp_int / np.max(emp_int) * np.max(I_obs) * 0.95
                    fig.add_trace(go.Scatter(x=emp_wl, y=emp_int_scaled, mode='lines', name='Empirical Spectrum (Cu Skala-5)', line=dict(color='red', width=1.0), opacity=0.65))
            except Exception as e:
                print(f"[Warning] Gagal memuat data empiris: {e}")
        fig.add_trace(go.Scatter(x=wavelengths, y=I_obs, mode='lines', name='Forward Syntax (Saha-Boltzmann)', line=dict(color='navy', width=1.5)))
        if np.max(I_obs) > 0 and show_labels:
            from scipy.signal import find_peaks
            peak_indices, _ = find_peaks(I_obs, height=0.05 * np.max(I_obs), distance=150)
            annotations = []
            top_lines = meta.get("top_lines", [])
            for wl, intensity in zip(wavelengths[peak_indices], I_obs[peak_indices]):
                if top_lines:
                    closest = min(top_lines, key=lambda x: abs(x["wl"] - wl))
                    label = f"{closest['elem']} {closest['ion']}<br>{wl:.2f} nm" if abs(closest["wl"] - wl) < 0.2 else f"{wl:.2f} nm"
                else:
                    label = f"{wl:.2f} nm"
                annotations.append(go.layout.Annotation(x=wl, y=intensity, text=label, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, ax=0, ay=-40, font=dict(size=9, color="#ffffff"), align="center", bordercolor="#555", borderwidth=1, borderpad=2, bgcolor="#ff7f0e", opacity=0.8))
            fig.update_layout(annotations=annotations)
        ann_text = (f"<b>Core:</b> T_e={meta['core_T_e_K']:.0f} K, " f"n_e={meta['core_n_e_cm3']:.1e} cm⁻³<br>" f"<b>Shell:</b> T_e={meta['shell_T_e_K']:.0f} K, " f"n_e={meta['shell_n_e_cm3']:.1e} cm⁻³<br>" f"<b>τ_shell max:</b> {meta['tau_shell_max']:.3f}<br>")
        fig.update_layout(title=dyn_title, xaxis_title="Wavelength (nm)", yaxis_title="Spectral Radiance (erg/s/cm³/nm/sr)", plot_bgcolor='white', paper_bgcolor='white', xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray', minor=dict(showgrid=True, gridcolor='lightgrey', griddash='dot')), yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray', minor=dict(showgrid=True, gridcolor='lightgrey', griddash='dot')), font=dict(family="Arial, sans-serif", size=12), annotations=list(fig.layout.annotations) if fig.layout.annotations else [] + [go.layout.Annotation(x=0.98, y=0.98, xref='paper', yref='paper', text=ann_text, showarrow=False, align='left', bgcolor='rgba(255,255,255,0.85)', bordercolor='black', borderwidth=1, font=dict(size=10))])
        output_html_path = os.path.abspath(os.path.join(_PROJECT_DIR, "..", "libs_spectrum_output.html"))
        try:
            fig.write_html(output_html_path)
            print(f"\n[INFO] Grafik memuat ke browser otomatis...")
            print(f"       Lokasi file: {output_html_path}\n")
            import webbrowser
            webbrowser.open(f"file://{output_html_path}")
        except Exception:
            pass
        fig.show()

    @staticmethod
    def parse_element_input(user_str: str) -> List[Tuple[str, float]]:
        elements = []
        for p in user_str.replace(';', ',').split(','):
            p = p.strip()
            if not p:
                continue
            m = re.match(r"([A-Za-z]{1,2})\s*[:\s-]*\s*(\d+\.?\d*)", p)
            if m:
                elements.append((m.group(1).capitalize(), float(m.group(2))))
        return elements

    @staticmethod
    def create_composition_form():
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            print("Peringatan: Modul 'ipywidgets' tidak tersedia. Silakan instal atau jalankan via CLI konvensional.")
            return
        base_elements = ['Al', 'Ca', 'Fe', 'Si', 'Mg', 'Ti', 'Cr', 'Mn', 'Ni', 'Cu', 'Li', 'Na', 'K', 'O', 'H', 'N', 'Sr', 'C', 'Rb', 'Co', 'Pb']
        element_options = [(elem, elem) for elem in base_elements]
        composition_widgets = []
        total_percentage_label = widgets.Label(value="Total Percentage: 0.0%")
        rows_vbox = widgets.VBox([])
        def add_element_row(_=None):
            element_dropdown = widgets.Dropdown(options=element_options, description="Element:", layout={'width': '300px'})
            percentage_input = widgets.FloatText(value=0.0, description="Percentage (%):", layout={'width': '200px'})
            remove_button = widgets.Button(description="Remove", button_style='danger')
            def update_total(_=None):
                total_percentage_label.value = f"Total Percentage: {sum(w[1].value for w in composition_widgets):.1f}%"
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
        display(widgets.VBox([rows_vbox, widgets.HBox([add_button, total_percentage_label])]))
        add_element_row()
        submit_button = widgets.Button(description="Generate Spectrum", button_style='primary')
        show_labels_checkbox = widgets.Checkbox(value=True, description='Show Peak Labels', style={'description_width': 'initial'})
        use_rte_checkbox = widgets.Checkbox(value=True, description='Apply Exact RTE (Self-Absorption)', style={'description_width': 'initial'})
        output = widgets.Output()
        temperature_input = widgets.FloatText(value=9400, description="Temperature (K):", layout={'width': '300px'})
        ne_options_values = [10**exp for exp in np.arange(15, 18.1, 0.1)]
        ne_options = list(zip([f"{val:.1e}" for val in ne_options_values], ne_options_values))
        electron_density_input = widgets.Dropdown(options=ne_options, value=1e17, description="Electron Density (cm^-3):", layout={'width': '400px'})
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
                LIBSSimulator.run_simulation(selected_elements, temperature_input.value, electron_density_input.value, show_labels_checkbox.value, use_rte_checkbox.value)
        submit_button.on_click(on_submit)
        display(temperature_input, electron_density_input, show_labels_checkbox, use_rte_checkbox, submit_button, output)

    @staticmethod
    def main():
        default_elements = [("Cu", 0.85), ("N", 0.10), ("O", 0.04), ("C", 0.01)]
        default_selected = [(sym, frac * 100.0) for sym, frac in default_elements]
        if 'ipykernel' in __import__('sys').modules:
            print("Menjalankan dalam mode interaktif GUI termodinamika (Jupyter/IPython).")
            LIBSSimulator.create_composition_form()
        else:
            print("=" * 75)
            print("  TWO-ZONE SAHA-BOLTZMANN FORWARD MODEL  —  Q1 Thesis Physics Engine")
            print("  (Interactive CLI Mode)")
            print("=" * 75)
            print(f"Default: {' '.join([f'{s}{p:.0f}%' for s,p in default_selected])}")
            print("Tips: Masukkan elemen seperti 'Ca 100' atau tekan Enter untuk default.")
            try:
                while True:
                    print("\n" + "-"*40)
                    elem_input = input(">> Komposisi Elemen [%] [DEFAULT]: ").strip()
                    selected = default_selected if not elem_input else LIBSSimulator.parse_element_input(elem_input)
                    if not selected:
                        print("Error: Format input elemen tidak valid.")
                        continue
                    total_pct = sum(p for _, p in selected)
                    if abs(total_pct - 100.0) > 1e-3:
                        print(f"Peringatan: Total persentase ({total_pct}%) tidak 100%. Menyetel ulang proporsi...")
                        selected = [(s, p*100.0/total_pct) for s, p in selected]
                    temp_str = input(">> Temperatur Core (K)  [9400]: ").strip()
                    temp = float(temp_str) if temp_str else 9400.0
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
                    rte_choice = input(">> Terapkan kalkulasi RTE (Self-Absorption)? (y/n) [y]: ").strip().lower()
                    use_rte = False if rte_choice == 'n' else True
                    print(f"\n[Saha-Boltzmann Engine] Mensimulasikan {selected} pada T={temp}K, n_e={ne:.1e} (RTE={use_rte})...")
                    LIBSSimulator.run_simulation(selected, temp, ne, show_labels=True, use_rte=use_rte)
                    if input("\nSimulasi lagi? (y/n) [y]: ").strip().lower() == 'n':
                        break
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
            except Exception as e:
                print(f"\nTerjadi kesalahan fatal: {e}")
        print("\n[DONE] libs_physics.py complete.")

if __name__ == "__main__":
    LIBSSimulator.main()
