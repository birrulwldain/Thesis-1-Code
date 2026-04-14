
import sys
import os
import glob
import pandas as pd
import numpy as np
import re
import json
import math
import h5py
import argparse
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).parent.resolve()
PLOT_STATE_DIR = BASE_DIR / "b"
TENE_FILE_1 = BASE_DIR / "TeNe_summary.xlsx"
TENE_FILE_2 = BASE_DIR / "c" / "TeNe_summary.xlsx"
TENE_FILE = TENE_FILE_1 if TENE_FILE_1.exists() else TENE_FILE_2
PARTITION_FILE = BASE_DIR / "partition_by_sample.csv"
XRF_REF_FILE = BASE_DIR / "xrf_reference.csv"
NIST_H5 = BASE_DIR / "nist_lines_all.h5"
# Try to locate atomic weights file
ATOMIC_WEIGHTS_H5_g1 = BASE_DIR.parent / "HDF5" / "atomic_weights_isotopes.h5"
ATOMIC_WEIGHTS_H5_g2 = BASE_DIR.parent.parent / "arsip" / "HDF5" / "atomic_weights_isotopes.h5"
ATOMIC_WEIGHTS_H5 = ATOMIC_WEIGHTS_H5_g2 if ATOMIC_WEIGHTS_H5_g2.exists() else ATOMIC_WEIGHTS_H5_g1

OUTPUT_FILE = BASE_DIR / "concentration_results.csv"

KB_EV = 8.617333262e-5  # Boltzmann constant eV/K

# Fallback atomic weights if HDF5 fails
ATOMIC_WEIGHTS_FALLBACK = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Sc": 44.956,
    "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.63, "As": 74.922, "Se": 78.96, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906,
    "Mo": 95.95, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87,
    "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60,
    "I": 126.90, "Xe": 131.29, "Cs": 132.91, "Ba": 137.33, "La": 138.91,
    "Ce": 140.12, "Pr": 140.91, "Nd": 144.24, "Sm": 150.36, "Eu": 151.96,
    "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93, "Er": 167.26,
    "Tm": 168.93, "Yb": 173.05, "Lu": 174.97, "Hf": 178.49, "Ta": 180.95,
    "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08,
    "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98,
    "Th": 232.04, "U": 238.03
}

IONIZATION_ENERGIES = {
    'Ca': 6.11316, 'Mg': 7.64624, 'Si': 8.15168, 'Fe': 7.9024,
    'Al': 5.98577, 'Ti': 6.8281, 'Na': 5.13908, 'K': 4.34066,
    'Mn': 7.43402, 'Sr': 5.69484, 'Ba': 5.21170, 'Cu': 7.72638,
    'Li': 5.39171, 'Ag': 7.5762, 'K': 4.34066
}

STOICHIOMETRY = {
    'Si': (1, 2), 'Al': (2, 3), 'Fe': (2, 3), # Fe2O3
    'Ca': (1, 1), 'Mg': (1, 1), 'Na': (2, 1),
    'K': (2, 1), 'Ti': (1, 2), 'Mn': (1, 1), # MnO
    'P': (2, 5), 'S': (1, 3), 'Cr': (2, 3),
    'Cu': (1, 1), 'Zn': (1, 1), 'Sr': (1, 1),
    'Ba': (1, 1), 'Eu': (2, 3), 'Y': (2, 3),
    'Zr': (1, 2), 'V': (2, 5), 'Ni': (1, 1),
    'Pb': (1, 1), 'Co': (1, 1)
}

W_O = 15.999

def load_atomic_weights():
    weights = ATOMIC_WEIGHTS_FALLBACK.copy()
    if ATOMIC_WEIGHTS_H5.exists():
        try:
            with h5py.File(ATOMIC_WEIGHTS_H5, 'r') as f:
                pass 
        except Exception as e:
            print(f"Warning: Could not load atomic weights HDF5: {e}")
    return weights

def load_nist_data():
    if not NIST_H5.exists():
        print(f"Error: {NIST_H5} not found.")
        return pd.DataFrame()
    try:
        df = pd.read_hdf(NIST_H5, key='nist_lines')
        cols = ['ritz_wl_air(nm)', 'Aki(s^-1)', 'Ek(eV)', 'g_k']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.rename(columns={
            'ritz_wl_air(nm)': 'wavelength',
            'Aki(s^-1)': 'Aki',
            'Ek(eV)': 'Ek',
            'g_k': 'gk',
            'sp_num': 'ion'
        })
        return df
    except Exception as e:
        print(f"Error loading NIST HDF5: {e}")
        return pd.DataFrame()

def calculate_saha_ratio(Te_K, Ne_cm3, IP_eV, Z_I, Z_II):
    if Ne_cm3 <= 0 or Te_K <= 0: return 0.0
    me = 9.10938356e-28
    kb = 1.380649e-16
    h = 6.62607015e-27
    ev_to_erg = 1.60218e-12
    
    T_part_base = (2.0 * np.pi * me * kb * Te_K) / (h**2)
    T_part = 2.0 * math.pow(T_part_base, 1.5)
    exp_factor = math.exp(-(IP_eV * ev_to_erg) / (kb * Te_K))
    
    if Z_I == 0: return 0.0
    ratio = (T_part / Ne_cm3) * (Z_II / Z_I) * exp_factor
    return ratio

def get_plot_state_files():
    pattern1 = str(PLOT_STATE_DIR / "plot_state_*.xlsx")
    pattern2 = str(PLOT_STATE_DIR / "plot_state_*.csv")
    files = glob.glob(pattern1) + glob.glob(pattern2)
    return sorted(files)

def extract_sample_name(filename):
    """Ekstrak nama file penuh setelah 'plot_state_' sebagai key unik.
    Contoh: plot_state_S1-D0.5us-...-1.csv -> 'S1-D0.5us-...-1'
    """
    basename = os.path.basename(filename)
    # Hapus ekstensi dan prefix 'plot_state_'
    stem = re.sub(r'\.xlsx$|\.csv$', '', basename)
    if stem.startswith('plot_state_'):
        return stem[len('plot_state_'):]
    return None

def extract_snumber(sample_key: str) -> str:
    """Ambil nomor sampel S1/S2/... dari key panjang untuk lookup TeNe.
    Contoh: 'S1-D0.5us-...-1' -> 'S1'
    """
    m = re.match(r'(S\d+)', sample_key)
    return m.group(1) if m else sample_key

class CFLAnalyzer:
    def __init__(self):
        self.temp_map = {}
        self.ne_map = {}
        self.z_map = {}
        self.nist_df = pd.DataFrame()
        self.atomic_weights = {}
        self.xrf_quant = {}
        self.xrf_qual = {}
        self.plot_files_map = {} # Map SampleID -> FilePath
        self.plot_state_dir = PLOT_STATE_DIR  # Default, bisa diubah via GUI
        
        self.load_data()

    def load_data(self):
        # 0. Reload fresh file map dari direktori aktif
        self.plot_files_map.clear()
        src_dir = getattr(self, 'plot_state_dir', PLOT_STATE_DIR)
        for ext in ["*.xlsx", "*.csv"]:
            for f in glob.glob(str(src_dir / ext)):
                 sid = extract_sample_name(f)
                 if sid: self.plot_files_map[sid] = f
        # 1. TeNe
        if TENE_FILE.exists():
            tene_df = pd.read_excel(TENE_FILE)
            if 'sample' in tene_df.columns and 'Te_K' in tene_df.columns:
                self.temp_map = dict(zip(tene_df['sample'], tene_df['Te_K']))
                ne_col = next((c for c in ['Ne_mean_cm-3', 'Ne_cm3', 'Ne'] if c in tene_df.columns), None)
                if ne_col:
                    self.ne_map = dict(zip(tene_df['sample'], tene_df[ne_col]))
        
        # 2. Partition
        if PARTITION_FILE.exists():
            part_df = pd.read_csv(PARTITION_FILE)
            for _, row in part_df.iterrows():
                key = (str(row['sample']), str(row['element']), int(float(row['sp_num'])))
                self.z_map[key] = float(row['Z'])
        
        # 3. NIST & Atomic Weights
        self.nist_df = load_nist_data()
        self.atomic_weights = load_atomic_weights()
        
        # 4. XRF
        self._load_xrf()
        
        # 5. Plot Files mapping is already done at top of load_data()

    def _load_xrf(self):
        XRF_JSON = BASE_DIR / "xrf_data.json"
        XRF_MATRIX = BASE_DIR.parent.parent / "paper" / "xrf_matrix.csv"
        
        if XRF_JSON.exists():
            try:
                with open(XRF_JSON, 'r') as f: self.xrf_quant = json.load(f)
            except: pass
            
        # Prioritas: Load dari CSV referensi baru (S1-S24)
        if XRF_REF_FILE.exists():
            try:
                ref_df = pd.read_csv(XRF_REF_FILE)
                # Format: sample, element, xrf_perc
                for _, row in ref_df.iterrows():
                    sid = str(row['sample'])
                    el = str(row['element'])
                    val = float(row['xrf_perc'])
                    if sid not in self.xrf_quant: self.xrf_quant[sid] = {}
                    self.xrf_quant[sid][el] = val
                print(f"[XRF] Loaded reference data for {ref_df['sample'].nunique()} samples.")
            except Exception as e:
                print(f"[XRF] Error loading {XRF_REF_FILE}: {e}")
            
        if XRF_MATRIX.exists():
            try:
                mat_df = pd.read_csv(XRF_MATRIX)
                mat_df['Elemen'] = mat_df['Elemen'].astype(str).str.strip()
                for _, row in mat_df.iterrows():
                    el = row['Elemen']
                    for col in mat_df.columns:
                        if col.startswith('S'):
                            val = str(row[col]).strip()
                            if val == '1' or 'xrfdot' in val or val.lower() == 'yes':
                                if col not in self.xrf_qual: self.xrf_qual[col] = set()
                                self.xrf_qual[col].add(el)
            except: pass

    def analyze_sample(self, sample_id, saha_elements=[], saha_te_overrides={}, 
                       excluded_lines=set(), exclude_elements=[]):
        """
        Analyze a single sample.
        excluded_lines: set of tuples (Element, center_nm) to skip.
        exclude_elements: list of element symbols (e.g. ['Fe', 'K']) to skip entirely.
        Returns (results_list, details_list, log_messages)
        """
        log = []
        if sample_id not in self.plot_files_map:
            log.append(f"Error: No plot file found for {sample_id}")
            return [], [], log
            
        pfile = self.plot_files_map[sample_id]
        # Lookup Te/Ne: coba key panjang dulu, lalu fallback ke nomor S saja
        s_short = extract_snumber(sample_id)
        te = self.temp_map.get(sample_id) or self.temp_map.get(s_short)
        
        if te is None:
            te = 8000.0 # Default fallback
            log.append(f"Warning: No specific Te found for '{sample_id}'. Using fallback {te} K.")
            
        try:
            if pfile.endswith('.csv'):
                # Sniff: deteksi separator dan decimal otomatis
                with open(pfile, 'r', encoding='utf-8-sig') as _f:
                    _first = _f.readline()
                _sep = ';' if ';' in _first else ','
                _dec = ',' if _sep == ';' else '.'
                state_df = pd.read_csv(pfile, sep=_sep, decimal=_dec, on_bad_lines='skip')
            else:
                state_df = pd.read_excel(pfile)
            state_df.columns = [c.strip() for c in state_df.columns]
        except Exception as e:
            log.append(f"Error reading {pfile}: {e}")
            return [], [], log
            
        col_wl = 'center_nm'
        col_area = 'area'
        col_element = 'element'
        # Group calculation by Element
        element_concentrations = {} 
        printed_saha_elements = set()
        saha_tracking = {} 
        
        results_detail = []
        
        # Helper to force cast to float, handling string commas
        def _parse_float(val):
            if pd.isna(val) or val == '': return 0.0
            if isinstance(val, str):
                val = val.replace(',', '.')
            try: return float(val)
            except: return 0.0
            
        for idx, row in state_df.iterrows():
            el = str(row.get(col_element, '')).strip()
            if not el or el.lower() == 'nan': continue
            
            # --- Element Filter ---
            if exclude_elements and el in exclude_elements:
                continue
                
            wl_obs = _parse_float(row.get(col_wl, 0))
            
            # --- Exclusion Check ---
            # Fuzzy match (within 0.001 nm) to avoid float precision issues if coming from GUI
            # excluded_lines expects tuples: (Element, Wavelength)
            is_excluded = False
            for (ex_el, ex_wl) in excluded_lines:
                if ex_el == el and abs(wl_obs - ex_wl) < 0.001:
                    is_excluded = True
                    break
            if is_excluded:
                continue
            
            ion_str = str(row.get('ion_or_sp', '1')).strip()
            if ion_str == 'I': ion = 1
            elif ion_str == 'II': ion = 2
            else:
                try: ion = int(float(ion_str))
                except: ion = 1
                
            area = _parse_float(row.get(col_area, 0))
            if area <= 0: continue
            
            # --- Params ---
            use_baked = False
            Aki, gk, Ek = 0.0, 0.0, 0.0
            nist_wl_val = wl_obs
            
            if 'aki' in row and 'gk' in row and 'ek_ev' in row:
                try:
                    Aki = _parse_float(row['aki'])
                    gk = _parse_float(row['gk'])
                    Ek = _parse_float(row['ek_ev'])
                    if Aki > 0 and gk > 0:
                        use_baked = True
                        if 'nist_wavelength_nm' in row:
                             nist_wl_val = _parse_float(row['nist_wavelength_nm'])
                except: pass
            
            if not use_baked and not self.nist_df.empty:
                subset = self.nist_df[
                    (self.nist_df['element'] == el) & 
                    (self.nist_df['ion'] == ion) & 
                    (self.nist_df['wavelength'] >= wl_obs - 0.05) & 
                    (self.nist_df['wavelength'] <= wl_obs + 0.05)
                ]
                if not subset.empty:
                    subset = subset.copy()
                    subset['diff'] = abs(subset['wavelength'] - wl_obs)
                    match = subset.sort_values('diff').iloc[0]
                    Aki = float(match['Aki'])
                    gk = float(match['gk'])
                    Ek = float(match['Ek'])
                    nist_wl_val = float(match['wavelength'])
            
            if Aki <= 0 or gk <= 0: continue
            
            # Te Override
            calc_te = saha_te_overrides.get(el, te)
            
            # Boltzmann
            Z = self.z_map.get((s_short, el, ion), 1.0) 
            wl_m = nist_wl_val * 1e-9
            E_J = Ek * 1.60218e-19 
            kT_J = 1.3806e-23 * calc_te
            
            boltzmann_factor = math.exp(E_J / kT_J)
            val = (area * Z) / (Aki * gk) * boltzmann_factor
            
            # Saha
            saha_correction_factor = 1.0
            if saha_elements and el in saha_elements:
                Ne = self.ne_map.get(sample_id) or self.ne_map.get(s_short, 0.0)
                if Ne > 0:
                     IP = IONIZATION_ENERGIES.get(el)
                     if IP:
                        z1 = self.z_map.get((s_short, el, 1), 1.0)
                        z2 = self.z_map.get((s_short, el, 2), 1.0) 
                        if z1 > 0 and z2 > 0:
                            saha_te = saha_te_overrides.get(el, te)
                            R = calculate_saha_ratio(saha_te, Ne, IP, z1, z2)
                            
                            if ion == 1: saha_correction_factor = (1.0 + R)
                            elif ion == 2 and R > 0: saha_correction_factor = (1.0 + (1.0/R))
                            
                            val = val * saha_correction_factor
                            
                            if el not in printed_saha_elements:
                                log.append(f"[Saha] {el}: Te_exc={te:.0f}, Te_ion={saha_te:.0f}, Ne={Ne:.2e}, R={R:.4e}, Factor={saha_correction_factor:.3f}")
                                printed_saha_elements.add(el)
                                
                            saha_tracking[el] = {'Applied': True, 'R': R, 'Factor': saha_correction_factor}
            
            species_key = f"{el} {int(ion)}"
            if species_key not in element_concentrations: element_concentrations[species_key] = []
            element_concentrations[species_key].append(val)
            
            Ne_used = self.ne_map.get(sample_id) or self.ne_map.get(s_short, 0.0)

            # Build enriched row: start with ALL original source columns
            detail_row = dict(row)  # preserves timestamp, file, model, png_path, fwhm, etc.
            # Overwrite/add CFL-specific physics columns
            detail_row.update({
                'cfl_te_K': calc_te,
                'cfl_ne_cm3': Ne_used,
                'cfl_z_partisi': Z,
                'cfl_aki_used': Aki,
                'cfl_gk_used': gk,
                'cfl_ek_ev_used': Ek,
                'cfl_nist_wl_nm': nist_wl_val,
                'cfl_delta_wl_nm': wl_obs - nist_wl_val,
                'cfl_boltzmann_factor': boltzmann_factor,
                'cfl_saha_factor': saha_correction_factor,
                'cfl_val_relatif': val,
            })
            results_detail.append(detail_row)

        # Average species
        final_element_conc = {}
        for sp_key, vals in element_concentrations.items():
            if vals: final_element_conc[sp_key] = sum(vals) / len(vals)
            
        if not final_element_conc: return [], [], log

        # Mass Fractions
        EXCLUDED_ELEMENTS = set() # {'N', 'O', 'Ar', 'H'} - User requested to include them
        total_mass = 0.0
        masses = {}
        
        for sp_key, conc in final_element_conc.items():
            el = sp_key.split()[0]
            if el in EXCLUDED_ELEMENTS: continue
            weight = self.atomic_weights.get(el, 0.0)
            m = conc * weight
            masses[sp_key] = m
            total_mass += m
            
        if total_mass == 0: return [], [], log
        
        # Aggregation
        results = []
        element_masses = {}
        element_species_map = {}
        for sp_key in masses.keys():
            el = sp_key.split()[0]
            if el not in element_species_map: element_species_map[el] = []
            element_species_map[el].append(sp_key)
            
        for el, species_keys in element_species_map.items():
            is_saha = (el in saha_tracking and saha_tracking[el]['Applied'])
            
            if is_saha:
                 # Average estimates
                 total_c = sum(final_element_conc.get(sp, 0.0) for sp in species_keys)
                 avg_conc_atom = total_c / len(species_keys) if species_keys else 0.0
                 element_masses[el] = avg_conc_atom * self.atomic_weights.get(el, 0.0)
            else:
                 # Sum species
                 element_masses[el] = sum(masses[sp] for sp in species_keys)
        
        # Merge XRF and Create Results
        libs_elements = set(element_masses.keys())
        xrf_elements = set()
        if sample_id in self.xrf_quant:
            xrf_elements.update(self.xrf_quant[sample_id]["Elements"].keys())
        all_elements = libs_elements.union(xrf_elements)
        
        # Correction: Recalculate Total Mass from Final Element Masses (not raw species sum)
        # Raw sum acts as sum of parts, but Saha/Avg logic might treat them as estimates of the whole.
        # To ensure percentages sum to 100%, we must normalize against the processed sum.
        total_processed_mass = sum(element_masses.values())
        
        # Calculate Total Moles for Atomic_Percent
        total_moles = 0.0
        for el in all_elements:
            m_el = element_masses.get(el, 0.0)
            w_el = self.atomic_weights.get(el, 0.0)
            if w_el > 0:
                total_moles += (m_el / w_el)
        
        processed_oxides = []
        
        for el in all_elements:
            m_el = element_masses.get(el, 0.0)
            percentage_el = (m_el / total_processed_mass * 100.0) if total_processed_mass > 0 else 0.0
            
            w_el = self.atomic_weights.get(el, 0.0)
            rel_conc_atom = (m_el / w_el) if w_el > 0 else 0.0
            atomic_perc = (rel_conc_atom / total_moles * 100.0) if total_moles > 0 else 0.0
            
            saha_info = saha_tracking.get(el, {'Applied': False, 'R': 0.0, 'Factor': 1.0})
            est_ion_conc = 0.0
            if saha_info['Applied'] and saha_info['R'] > 0:
                est_ion_conc = percentage_el * (saha_info['R'] / (1.0 + saha_info['R']))

            # XRF Lookup
            xrf_conc = np.nan
            diff = np.nan
            xrf_detected = "NO"
            
            # 1. Detection
            if sample_id in self.xrf_qual and el in self.xrf_qual[sample_id]:
                xrf_detected = "YES"

            # 2. Quant
            target_sid = s_short if s_short in self.xrf_quant else sample_id
            if target_sid in self.xrf_quant:
                # Try new flat structure first
                val = self.xrf_quant[target_sid].get(el)
                # Fallback to old nested structure if needed
                if val is None and isinstance(self.xrf_quant[target_sid], dict) and "Elements" in self.xrf_quant[target_sid]:
                    el_data = self.xrf_quant[target_sid]["Elements"].get(el)
                    if el_data: val = el_data.get('Element_Conc')
                
                if val is not None:
                    xrf_conc = float(val)
                    if xrf_conc > 0:
                        diff = percentage_el - xrf_conc

            results.append({
                'Sample': sample_id, 'Type': 'Element', 'Name': el,
                'Mass_Fraction_Percent': percentage_el, 
                'Atomic_Percent': atomic_perc,
                'Mass_Score': m_el,
                'Total_Rel_Conc': rel_conc_atom, 'Is_In_Libs': (el in libs_elements),
                'Saha_Applied': "YES" if saha_info['Applied'] else "NO",
                'Saha_R': saha_info['R'], 'Saha_Factor': saha_info['Factor'],
                'Est_Ion_Conc_%': est_ion_conc,
                'XRF_Conc_%': xrf_conc,
                'Diff_%': diff,
                'XRF_Detected': xrf_detected
            })
            
            # Oxide
            if el in STOICHIOMETRY:
                n_el, n_o = STOICHIOMETRY[el]
                if w_el > 0:
                    mol_w_oxide = (n_el * w_el) + (n_o * W_O)
                    factor = mol_w_oxide / (n_el * w_el)
                    m_ox = m_el * factor
                    fmt = f"{el}{n_el if n_el>1 else ''}O{n_o if n_o>1 else ''}"
                    processed_oxides.append((fmt, m_ox, el in libs_elements, el, mol_w_oxide)) # Store el for XRF lookup

        total_ox_mass = sum(m for _, m, _, _, _ in processed_oxides)
        
        # Calculate Total Oxide Moles
        total_ox_moles = 0.0
        for fmt, m_ox, in_libs, base_el, w_ox in processed_oxides:
            if w_ox > 0:
                total_ox_moles += (m_ox / w_ox)
                
        for fmt, m_ox, in_libs, base_el, w_ox in processed_oxides:
            ox_perc = (m_ox / total_ox_mass * 100.0) if total_ox_mass > 0 else 0.0
            ox_at_perc = ((m_ox / w_ox) / total_ox_moles * 100.0) if (w_ox > 0 and total_ox_moles > 0) else 0.0
            
            # XRF Oxide
            xrf_ox_conc = np.nan
            diff_ox = np.nan
            xrf_det_ox = "NO" # Infer from element
            
            if sample_id in self.xrf_qual and base_el in self.xrf_qual[sample_id]:
                xrf_det_ox = "YES"
                
            target_sid = s_short if s_short in self.xrf_quant else sample_id
            if target_sid in self.xrf_quant:
                # Try new flat structure first
                val = self.xrf_quant[target_sid].get(base_el)
                # Fallback to old nested structure
                if val is None and isinstance(self.xrf_quant[target_sid], dict) and "Elements" in self.xrf_quant[target_sid]:
                    el_data = self.xrf_quant[target_sid]["Elements"].get(base_el)
                    if el_data: val = el_data.get('Oxide_Conc')
                
                if val is not None:
                    xrf_ox_conc = float(val)
                    if xrf_ox_conc > 0:
                        diff_ox = ox_perc - xrf_ox_conc

            results.append({
                'Sample': sample_id, 'Type': 'Oxide', 'Name': fmt,
                'Mass_Fraction_Percent': ox_perc,
                'Atomic_Percent': ox_at_perc,
                'Mass_Score': m_ox, 'Total_Rel_Conc': 0.0, 'Is_In_Libs': in_libs,
                'Saha_Applied': "N/A", 'Saha_R': 0.0, 'Saha_Factor': 1.0,
                'Est_Ion_Conc_%': 0.0,
                'XRF_Oxide_Conc_%': xrf_ox_conc,
                'Diff_%': diff_ox, 
                'XRF_Detected': xrf_det_ox
            })
            
        return results, results_detail, log

    def delete_lines_from_source(self, sample_id, lines_to_delete):
        """
        Permanently delete multiple lines from the source Excel file.
        lines_to_delete: list of tuples (element, wavelength)
        Returns (success, message)
        """
        if sample_id not in self.plot_files_map:
            return False, f"File for {sample_id} not found."
            
        pfile = self.plot_files_map[sample_id]
        try:
            df = pd.read_excel(pfile)
            clean_cols = [c.strip() for c in df.columns]
            col_map = dict(zip(clean_cols, df.columns))
            c_el = col_map.get('element')
            c_wl = col_map.get('center_nm')
            
            if not c_el or not c_wl:
                return False, "Could not find 'element' or 'center_nm' columns."

            # Filter
            # Need fuzzy match on wavelength (float tolerance)
            # targets is list of (element, wavelength)
            
            mask_to_keep = []
            deleted_count = 0
            
            # Mutable copy of targets to handle duplicates count
            # Use list because we might have multiple identical targets
            remaining_targets = list(lines_to_delete) 
            
            for idx, row in df.iterrows():
                el_val = str(row[c_el]).strip()
                try:
                    wl_val = float(row[c_wl])
                except:
                    mask_to_keep.append(True)
                    continue

                match_found = False
                match_index = -1
                
                for i, (t_el, t_wl) in enumerate(remaining_targets):
                    # Strict element match, Fuzzy wavelength match (0.005 nm tolerance)
                    if el_val == t_el and abs(wl_val - t_wl) < 0.005:
                        match_found = True
                        match_index = i
                        break
                
                if match_found:
                    deleted_count += 1
                    mask_to_keep.append(False)
                    # Remove the specific target instance so it doesn't delete subsequent duplicates (unless requested)
                    remaining_targets.pop(match_index)
                else:
                    mask_to_keep.append(True)
            
            if deleted_count == 0:
                return False, f"No lines found matching {lines_to_delete[0]} etc."
                
            df_new = df[mask_to_keep]
            df_new.to_excel(pfile, index=False, engine='openpyxl')
            return True, f"Deleted {deleted_count} line(s) from {os.path.basename(pfile)}"
            
        except Exception as e:
            return False, f"Error deleting lines: {e}"

    def delete_line_from_source(self, sample_id, element, wavelength):
        return self.delete_lines_from_source(sample_id, [(element, wavelength)])

    def reload(self):
        """Reload all data sources dari folder yg aktif."""
        self.plot_files_map.clear()
        self.temp_map.clear()
        self.ne_map.clear()
        self.load_data()
        return f"Reloaded data. Found {len(self.plot_files_map)} samples dari {self.plot_state_dir}."
    
    def set_plot_state_dir(self, folder: Path):
        """Ubah direktori sumber plot_state ke folder lain, lalu reload."""
        self.plot_state_dir = folder
        return self.reload()

def main():
    print("=== Starting Concentration Calculation (Refactored) ===")
    
    parser = argparse.ArgumentParser(description="Calculate CF-LIBS Concentrations.")
    parser.add_argument("--sample", type=str, help="Process only specific sample (e.g., S1)", default=None)
    parser.add_argument("--saha", nargs='+', help="Enable Saha correction for specific elements", default=[])
    parser.add_argument('--saha-te', nargs='+', help='Override Te (e.g. Ag=12000)', default=[])
    args = parser.parse_args()

    saha_te_map = {}
    for item in args.saha_te:
        if '=' in item:
            k, v = item.split('=')
            try: saha_te_map[k.strip()] = float(v)
            except: pass
            
    analyzer = CFLAnalyzer()
    
    targets = sorted(analyzer.plot_files_map.keys())
    if args.sample:
        if args.sample in targets: targets = [args.sample]
        else:
             print(f"Sample {args.sample} not found.")
             sys.exit(0)
             
    all_results = []
    all_details = []
    
    for sid in targets:
        print(f"Processing {sid}...")
        res, det, logs = analyzer.analyze_sample(sid, args.saha, saha_te_map)
        for l in logs: print(f"  {l}")
        all_results.extend(res)
        all_details.extend(det)
        
    if all_results:
        df = pd.DataFrame(all_results)
        
        # XRF Post-Merge (Simulated here or already done in analyze_sample?)
        # analyze_sample already merged XRF data in! 
        # But we need to handle Diff_% and XRF_Conc_% column filling which analyze_sample logic did partially?
        # Actually analyze_sample returned neat rows but didn't fill "XRF_Conc_%" explicitly, it just used it for union.
        # Let's add the column filling logic here based on stored XRF dict in analyzer
        
        df['XRF_Conc_%'] = np.nan
        df['XRF_Oxide_Conc_%'] = np.nan
        df['XRF_Detected'] = "NO"
        df['Diff_%'] = np.nan
        
        for idx, row in df.iterrows():
            sid = row['Sample']
            name = row['Name']
            rtype = row['Type']
            
            # Detection
            target_el = name
            if rtype == 'Oxide':
                 match = re.match(r"([A-Z][a-z]?)", name)
                 if match: target_el = match.group(1)
            
            if sid in analyzer.xrf_qual and target_el in analyzer.xrf_qual[sid]:
                df.at[idx, 'XRF_Detected'] = "YES"
                
            if sid in analyzer.xrf_quant:
                el_data = analyzer.xrf_quant[sid]["Elements"].get(target_el)
                if el_data:
                    if rtype == 'Element':
                         x = el_data.get('Element_Conc')
                         df.at[idx, 'XRF_Conc_%'] = x
                         if x and x > 0: df.at[idx, 'Diff_%'] = row['Mass_Fraction_Percent'] - x
                    elif rtype == 'Oxide':
                         x = el_data.get('Oxide_Conc')
                         df.at[idx, 'XRF_Oxide_Conc_%'] = x
                         if x and x > 0: df.at[idx, 'Diff_%'] = row['Mass_Fraction_Percent'] - x

        # Save Elements
        df_el = df[df['Type']=='Element'].copy().sort_values(['Sample', 'Mass_Fraction_Percent'], ascending=[True, False])
        cols_el = ['Sample', 'Name', 'Total_Rel_Conc', 'Mass_Fraction_Percent', 'XRF_Conc_%', 'Diff_%', 'XRF_Detected', 'Saha_Applied', 'Saha_R', 'Saha_Factor', 'Est_Ion_Conc_%']
        df_el.to_csv(BASE_DIR/"concentration_results_elements.csv", index=False, columns=[c for c in cols_el if c in df_el.columns])
        
        # Save Oxides
        df_ox = df[df['Type']=='Oxide'].copy().sort_values(['Sample', 'Mass_Fraction_Percent'], ascending=[True, False])
        cols_ox = ['Sample', 'Name', 'Mass_Fraction_Percent', 'XRF_Oxide_Conc_%', 'Diff_%', 'XRF_Detected']
        df_ox.to_csv(BASE_DIR/"concentration_results_oxides.csv", index=False, columns=[c for c in cols_ox if c in df_ox.columns])
        
        # Detail
        if all_details:
            pd.DataFrame(all_details).to_csv(BASE_DIR/"concentration_details.csv", index=False)
            
        print("Done. Saved results.")
    else:
        print("No results.")

if __name__ == "__main__":
    main()
