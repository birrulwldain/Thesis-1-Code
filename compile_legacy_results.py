"""
Script untuk merekonsiliasi (mencangkok) data empiris masa lalu dari:
1. Hasil XRF/CF-LIBS Konsentrasi Elemen (Excel Aceh)
2. Hasil Suhu dan Densitas (Saha-Boltzmann Konvensional dari folder raw)
Menjadi satu buah file .pkl dan .h5 yang rapi, mutlak, dan modular.
"""

import pandas as pd
import numpy as np
import joblib
import h5py
import os

def compile_legacy_data():
    excel_path = "data/Hasil CF LIBS_Aceh 2.xlsx"
    saha_path = "raw/Skala-5/c/b_ALL_TeNe_summary.csv"
    output_pkl = "data/ground_truth_legacy.pkl"
    output_hdf5 = "data/ground_truth_legacy.h5"
    
    print("Mengekstrak Konsentrasi Elemen dari Excel...")
    df_excel = pd.read_excel(excel_path, sheet_name=None, header=None)
    
    # 1. Parsing Elemen per Sampel
    sample_data = {}
    for sheet_name, df_sheet in df_excel.items():
        sample_id = f"S{sheet_name}"
        
        # Mencari kolom Element dan Conc % dari Struktur Excel (Toleransi Kolom 10 & 11)
        elements_dict = {}
        try:
            # Drop NaN rows yang tidak relevan
            df_clean = df_sheet.dropna(subset=[10, 11])
            # Filter hanya baris yang kolom 10 nya string (nama elemen)
            df_elements = df_clean[df_clean[10].apply(lambda x: isinstance(x, str) and len(x) <= 2)]
            
            for _, row in df_elements.iterrows():
                el = row[10].strip()
                conc = float(row[11])
                elements_dict[el] = conc
        except Exception as e:
            pass
            
        sample_data[sample_id] = {
            "composition_percent": elements_dict,
            "T_e_K": np.nan,
            "n_e_cm3": np.nan
        }
    
    # 2. Parsing Parameter Makroskopik (T_e, n_e) dari Saha
    print("Mengekstrak Suhu dan Densitas dari b_ALL_TeNe_summary.csv...")
    if os.path.exists(saha_path):
        df_saha = pd.read_csv(saha_path)
        # Kelompokkan per sampel berdasarkan kolom 'sample' (S1, S2, dsb)
        saha_grouped = df_saha.set_index('sample')
        
        for sample_id, row in saha_grouped.iterrows():
            if sample_id in sample_data:
                sample_data[sample_id]["T_e_K"] = float(row["Te_K"])
                sample_data[sample_id]["n_e_cm3"] = float(row["Ne_mean_cm-3"])
                
    # 3. Simpan ke PKL (Python Native Object - Sangat Rapi untuk Dictionary nested)
    print(f"Menyimpan ke dictionary hierarkis: {output_pkl}")
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    joblib.dump(sample_data, output_pkl)
    
    # 4. Simpan ke HDF5 (Untuk Skalabilitas Numerik Murni)
    print(f"Menyimpan ke HDF5 binar murni: {output_hdf5}")
    with h5py.File(output_hdf5, 'w') as f:
        for s_id, s_info in sample_data.items():
            grp = f.create_group(s_id)
            grp.attrs['T_e_K'] = s_info['T_e_K']
            grp.attrs['n_e_cm3'] = s_info['n_e_cm3']
            
            # Konversi dictionary komposisi menjadi array
            if s_info['composition_percent']:
                elements = list(s_info['composition_percent'].keys())
                concs = list(s_info['composition_percent'].values())
                
                # Numpy structured array
                dt = np.dtype([('Element', 'S2'), ('Concentration_Percent', np.float32)])
                comp_array = np.array(list(zip(elements, concs)), dtype=dt)
                grp.create_dataset('Composition', data=comp_array)

    print("✅ PENGGABUNGAN SUKSES! Data legacy kini terbungkus rapi, siap dipanggil 1 baris kode saja.")
    
if __name__ == '__main__':
    compile_legacy_data()
