import pandas as pd
import numpy as np

xls = pd.ExcelFile('data/Hasil CF LIBS_Aceh 2.xlsx')

target_elements = ['Si', 'Fe', 'Al', 'Ca']
xrf_data = {el: [] for el in target_elements}
cf_data  = {el: [] for el in target_elements}

for sheet in xls.sheet_names:
    df = pd.read_excel('data/Hasil CF LIBS_Aceh 2.xlsx', sheet_name=sheet, header=None)
    
    # XRF data calculation loop
    for _, row in df.iterrows():
        xrf_el = str(row[0]).strip() if not pd.isna(row[0]) else str(row[1]).strip()
        cf_el = str(row[10]).strip()
        
        try:
            xrf_val = float(row[2]) if str(row[2]).replace('.','',1).isdigit() else np.nan
        except: xrf_val = np.nan
        
        try:
            cf_val = float(row[11]) if str(row[11]).replace('.','',1).isdigit() else np.nan
        except: cf_val = np.nan
        
        if xrf_el in target_elements and not np.isnan(xrf_val):
            xrf_data[xrf_el].append(xrf_val)
        if cf_el in target_elements and not np.isnan(cf_val):
            cf_data[cf_el].append(cf_val)

print("=== AVERAGE CONCENTRATION (S1-S24) ===\n")
print(f"{'Element':<10} | {'XRF (%)':<15} | {'CF-LIBS (%)':<15} | {'Diff (%)':<10}")
print("-" * 60)

total_xrf = 0
total_cf = 0

for el in target_elements:
    m_xrf = np.mean(xrf_data[el]) if xrf_data[el] else 0.0
    m_cf = np.mean(cf_data[el]) if cf_data[el] else 0.0
    diff = abs(m_xrf - m_cf)
    print(f"{el:<10} | {m_xrf:<15.2f} | {m_cf:<15.2f} | {diff:<10.2f}")
    total_xrf += m_xrf
    total_cf += m_cf
    
print("\n=== STOICHIOMETRIC FRACTION IN 4-ELEMENT PLASMA ===")
for el in target_elements:
    m_xrf = np.mean(xrf_data[el]) if xrf_data[el] else 0.0
    m_cf = np.mean(cf_data[el]) if cf_data[el] else 0.0
    frac_xrf = m_xrf / total_xrf if total_xrf > 0 else 0
    frac_cf = m_cf / total_cf if total_cf > 0 else 0
    print(f"{el:<4}: XRF={frac_xrf:.2f}, CF={frac_cf:.2f}")
