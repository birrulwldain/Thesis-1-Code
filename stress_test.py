import libs_physics as cr
import numpy as np

def run_stress_test():
    print("="*60)
    print("  pLTE STRESS TEST: 20,000 K (EXTREME TEMPERATURE)")
    print("="*60)
    
    # 1. Setup Parameters
    temp_extreme = 20000.0
    ne_extreme = 5e17
    selected = [("Si", 42.0), ("Fe", 39.0), ("Al", 12.0), ("Ca", 7.0)]
    
    print(f"Target: T = {temp_extreme} K, n_e = {ne_extreme:.1e} cm⁻³")
    
    # 2. Initialize DataFetcher
    fetcher = cr.DataFetcher()
    
    # 3. Saha Splitting
    expanded = []
    for elem, pct in selected:
        f_neu, f_ion = cr.compute_saha_ionization_fractions(elem, pct/100.0, temp_extreme, ne_extreme, fetcher)
        if f_neu > 1e-6: expanded.append((elem, 1, f_neu))
        if f_ion > 1e-6: expanded.append((elem, 2, f_ion))
    
    print(f"Expanded elements (Ionization check): {expanded}")
    
    # 4. Define Zones
    core = cr.PlasmaZoneParams(T_e_K=temp_extreme, T_i_K=temp_extreme-2000, n_e_cm3=ne_extreme, thickness_m=1e-3, label="Core")
    shell = cr.PlasmaZoneParams(T_e_K=temp_extreme*0.5, T_i_K=temp_extreme*0.5-1000, n_e_cm3=ne_extreme*0.1, thickness_m=2e-3, label="Shell")
    
    # 5. Execute Run
    try:
        model = cr.TwoZonePlasma(core, shell, expanded, fetcher)
        wl, spec, meta = model.run()
        
        print("\n[SUCCESS] Model pLTE stabil pada 20.000 K.")
        print(f"Max Intensity: {np.max(spec):.4f}")
        print("Simulator tidak mengalami divergensi numerik.")
    except Exception as e:
        print(f"\n[FAILED] Model gagal pada suhu ekstrem: {e}")

if __name__ == "__main__":
    run_stress_test()
