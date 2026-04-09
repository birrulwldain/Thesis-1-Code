import numpy as np
import plotly.graph_objects as go
from src import sim  # Legacy Model
from src import libs_physics as cr  # New Q1 Inversion Engine
import h5py
import pandas as pd
from typing import List, Tuple

def get_legacy_spectrum(selected_elements: List[Tuple[str, float]], 
                        temperature: float, 
                        electron_density: float):
    """Mengeksekusi simulasi menggunakan sim.py (Legacy LTE)"""
    # 1. Setup Data
    nist_path = "data/nist_data_hog.h5"
    atomic_data_path = "data/atomic_data1.h5"
    
    # 2. Get Ionization Energies
    ion_energies = {}
    with h5py.File(atomic_data_path, 'r') as f:
        dset = f['elements']
        cols = dset.attrs['columns']
        for item in dset[:]:
            ion_energies[item[1].decode('utf-8')] = float(item[5]) # Sp. Name -> Ion Energy

    # 3. Create Simulators
    fetcher = sim.DataFetcher(nist_path)
    simulators = []
    delta_E_max_dict = {}
    
    for elem, pct in selected_elements:
        for ion in [1, 2]:
            nist_data, delta_E = fetcher.get_nist_data(elem, ion)
            if not nist_data: continue
            
            delta_E_max_dict[f"{elem}_{ion}"] = delta_E
            ion_energy = ion_energies.get(f"{elem} {'I' if ion == 1 else 'II'}", 0.0)
            simulator = sim.SpectrumSimulator(nist_data, elem, ion, temperature, ion_energy)
            simulators.append(simulator)

    # 4. Inject into sim's global namespace to avoid NameError
    sim.ionization_energies = ion_energies

    # 5. Run Generation
    mixed_sim = sim.MixedSpectrumSimulator(simulators, electron_density, delta_E_max_dict)
    wl, spec, _, _ = mixed_sim.generate_spectrum(selected_elements, temperature)
    return wl, spec

def run_comparison():
    print("="*80)
    print("  LIBS MODEL COMPARISON: CR (New) vs LTE (Legacy)")
    print("="*80)
    
    # Standard Q1 Geologic Composition
    selected = [("Si", 42.0), ("Fe", 39.0), ("Al", 12.0), ("Ca", 7.0)]
    temperature = 14000.0
    n_e = 1e17
    
    print(f"\n[1/2] Menjalankan Model Legacy (sim.py / LTE)...")
    wl_lte, spec_lte = get_legacy_spectrum(selected, temperature, n_e)
    print(f"      -> LTE Max: {np.max(spec_lte):.4f}")
    
    print(f"[2/2] Menjalankan Model Deterministik (libs_physics.py / Two-Zone CR)...")
    # ... (rest of logic)
    fetcher_cr = cr.DataFetcher()
    # Saha splitting for CR inputs
    expanded_elements = []
    for elem, pct in selected:
        frac_neu, frac_ion = cr.compute_saha_ionization_fractions(elem, pct/100.0, temperature, n_e, fetcher_cr)
        if frac_neu > 1e-4: expanded_elements.append((elem, 1, frac_neu))
        if frac_ion > 1e-4: expanded_elements.append((elem, 2, frac_ion))
    
    core = cr.PlasmaZoneParams(T_e_K=temperature, T_i_K=temperature-2000, n_e_cm3=n_e, thickness_m=1e-6, label="Core")
    shell = cr.PlasmaZoneParams(T_e_K=temperature*0.5, T_i_K=temperature*0.5-1000, n_e_cm3=n_e*0.1, thickness_m=1e-6, label="Shell")
    
    model_cr = cr.TwoZonePlasma(core, shell, expanded_elements, fetcher_cr)
    wl_cr, spec_cr, meta = model_cr.run()

    # 5. Global Normalization to 1.0 for better comparison of line shapes
    if np.max(spec_lte) > 0: spec_lte /= np.max(spec_lte)
    if np.max(spec_cr) > 0: spec_cr /= np.max(spec_cr)

    # 6. Plotting Comparison
    fig = go.Figure()
    
    # LTE Plot - BLUE for clear contrast
    fig.add_trace(go.Scatter(
        x=wl_lte, y=spec_lte,
        name='LTE Model (sim.py - Baseline)',
        line=dict(color='royalblue', width=1.5),
        opacity=0.8
    ))
    
    # CR Plot - RED for current Q1 engine
    fig.add_trace(go.Scatter(
        x=wl_cr, y=spec_cr,
        name='Two-Zone CR Model (Q1 Deterministic)',
        line=dict(color='crimson', width=2.0)
    ))

    element_str = " ".join([f"{s}{p:.0f}%" for s,p in selected])
    # Find zoom range based on peaks (e.g., 200-500 nm often contains most lines)
    fig.update_layout(
        title=f"<b>CR (Non-LTE) vs LTE Comparison</b><br>{element_str} | T={temperature} K | n_e={n_e:.1e} cm⁻³",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Normalized Intensity (Relative Shape)",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)', bordercolor='Black', borderwidth=1),
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray', range=[200, 500]), # Zoom to active region
        yaxis=dict(showgrid=True, gridcolor='lightgray', range=[-0.05, 1.1])
    )
    
    fig.show()
    print("\n[DONE] Skrip komparasi selesai. Silakan periksa browser untuk visualisasi.")

if __name__ == "__main__":
    run_comparison()
