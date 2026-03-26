"""
libs_inversion.py — Inversion Engine 
==================================================================
Q1 Thesis: Inverse Model for extracting plasma parameters (T_e, n_e)
from an observed LIBS spectrum using the forward model in libs_physics.py.

Design Principle:
  - This module is STRICTLY separated from libs_physics.py.
  - Machine learning (SVR) is used ONLY as a mathematical inversion 
    tool mapping simulated macroscopic spectra to microscopic parameters.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import warnings
from itertools import product
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.libs_physics import (
    DataFetcher, PlasmaZoneParams, TwoZonePlasma, SIMULATION_CONFIG,
    instrumental_broadening
)

class ForwardModelWrapper:
    """
    Wraps TwoZonePlasma as a function θ → I_obs(λ) for use in inversion.
    """

    def __init__(self, elements: list, fetcher=None, fwhm_nm: float = 0.5):
        self.elements = elements
        self.fetcher  = fetcher or DataFetcher()
        self.fwhm_nm  = fwhm_nm

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """
        Evaluate forward model at parameter vector θ.

        Args:
            theta: [T_e_core_K, T_e_shell_K, n_e_core_cm3, n_e_shell_cm3]
        Returns:
            I_obs: np.ndarray [N_wavelengths] (normalized)
        """
        T_core, T_shell, ne_core, ne_shell = theta
        
        # Geometrical thicknesses (fixed for this MVP inversion scheme to avoid underdetermination)
        d_core_m  = 1e-3
        d_shell_m = 2e-3
        
        core = PlasmaZoneParams(T_e_K=T_core, T_i_K=T_core*0.8, n_e_cm3=ne_core, thickness_m=d_core_m, label='Core')
        shell = PlasmaZoneParams(T_e_K=T_shell, T_i_K=T_shell*0.8, n_e_cm3=ne_shell, thickness_m=d_shell_m, label='Shell')
        
        model = TwoZonePlasma(core, shell, self.elements, self.fetcher)
        wl, I_raw, meta = model.run()
        
        # Apply instrumental broadening
        if self.fwhm_nm > 0.0:
            I_sim = instrumental_broadening(I_raw, wl, self.fwhm_nm)
        else:
            I_sim = I_raw
            
        m = I_sim.max()
        if m > 0.0:
            I_sim = I_sim / m
            
        return I_sim


class GridSearchInverter:
    """
    Exhaustive grid search over parameter space to find (T_e, n_e) that
    minimises the Mean Squared Error against the measured spectrum.
    Used for baseline validation.
    """

    def __init__(self, forward_model: ForwardModelWrapper):
        self.forward_model = forward_model

    def fit(self, I_measured: np.ndarray, param_grid: Dict[str, np.ndarray]) -> Dict:
        """
        Args:
            I_measured : Observed (or synthetic) spectrum [N_λ]
            param_grid : Dict of parameter ranges
        Returns:
            Dict with best-fit parameter values and minimal MSE
        """
        keys = ['T_e_core', 'T_e_shell', 'n_e_core', 'n_e_shell']
        for k in keys:
            if k not in param_grid:
                raise ValueError(f"param_grid must contain '{k}'")
                
        best_mse = np.inf
        best_params = None
        best_I = None
        
        # Evaluate all permutations
        n_iters = len(param_grid['T_e_core']) * len(param_grid['T_e_shell']) * \
                  len(param_grid['n_e_core']) * len(param_grid['n_e_shell'])
        print(f"[GridSearchInverter] Evaluating {n_iters} configurations...")

        for T_c, T_s, ne_c, ne_s in product(param_grid['T_e_core'], param_grid['T_e_shell'], 
                                            param_grid['n_e_core'], param_grid['n_e_shell']):
            theta = np.array([T_c, T_s, ne_c, ne_s])
            I_sim = self.forward_model(theta)
            
            mse = np.mean((I_measured - I_sim)**2)
            if mse < best_mse:
                best_mse = mse
                best_params = {'T_e_core': T_c, 'T_e_shell': T_s, 'n_e_core': ne_c, 'n_e_shell': ne_s}
                best_I = I_sim
                
        return {'params': best_params, 'mse': best_mse, 'I_sim': best_I}


class SVRInverter:
    """
    Support Vector Regression inverter.
    
    Training procedure:
      1. Sample N parameter vectors θ uniformly from parameter space.
      2. Evaluate forward_model(θ) → I_obs_features for each θ.
      3. Apply PCA for dimensionality reduction (24480 λ points → 50 principal components).
      4. Train SVR: I_pca → θ.
    """

    def __init__(self, forward_model: ForwardModelWrapper, use_pca: bool = True, n_components: int = 50):
        self.forward_model = forward_model
        
        # Scikit-learn models
        self.model = MultiOutputRegressor(SVR(kernel='rbf', C=100.0, gamma='scale', epsilon=0.01))
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.use_pca = use_pca
        if self.use_pca:
            self.pca = PCA(n_components=n_components)
        
    def generate_training_data(self, n_samples: int, param_bounds: Dict[str, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate (X=spectra, y=params) via Random Uniform Sampling."""
        print(f"[SVRInverter] Generating {n_samples} synthetic spectra for training...")
        X_list = []
        y_list = []
        
        keys = ['T_e_core', 'T_e_shell', 'n_e_core', 'n_e_shell']
        for i in range(n_samples):
            theta = np.zeros(4)
            for j, k in enumerate(keys):
                lower, upper = param_bounds[k]
                theta[j] = np.random.uniform(lower, upper)
                
            I_sim = self.forward_model(theta)
            X_list.append(I_sim)
            y_list.append(theta)
            
            if (i+1) % max(1, n_samples // 10) == 0:
                print(f"  ... {i+1}/{n_samples} generated")
                
        return np.array(X_list), np.array(y_list)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Pipeline: StandardScaler -> PCA -> SVR."""
        print("[SVRInverter] Training SVR Model...")
        X_features = X
        if self.use_pca:
            print(f"  ... applying PCA (components={self.pca.n_components})")
            X_features = self.pca.fit_transform(X_features)
            
        X_scaled = self.scaler_X.fit_transform(X_features)
        y_scaled = self.scaler_y.fit_transform(y)
        
        self.model.fit(X_scaled, y_scaled)
        print("[SVRInverter] Training complete.")

    def predict(self, I_measured: np.ndarray) -> Dict:
        """Predict plasma parameters for a measured spectrum."""
        if not hasattr(self.scaler_X, 'mean_'):
            raise ValueError("SVR model must be trained before calling predict().")
            
        X_features = I_measured.reshape(1, -1)
        if self.use_pca:
            X_features = self.pca.transform(X_features)
            
        X_scaled = self.scaler_X.transform(X_features)
        y_pred_scaled = self.model.predict(X_scaled)
        theta_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0]
        
        return {
            'T_e_core' : theta_pred[0],
            'T_e_shell': theta_pred[1],
            'n_e_core' : theta_pred[2],
            'n_e_shell': theta_pred[3]
        }


# =============================================================================
# DEMONSTRATION RUN
# =============================================================================
if __name__ == "__main__":
    def run_inversion_demo():
        print("=== Two-Zone CR-LIBS Inversion Engine Demo ===\n")
        
        # 1. Setup Ground Truth & Wrapper
        elements = [('Si', 1, 0.4), ('Al', 1, 0.6)]
        wrapper  = ForwardModelWrapper(elements, fwhm_nm=0.5)
        
        truth_theta = np.array([12000.0, 7000.0, 2e17, 5e15])
        print(f"Ground Truth Parameters:\n  T_core = {truth_theta[0]} K, T_shell = {truth_theta[1]} K")
        print(f"  n_e_core = {truth_theta[2]:.1e} cm^-3, n_e_shell = {truth_theta[3]:.1e} cm^-3\n")
        
        # Synthesize "experimental" data (with slight noise)
        I_measured = wrapper(truth_theta)
        noise = np.random.normal(0, 0.02, size=I_measured.shape)
        I_measured = np.clip(I_measured + noise, 0, 1)
        
        # 2. Grid Search (Baseline)
        print("--- 1. Baseline: Grid Search ---")
        param_grid = {
            'T_e_core': np.array([12000.0, 14000.0]),
            'T_e_shell': np.array([7000.0]),
            'n_e_core': np.array([1e17, 2e17]),
            'n_e_shell': np.array([5e15])
        }
        grid_inv = GridSearchInverter(wrapper)
        res_grid = grid_inv.fit(I_measured, param_grid)
        print(f"Grid Search Best Fit MSE: {res_grid['mse']:.4e}")
        for k, v in res_grid['params'].items():
            print(f"  {k} = {v:.2e}")
            
        print("\n--- 2. Machine Learning: PCA + SVR ---")
        # Define bounds for random sampling
        bounds = {
            'T_e_core' : (9000.0, 15000.0),
            'T_e_shell': (5000.0, 9000.0),
            'n_e_core' : (1e17, 5e17),
            'n_e_shell': (1e15, 1e16)
        }
        
        svr_inv = SVRInverter(wrapper, use_pca=True, n_components=10)
        
        # Generate relatively small dataset for quick demo (n=15)
        X_train, y_train = svr_inv.generate_training_data(n_samples=15, param_bounds=bounds)
        svr_inv.train(X_train, y_train)
        
        pred = svr_inv.predict(I_measured)
        print("\nSVR Predictions:")
        for k, v in pred.items():
            print(f"  {k} = {v:.2e}  (Truth: {truth_theta[['T_e_core','T_e_shell','n_e_core','n_e_shell'].index(k)]:.2e})")

    run_inversion_demo()
