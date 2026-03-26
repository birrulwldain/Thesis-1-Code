import os
import yaml
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, TransformerMixin

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(_CONFIG_PATH, 'r') as f:
    _CONFIG = yaml.safe_load(f)

class PhysicsFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Ekstraksi Fitur Fisika Determinis (White-Box) dari spektrum LIBS.
    Mentransformasi spektrum N-piksel menjadi 4 fitur fisik:
    1. FWHM Stark (Kerapatan Elektron, n_e)
    2. Rasio Boltzmann (Suhu Elektron, T_e)
    3. Kemiringan Kontinuum Wien (Temperatur Global)
    4. Kedalaman Self-Reversal (Ketebalan Optik)
    """
    def __init__(self, wavelengths: np.ndarray):
        self.wavelengths = wavelengths

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_samples = X.shape[0]
        features = np.zeros((n_samples, 4), dtype=np.float64)
        
        cfg = _CONFIG['physics_extraction']
        c_wins = cfg['continuum_windows']
        b_wins = cfg['boltzmann_windows']
        s_win  = cfg['stark_window']
        
        # 1. Prekomputasi Kontinuum (Line-free windows)
        idx_cont1 = np.where((self.wavelengths >= c_wins[0][0]) & (self.wavelengths <= c_wins[0][1]))[0]
        idx_cont2 = np.where((self.wavelengths >= c_wins[1][0]) & (self.wavelengths <= c_wins[1][1]))[0]
        lam_c1 = np.mean(self.wavelengths[idx_cont1]) if len(idx_cont1) > 0 else (c_wins[0][0]+c_wins[0][1])/2.0
        lam_c2 = np.mean(self.wavelengths[idx_cont2]) if len(idx_cont2) > 0 else (c_wins[1][0]+c_wins[1][1])/2.0
        delta_lam_cont = lam_c2 - lam_c1
        
        # 2. Prekomputasi Rasio Boltzmann
        idx_al1 = np.where((self.wavelengths >= b_wins[0][0]) & (self.wavelengths <= b_wins[0][1]))[0]
        idx_al2 = np.where((self.wavelengths >= b_wins[1][0]) & (self.wavelengths <= b_wins[1][1]))[0]
        wl_al1 = self.wavelengths[idx_al1] if len(idx_al1) > 0 else np.array([(b_wins[0][0]+b_wins[0][1])/2.0])
        wl_al2 = self.wavelengths[idx_al2] if len(idx_al2) > 0 else np.array([(b_wins[1][0]+b_wins[1][1])/2.0])
        
        # 3. Prekomputasi FWHM Stark dan Self-Reversal
        idx_ca = np.where((self.wavelengths >= s_win[0]) & (self.wavelengths <= s_win[1]))[0]
        wl_ca  = self.wavelengths[idx_ca] if len(idx_ca) > 0 else np.array([])
        dl = self.wavelengths[1] - self.wavelengths[0] if len(self.wavelengths) > 1 else 1.0
        
        for i in range(n_samples):
            spec = X[i]
            
            # --- FITUR 3: Kemiringan Kontinuum (ΔI/Δλ) ---
            cont_1 = np.mean(spec[idx_cont1]) if len(idx_cont1) > 0 else 0.0
            cont_2 = np.mean(spec[idx_cont2]) if len(idx_cont2) > 0 else 0.0
            slope = (cont_2 - cont_1) / delta_lam_cont if delta_lam_cont != 0 else 0.0
            features[i, 2] = slope
            
            # --- FITUR 2: Rasio Boltzmann (Area Integral I_1 / I_2) ---
            area1 = np.trapezoid(spec[idx_al1], wl_al1) if len(idx_al1) > 2 else 0.0
            area2 = np.trapezoid(spec[idx_al2], wl_al2) if len(idx_al2) > 2 else 0.0
            ratio = area1 / (area2 + 1e-12)
            features[i, 1] = ratio
            
            # --- FITUR 1: FWHM Stark & FITUR 4: Kedalaman Self-Reversal ---
            fwhm = 0.0
            self_reversal = 0.0
            
            if len(idx_ca) > 5:
                local_spec = spec[idx_ca]
                peak_idx_local = np.argmax(local_spec)
                max_val = local_spec[peak_idx_local]
                base_val = min(local_spec[0], local_spec[-1])
                half_max = base_val + (max_val - base_val) / 2.0
                
                try:
                    spline = UnivariateSpline(wl_ca, local_spec - half_max, s=0)
                    roots = spline.roots()
                    if len(roots) >= 2:
                        fwhm = roots[-1] - roots[0]
                except Exception:
                    pass
                
                peak_idx_global = idx_ca[peak_idx_local]
                if 2 <= peak_idx_global < len(self.wavelengths) - 2:
                    d2 = (spec[peak_idx_global + 1] - 2*spec[peak_idx_global] + spec[peak_idx_global - 1]) / (dl**2)
                    self_reversal = d2
                    
            features[i, 0] = fwhm
            features[i, 3] = self_reversal

        return features
