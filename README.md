# Deterministic Two-Zone CR-LIBS Engine 🔬💻

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)

**Computational Physics Engine & Machine Learning Inversion for Laser-Induced Breakdown Spectroscopy (LIBS)**.

This repository embodies the complete computational architecture for a Q1 Master's Thesis. It systematically overthrows the traditional Saha/Boltzmann Local Thermodynamic Equilibrium (LTE) assumptions by deploying a rigorous **Collisional-Radiative (CR)** deterministic solver conjointly with an explicit **Support Vector Regression (SVR)** inversion pipeline.

## 🏗️ The 4-Block Architecture

### BLOK 1: Deterministic Physics Engine (`libs_physics.py`)
The thermodynamic core. It orchestrates the stiff CR ODE matrix and solves the atomic level configurations precisely using an implicit Radau/BDF method with an analytical Jacobian. It subsequently computes Voigt profiles and solves the 1D Radiative Transfer Equation (RTE) spanning a two-zone plasma geometry (Core + Shell) to meticulously simulate intense self-absorption.

### BLOK 2: Synthetic Data Factory (`generate_dataset.py`)
Leverages brute-force stochastic parameter sampling mapped across full CPU hardware parallelism (`multiprocessing`). It executes thousands of synthetic collisions, injecting realistic Gaussian sensor noise and spectrometer resolution profiles, rendering highly efficient clustered training databases in HDF5 (`.h5`).

### BLOK 3: ML Inversion Engine (`train_inversion_model.py`)
A computationally un-biased artificial intelligence un-mixer. It compresses the dense physical geometries (24,480+ spectral pixels) through PCA, establishes scaled representations, and trains a highly robust non-linear `MultiOutputRegressor(SVR)`. It isolates mappings from a convolution of observed spectra explicitly down to physical micro-parameters ($T_e$, $n_e$).

### BLOK 4: Empirical Validation (`empirical_validation.py`)
The empirical trial. Synthesizes field experiments by ingesting untouched `.csv` outputs generated from real spectrometers. Validates inferences seamlessly utilizing the stored ML pipeline whilst calculating the final metric required to decisively prove physical compliance: **The McWhirter Criterion**.

## 🚀 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/birrulwldain/Thesis-1-Code.git
cd Thesis-1-Code

# Install dependencies
pip install -r requirements.txt

# Install Git hooks (for Conventional Commits enforcement)
pre-commit install --hook-type commit-msg
```

## 🧠 Usage Workflow

1. **Verify the Physics (Blok 1):**  
   Run `python libs_physics.py` to test the deterministic ODE solver under a sample state and generate spectral visualization.
2. **Stockpile Training Data (Blok 2):**  
   Run `python generate_dataset.py --samples 500 --cores 8`. This constructs the mass-tensor repository `dataset_synthetic.h5`.
3. **Train the SVR Architecture (Blok 3):**  
   Run `python train_inversion_model.py --dataset dataset_synthetic.h5`. This evaluates and persists `model_inversi_svr.pkl`.
4. **Break the Illusion (Blok 4):**  
   Run `python empirical_validation.py --model model_inversi_svr.pkl --csv data_eksperimen.csv` to conclude the hypothesis.

---
**Author:** Birrul Walidain  
**Domain:** Computational Plasma Physics & Machine Learning Spectroscopy
