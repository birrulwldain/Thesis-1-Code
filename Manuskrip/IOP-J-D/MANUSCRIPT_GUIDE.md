# Panduan Manuskrip: CNN–Transformer Encoder–Decoder for LIBS

## Status Dokumen
- **Target jurnal:** IOP Machine Learning: Science and Technology (ML:ST)
- **Status:** DRAFT — hanya sampai §2 (Methodology). §3 Results dan §4 Conclusions masih kosong.
- **Tujuan saat ini:** Diperiksa oleh dosen sebagai reviewer proposal tesis.
- **File utama:** `iopjournal-template.tex` (manuskrip), `revv.bib` (bibliografi)
- **Kompilasi:** `pdflatex` → `bibtex` → `pdflatex` × 2. Terakhir sukses: 9 halaman, 0 error.

---

## 1. Konteks Penelitian

### 1.1 Apa yang sudah dilakukan
- **Data XRF** sudah tersedia (24 sampel tanah vulkanik Gunung Seulawah Agam)
- **Data LIBS** sudah diakuisisi (spektra eksperimental)
- **CF-LIBS analysis** sedang berlangsung (belum selesai)
- **Model CNN–Transformer** sedang dibangun (belum ditraining)

### 1.2 Apa yang belum dilakukan
- Training model (hyperparameter di tabel adalah PLACEHOLDER dari literatur)
- Evaluasi model (RMSE, R², MAPE)
- Perbandingan baseline (PLS, CNN-only, Transformer-only, Informer)
- §3 Results & Discussion
- §4 Conclusions
- Figure 1 (framework diagram) — masih di-comment out karena TikZ rusak

### 1.3 Rencana publikasi ganda
- **Paper A (belum submit):** CF-LIBS analysis — detail preparasi sampel, analisis spektral, hasil CF-LIBS
- **Paper B (ini):** CNN–Transformer architecture — fokus pada model DL, bukan analisis spektral

Menggunakan dataset yang SAMA di dua paper adalah **acceptable** karena kontribusi berbeda. Jika Paper A dipublikasi duluan, Paper B harus cite Paper A. Saat ini Paper A belum di-submit, jadi detail eksperimen ditulis langsung di Paper B.

---

## 2. Data & Parameter Simulasi

### 2.1 Data Eksperimental
| Parameter | Nilai |
|-----------|-------|
| Sumber sampel | Gunung Seulawah Agam, Aceh, Indonesia |
| Jumlah sampel | 24 (S-1 – S-24) |
| Arah sampling | 4 (North, West, South, East) |
| Kedalaman | 3 level (0–20, 20–40, 40–60 cm) |
| Laser | Q-switched Nd:YAG, 1064 nm |
| Energi pulsa | 114 mJ |
| Gate delay | 0.5 μs |
| Gate width | 0.5 μs |
| Spectrometer | Echelle |
| Detector | OMA (Optical Multichannel Analyser) |
| Shots per sample | 3 (di-average) |
| Rentang spektral | 200–900 nm |
| Ground truth | XRF (% berat) |

**PENTING:** Ada juga data di 57 mJ (3 repetisi), tapi TIDAK digunakan untuk paper ini. Hanya data 114 mJ yang dipakai.

### 2.2 Data Sintetis
| Parameter | Nilai |
|-----------|-------|
| Electron temperature $T_e$ | 6,000–15,000 K |
| Electron density $n_e$ | 10¹⁶–10¹⁷ cm⁻³ |
| Resolusi spektral $Δλ$ | 0.02 nm |
| Jumlah spectral channels | ~35,000 |
| Jumlah spektra sintetis | 10,000 |
| Sampling method | Uniform random |
| Train/test split | 80/20 |
| Elemen disimulasikan | 7 major: Si, Al, Fe, Ca, Mg, Na, K |

### 2.3 Komposisi Elemen (% berat)
| Element | Range (%) | Notes |
|---------|-----------|-------|
| Si | 40–60 | Dominant silicate |
| Al | 10–20 | Feldspar minerals |
| Fe | 5–15 | Iron oxides |
| Ca | 3–10 | Plagioclase/carbonate |
| Mg | 2–8 | Olivine/pyroxene |
| Na | 1–5 | Alkali feldspar |
| K | 1–5 | Potassium feldspar |

Trace elements (Mn, P, Ba, Sr, Zn, Cu, Ni, Cr, V, Rb, Pb, Co, Mo, Y, Zr) ada di data tapi TIDAK disimulasikan untuk model ini.

### 2.4 Sumber data detail
- File XRF: `raw/Ringkasan_XRF_CFLIBS_Semua_Sampel.xlsx`
  - Sheet: "XRF & CF-LIBS — Seulawah Agam"
  - Header: 24 sampel, 4 arah, konsentrasi % berat

---

## 3. Arsitektur Model

### 3.1 Pipeline
```
OFFLINE (Data Generation):
  NIST ASD → Saha–Boltzmann Forward Model → Monoatomic spectra (per element)
                                           → Polyatomic spectra (mixture)

ONLINE (Model):
  Mono spectra → 1D CNN → Transformer Encoder ──cross-attention──┐
  Poly spectrum → 1D CNN → Transformer Decoder ←─────────────────┘
                                    ↓
                          3 Output Heads:
                          ├── Concentration (Softplus)
                          ├── Spectrum Reconstruction (Sigmoid)
                          └── Physics Parameters (Softplus)

DOMAIN ADAPTATION:
  Shared Encoder → GRL → Domain Classifier (synthetic vs experimental)
```

### 3.2 Hyperparameter (PLACEHOLDER — harus diganti setelah training)
| Parameter | Nilai | Sumber |
|-----------|-------|--------|
| CNN layers | 3 (kernel: 7, 5, 3) | Best practice |
| CNN filters | 64, 128, 128 | Progressive design |
| Model dimension $d$ | 128 | Liu 2026 |
| Attention heads $h$ | 8 | Standard ($d/h = 16$) |
| Encoder layers $N_\text{enc}$ | 4 | Standard Transformer |
| Decoder layers $N_\text{dec}$ | 4 | Match encoder |
| Dropout | 0.1 | Standard |
| Batch size | 32 | |
| Pre-training epochs | 200 | |
| Fine-tuning epochs | 50 | |
| LR (pre-train) | 10⁻⁴ | |
| LR (fine-tune) | 10⁻⁵ | Factor 10 reduction |
| Reconstruction weight α | 0.1 | Auxiliary loss convention |
| GRL weight λ | 0.1 | Ganin 2016 default |

### 3.3 Loss Function
```
L = (1/Z) Σ(ĉ_z - c_z)² + α·‖S_poly - Ŝ_poly‖²
```
- Term 1: MSE konsentrasi
- Term 2: Spectral reconstruction regularisation (α = 0.1)

### 3.4 Baselines (4 model)
1. **PLS** — Partial Least Squares [Wold 2001]
2. **CNN-only** — CNN feature extractor + linear head (no Transformer)
3. **Transformer-only** — Transformer encoder on raw polyatomic (no CNN, no mono-poly decomposition)
4. **Informer** — ProbSparse self-attention [Zhou 2021], adapted per [Walidain 2026]

**CATATAN:** TransCNN (Liu 2026) pernah ada sebagai baseline ke-5 tapi di-comment out. Teks sudah diperbaiki ke "Four baselines".

---

## 4. Struktur Manuskrip

```
Title
Authors & Affiliations
Abstract (proposal-style: "proposes", "will be", "are planned")
Keywords

§1 Introduction
  ├── Paragraf 1: General Background (LIBS, tanah vulkanik, CF-LIBS, LTE)
  ├── Paragraf 2: Previous Findings (MERLIN, ML/DL, CNN-Transformer, gap mono-poliatom)
  └── Paragraf 3: State of the Art (kontribusi paper ini, 3 poin)

§2 Methodology
  ├── §2.1 Framework overview (4 kalimat, two-stage workflow)
  ├── §2.2 Synthetic data generation (persamaan Saha, Boltzmann, emisi, Voigt, parameter grid)
  │   └── Table 3: Composition ranges
  ├── §2.3 Experimental data (lokasi, instrumen, XRF ground truth)
  │   └── Table 1: Experimental parameters (13 baris)
  ├── §2.4 CNN–Transformer architecture
  │   ├── Figure 1: Architecture PNG (img/Arsitektur-PCST-P5.drawio.png)
  │   ├── Table 4: Model hyperparameters
  │   ├── §2.4.1 Input representation
  │   ├── §2.4.2 1D CNN feature extractor
  │   ├── §2.4.3 Transformer encoder
  │   ├── §2.4.4 Transformer decoder with cross-attention
  │   └── §2.4.5 Training strategy
  │       ├── Pre-training on synthetic data
  │       ├── Fine-tuning on experimental data
  │       └── Domain adaptation (+ Figure 2: Domain Adaption.png)
  └── §2.5 Evaluation protocol (3 metrics, 4 baselines, cross-validation, inherent errors)

§3 Results and Discussion — COMMENTED OUT (menunggu data)
  ├── §3.1 Synthetic data characteristics
  ├── §3.2 Model performance on synthetic test set
  ├── §3.3 Fine-tuning on experimental spectra
  └── §3.4 Cross-attention interpretability

§4 Conclusions — COMMENTED OUT (menunggu data)

Bibliography: revv.bib
```

---

## 5. Konvensi Penulisan

### 5.1 Tense
- **§1 Introduction:** Present tense (menyatakan fakta umum dan state of the art)
- **§2 Methodology:** **Past tense** (seolah eksperimen sudah dilakukan — konvensi standar)
- **Persamaan matematika:** Present tense (kebenaran universal)
- **Abstract:** Campuran — "proposes" (present), "will be" (future), "are planned" (proposal)

### 5.2 Prinsip Sitasi (aturan dosen)
- **Published method** → HARUS di-cite ke paper asli
  - Saha equation → Fujimoto 2004
  - Boltzmann distribution → Fujimoto 2004
  - Attention mechanism → Vaswani 2017
  - Batch normalisation → Ioffe 2015
  - Adam optimiser → Kingma 2015
  - CNN/ResNet → He 2016
  - PLS → Wold 2001
  - GRL → Ganin 2016
  - Cosine annealing → Loshchilov 2017
  - NIST ASD → Kramida 2024
- **Novel method** (arsitektur encoder-decoder kita) → dideskripsikan detail, tidak perlu cite

### 5.3 Prinsip Paragraf
- 1 paragraf = 1 gagasan utama
- Setiap paragraf di §2 sudah diaudit dan memenuhi prinsip ini

### 5.4 Bahasa
- Passive voice dominan (standar scientific writing)
- Hindari klaim absolut seperti "the first", "no prior work" tanpa kualifikasi
- Gunakan "to the best of our knowledge" untuk klaim novelty

---

## 6. Etika & Peringatan

### 6.1 Status draft
- Ini adalah **proposal/draft** untuk review dosen, BUKAN submission final
- Abstract sudah ditulis dalam gaya proposal ("proposes", "will be", "are planned")
- §3 dan §4 masih kosong — **JANGAN submit** sebelum diisi

### 6.2 Hyperparameter placeholder
- Semua nilai di Table 4 adalah placeholder dari literatur
- HARUS diganti dengan nilai aktual setelah model ditraining

### 6.3 Klaim novelty
- "To the best of our knowledge" sudah digunakan (bukan klaim absolut)
- Perlu verifikasi literature review lebih mendalam sebelum submission

### 6.4 Data sharing antar paper
- Dataset eksperimental sama dengan paper CF-LIBS (belum submit)
- Acceptable karena kontribusi berbeda (CF-LIBS vs DL architecture)
- Jika paper CF-LIBS submit duluan, paper ini harus cite paper CF-LIBS

---

## 7. Referensi (revv.bib)

### 7.1 Referensi dari user (sudah diverifikasi)
Semua referensi selain 10 di bawah ini sudah diverifikasi oleh penulis.

### 7.2 Referensi ditambahkan oleh AI (semua paper real & terkenal)
| Key | Paper | Verifikasi |
|-----|-------|------------|
| `vaswani_attention_2017` | Attention Is All You Need | ✅ 600k+ citations |
| `he_deep_2016` | Deep Residual Learning | ✅ 200k+ citations |
| `zhou_informer_2021` | Informer: Beyond Efficient Transformer | ✅ AAAI 2021 Best Paper |
| `griem_spectral_1974` | Spectral Line Broadening by Plasmas | ✅ Classic textbook |
| `ioffe_batch_2015` | Batch Normalization | ✅ ICML 2015 |
| `kingma_adam_2015` | Adam: A Method for Stochastic Optimization | ✅ ICLR 2015 |
| `kramida_nist_2024` | NIST Atomic Spectra Database | ✅ NIST official |
| `wold_pls_2001` | PLS-regression: a basic tool | ✅ Standard PLS ref |
| `ganin_domain_2016` | Domain-Adversarial Training of Neural Networks | ✅ JMLR 2016 |
| `loshchilov_sgdr_2017` | SGDR: Stochastic Gradient Descent with Warm Restarts | ✅ ICLR 2017 |

### 7.3 BibTeX warnings (minor, tidak berpengaruh)
- `wang_simulation_2024`: empty journal field — cek apakah paper ini sudah published
- `zhou_informer_2021`: volume + number conflict — kosmetik saja

---

## 8. File & Gambar

### 8.1 Gambar aktif
| File | Digunakan sebagai | Width |
|------|-------------------|-------|
| `img/Arsitektur-PCST-P5.drawio.png` | Figure 1 (architecture) | 0.65\textwidth |
| `img/Domain Adaption.png` | Figure 2 (domain adaptation) | 0.55\textwidth |

### 8.2 Figure yang di-exclude
- **Framework diagram (TikZ)** — di-comment out karena layout rusak. Kode TikZ masih ada di file. Opsi: gambar ulang di draw.io lalu `\includegraphics`.

---

## 9. Langkah Selanjutnya (Prioritas)

1. **Gambar ulang Figure 1** (framework) di draw.io → ganti PNG
2. **Implementasi model** → training dengan data sintetis
3. **Isi hyperparameter** aktual → ganti placeholder di Table 4
4. **Fine-tune** pada data eksperimental
5. **Isi §3 Results** dengan metrik per-elemen (RMSE, R², MAPE)
6. **Isi §4 Conclusions**
7. **Update abstract** dari proposal-style ke results-style
8. **Proofreading final** sebelum submission
