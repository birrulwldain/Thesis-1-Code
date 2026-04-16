# Isian Substansial Proposal (Skema Penelitian Tesis Magister — PTM)

*(Silakan salin dan tempel bagian-bagian di bawah ini ke dalam dokumen Word PTM Anda sesuai urutan)*

---

## Nama Ketua: Nasrullah Idris | NIDN: 0003077609

---

## JUDUL

CNN–Transformer Encoder–Decoder untuk Analisis Kuantitatif LIBS: Dekomposisi Spektral Mono–Poliatomik Eksplisit pada Karakterisasi Tanah Vulkanik

---

## RINGKASAN

Laser-Induced Breakdown Spectroscopy (LIBS) merupakan teknik spektroskopi emisi yang memungkinkan analisis multi-elemen secara cepat tanpa memerlukan preparasi sampel yang kompleks. Meskipun pendekatan Calibration-Free LIBS (CF-LIBS) memungkinkan kuantifikasi unsur tanpa standar rujukan eksternal, akurasinya sangat sensitif terhadap penyimpangan asumsi Local Thermodynamic Equilibrium (LTE), terutama efek matriks berupa penyerapan mandiri (self-absorption) dan gradien spasial plasma saat diaplikasikan pada material heterogen seperti tanah vulkanik. Di sisi lain, pendekatan machine learning yang ada memperlakukan spektrum LIBS sebagai single composite signal, mengabaikan fakta fisik fundamental bahwa spektrum poliatomik sejatinya merupakan superposisi terbobot konsentrasi dari spektrum monoatomik penyusunnya. Belum ada riset yang secara eksplisit memodelkan dekomposisi mono→poliatomik dalam kerangka pembelajaran.

Penelitian ini mengusulkan arsitektur CNN–Transformer Encoder–Decoder yang secara struktural memodelkan hubungan dekomposisi spektral. Encoder memproses spektrum monoatomik individual yang dibangkitkan oleh Saha–Boltzmann forward model untuk mempelajari representasi laten per-elemen melalui self-attention. Decoder menerima spektrum campuran poliatomik dan memprediksi konsentrasi elemen melalui mekanisme cross-attention terhadap output encoder, sehingga mempelajari aturan komposisi spektral secara end-to-end. Lapisan 1D-CNN mengekstraksi fitur profil emisi lokal sebelum Transformer menangkap dependensi panjang gelombang global.

Model akan di-pre-train pada spektrum sintetis yang dibangkitkan dari transisi NIST Atomic Spectra Database dan di-fine-tune pada pengukuran LIBS eksperimental pada sampel tanah vulkanik dari Aceh, Indonesia, dengan data X-ray Fluorescence (XRF) sebagai ground-truth konsentrasi. Perbandingan baseline dilakukan terhadap model CNN-only, Transformer-only, Partial Least Squares, dan Informer. Kontribusi penelitian ini meliputi: (i) framework encoder–decoder yang memisahkan representasi spektral mono/poliatomik secara struktural, (ii) mekanisme cross-attention untuk regresi konsentrasi yang physically interpretable, dan (iii) aplikasi pada matriks tanah vulkanik kompleks Aceh, Indonesia.

---

## KATA KUNCI

LIBS; deep learning; CNN–Transformer; encoder–decoder; cross-attention; dekomposisi spektral; tanah vulkanik

---

## PENDAHULUAN

### Latar Belakang

Laser-Induced Breakdown Spectroscopy (LIBS) merupakan teknik spektroskopi emisi yang telah banyak digunakan untuk analisis komposisi unsur pada berbagai material kompleks, termasuk mineral, batuan, dan material tanah (Legnaioli et al., 2025). Kemampuan LIBS untuk melakukan analisis multi-elemen secara cepat tanpa memerlukan preparasi sampel yang kompleks menjadikannya metode yang menarik untuk aplikasi geokimia dan eksplorasi sumber daya alam (Sawyers et al., 2025). Berbagai penelitian menunjukkan bahwa LIBS telah berhasil digunakan untuk kuantifikasi unsur pada sampel geologi melalui beragam pendekatan analitik, mulai dari regresi multivariat, kurva kalibrasi univariat berbasis garis spektral tertentu, hingga metode berbasis machine learning (Babos et al., 2024; Manzoor et al., 2025). Pendekatan-pendekatan tersebut mampu mencapai akurasi analitik yang tinggi, namun keberhasilan kuantifikasi sangat bergantung pada strategi pemodelan spektrum serta sensitivitas metode terhadap efek matriks pada sampel geologi yang kompleks (Hao et al., 2024). Kondisi ini menjadi semakin relevan pada analisis tanah vulkanik, yang secara inheren memiliki komposisi multi-elemen dengan variasi matriks yang tinggi.

Dalam pendekatan kuantitatif LIBS, salah satu metode yang banyak digunakan adalah Calibration-Free LIBS (CF-LIBS), yang memungkinkan estimasi komposisi unsur tanpa memerlukan sampel standar kalibrasi. Metode ini bergantung pada ekstraksi parameter plasma fundamental seperti temperatur elektron (T_e) dan densitas elektron (n_e), yang dihitung berdasarkan asumsi Local Thermodynamic Equilibrium (LTE) (Cristoforetti et al., 2010). Namun, asumsi plasma homogen dalam kondisi LTE sering kali tidak memadai untuk mendeskripsikan plasma LIBS yang bersifat sangat transien dan bergradien tinggi (Zaitsev et al., 2024; Bultel et al., 2025). Penyimpangan dari kondisi LTE ini dapat memicu distorsi spektral akibat efek opasitas optik dan penyerapan mandiri (self-absorption), yang pada akhirnya mengganggu stabilitas estimasi temperatur dan densitas plasma (Tang et al., 2024). Pendekatan berbasis model fisika plasma seperti model radiative transfer adaptif MERLIN (Favre et al., 2025) yang memanfaatkan distribusi Saha–Boltzmann serta Persamaan Transfer Radiatif menunjukkan bahwa rekonstruksi spektrum LIBS yang kompleks dapat dicapai dengan mempertahankan konsistensi termodinamika plasma, meskipun implementasinya masih menghadapi hambatan komputasi yang signifikan akibat kebutuhan inversi numerik nonlinier (Favre et al., 2025b).

Perkembangan metode machine learning dan deep learning telah menawarkan alternatif untuk mempercepat analisis spektrum LIBS dengan mempelajari hubungan nonlinier antara spektrum dan parameter plasma secara langsung dari data. Convolutional Neural Networks (CNN) mampu menangkap fitur lokal seperti profil garis emisi, sementara arsitektur Transformer memodelkan dependensi jarak jauh sepanjang sumbu panjang gelombang (Liu et al., 2026). Model hibrida CNN–Transformer telah menunjukkan hasil menjanjikan untuk kuantifikasi unsur jejak pada baja (Liu et al., 2026) dan untuk klasifikasi signatur spektral pada sampel multi-elemen (Walidain et al., 2026). Namun, sebagian besar model data-driven beroperasi sebagai black box yang tidak secara eksplisit mempertimbangkan hukum fisika plasma yang mendasari pembentukan spektrum emisi (Hao et al., 2024). Fakta fisik fundamental yang belum dieksploitasi adalah: spektrum LIBS poliatomik merupakan superposisi terbobot konsentrasi dari spektrum monoatomik penyusunnya. Seluruh model eksisting—baik berbasis fisika maupun data-driven—memperlakukan spektrum sebagai single signal dan mengabaikan struktur komposisional ini. **Belum ada penelitian sebelumnya yang secara eksplisit memodelkan dekomposisi spektrum campuran ke dalam konstituennya dalam kerangka pembelajaran.** Cross-attention, mekanisme yang awalnya diusulkan untuk translasi sekuens-ke-sekuens (Vaswani et al., 2017), sangat ideal untuk tugas ini: decoder dapat meng-query representasi elemen dari encoder untuk mengetahui kontribusi monoatomik mana yang membentuk observasi poliatomik tertentu (Liu et al., 2026; Walidain et al., 2026).

**Penelitian ini mengusulkan arsitektur CNN–Transformer encoder–decoder yang memperkenalkan dekomposisi spektral mono–poliatomik eksplisit ke dalam analisis kuantitatif LIBS.** Encoder memproses spektrum monoatomik individual yang dibangkitkan oleh Saha–Boltzmann forward model untuk mempelajari representasi laten spesifik-elemen. Decoder menerima spektrum campuran poliatomik dan memprediksi konsentrasi elemen melalui cross-attention terhadap output encoder. Lapisan 1D-CNN mengekstraksi fitur emisi lokal sebelum Transformer menangkap dependensi panjang gelombang global. Model di-pre-train pada spektrum sintetis dari NIST Atomic Spectra Database dan di-fine-tune pada pengukuran LIBS eksperimental tanah vulkanik dari Aceh, Indonesia, dengan analisis XRF menyediakan ground-truth konsentrasi.

### Rumusan Masalah

1. Bagaimana memodelkan hubungan antara spektrum poliatomik yang dihasilkan dari plasma LIBS dengan spektrum monoatomik penyusun yang merepresentasikan karakteristik unsur individual?
2. Bagaimana merancang arsitektur pembelajaran mendalam yang mampu mengekstraksi fitur spektral lokal serta menangkap dependensi global antar panjang gelombang pada spektrum LIBS?
3. Bagaimana memanfaatkan arsitektur CNN–Transformer encoder–decoder untuk mempelajari hubungan struktural antara spektrum monoatomik dan spektrum poliatomik dalam analisis spektrum LIBS?
4. Bagaimana kinerja model yang diusulkan dalam memprediksi konsentrasi unsur serta parameter plasma dari spektrum LIBS pada sampel tanah vulkanik?

### Tujuan Khusus

1. Memodelkan komposisi spektral melalui arsitektur encoder–decoder yang secara eksplisit memodelkan hubungan antara spektrum monoatomik (representasi per-elemen) dan spektrum poliatomik (campuran terukur).
2. Mengembangkan arsitektur hibrida CNN–Transformer yang menggabungkan ekstraksi fitur lokal (CNN 1D untuk profil puncak emisi) dan pemodelan dependensi global (Transformer untuk korelasi antar panjang gelombang) dalam paradigma encoder–decoder.
3. Memprediksi konsentrasi unsur secara akurat pada sampel tanah vulkanik dari Aceh dan mengevaluasi kinerja model terhadap baseline (PLS, CNN-only, Transformer-only, Informer).

### Kebaruan (Novelty)

1. **Pemodelan spektral mono–poliatomik**: Pendekatan pertama yang secara eksplisit memodelkan hubungan komposisi antara spektrum monoatom dan poliatom LIBS melalui mekanisme cross-attention.
2. **Kerangka CNN–Transformer untuk analisis LIBS**: Arsitektur hibrida yang menggabungkan ekstraksi fitur lokal (CNN) dan pemodelan dependensi global (Transformer) dalam paradigma encoder–decoder.
3. **Karakterisasi tanah vulkanik**: Prediksi kuantitatif konsentrasi unsur pada matriks geologi kompleks, relevan untuk vulkanologi dan penilaian tanah pertanian.

---

## PENELITIAN TERDAHULU

### A. Fondasi Riset Grup — LIBS pada Material Geologi Aceh & Indonesia

Penelitian analisis geokimia material geologi dari wilayah Aceh dan Indonesia menggunakan teknik LIBS telah menjadi fokus utama grup riset. Mitaphonna et al. (2023) melakukan identifikasi deposit tsunami 2004 di Desa Pulot, Kabupaten Aceh Besar menggunakan analisis geokimia, membuktikan kemampuan teknik spektroskopi dalam mengkarakterisasi signatur geokimia tanah terdampak bencana alam. Dalam kelanjutannya, Mitaphonna et al. (2024) menerapkan teknik LIBS secara langsung untuk analisis geokimia kualitatif deposit tsunami di Seungko Mulat, Aceh Besar, berhasil mengidentifikasi keberadaan elemen-elemen mayor seperti Si, Al, Fe, Ca, Mg, Na, dan K pada sampel tanah. Studi ini menunjukkan bahwa LIBS mampu mendeteksi elemen-elemen target yang relevan, namun analisis yang dilakukan **masih bersifat kualitatif** — menjustifikasi kebutuhan pendekatan kuantitatif yang lebih akurat.

Pada skala nasional, Khumaeni, Idris et al. (2025) mendemonstrasikan penerapan CF-LIBS untuk analisis komposisi geokimia dan mineral tanah vulkanik yang terdampak erupsi Gunung Merapi di Jawa Tengah. Studi ini membuktikan bahwa CF-LIBS **dapat** diterapkan pada tanah vulkanik Indonesia, namun sekaligus mengekspos keterbatasan inherennya: akurasi bergantung pada validitas asumsi LTE yang sering dilanggar pada matriks tanah vulkanik yang sangat heterogen. Keterbatasan ini menjadi motivasi langsung untuk pengembangan pendekatan berbasis deep learning yang diusulkan dalam tesis ini.

### B. Lanskap Riset LIBS + Artificial Intelligence

Perkembangan penelitian LIBS telah berevolusi dari simulasi fisika murni menuju integrasi machine learning dan deep learning:

**Favre et al. (2025)** mengembangkan pendekatan ML CF-LIBS kuantitatif yang memanfaatkan ekstraksi fitur CNN dari database simulasi berskala besar. Pendekatan ini terbukti ampuh untuk diagnostik simultan tanpa rekalibrasi, namun performa baseline-nya **kolaps pada percampuran puncak dense overlap** yang umum dijumpai pada spektrum tanah vulkanik. Hal ini menunjukkan kebutuhan akan mekanisme kompensasi yang lebih canggih.

**Wang et al. (2024)** mengusulkan hibridisasi pra-filter wavelet dengan arsitektur Transformer dan CNN untuk analisis spectral berbantuan simulasi. Pendekatan ini mantap memadukan ikatan global dependency dan deteksi pendaran lokal, namun **celah overlap fisik garis emisi tetap diabaikan** dan diperlakukan sebagai single signal statis.

**Liu et al. (2026)** memperkenalkan hibridisasi CNN–Transformer untuk LIBS jarak jauh (remote LIBS) pada analisis unsur baja. Arsitektur ini berhasil menggabungkan deteksi pola lokal (CNN) dan dependensi global (Transformer), namun spektrum tetap diperlakukan sebagai **single composite signal** tanpa mempertimbangkan struktur komposisi spektral mono–poliatomik.

**Walidain et al. (2026)** mengadaptasi arsitektur Informer (ProbSparse attention) untuk klasifikasi spektral LIBS pada sampel geologi. Studi ini membuktikan bahwa mekanisme attention yang efisien mampu memproses spektrum LIBS bersekuens ultra-panjang, namun **hanya mencapai klasifikasi kualitatif**, bukan regresi kuantitatif konsentrasi.

### Kesimpulan Research Gap

**Observasi kunci**: Semua pendekatan di atas memperlakukan spektrum sebagai single signal. Struktur komposisi spektrum LIBS — yaitu bahwa spektrum poliatomik merupakan superposisi dari spektrum monoatomik penyusun — belum pernah dieksplorasi secara eksplisit. Tidak ada penelitian sebelumnya yang: (i) memisahkan spektrum monoatom vs poliatom, (ii) menggunakan mekanisme attention lintas komposisi, maupun (iii) menerapkan kerangka encoder–decoder untuk LIBS. Oleh karena itu, penelitian ini mengusulkan pendekatan di mana encoder mempelajari representasi monoatom, decoder merekonstruksi komposisi poliatom via cross-attention, dan prediksi konsentrasi unsur dilakukan secara end-to-end.

---

## PETA JALAN (ROADMAP) PENELITIAN

*(Sisipkan gambar diagram Peta Jalan di bawah ini)*

![Peta Jalan Penelitian](/Users/birrulwldain/.gemini/antigravity/brain/b0fc3393-c0bd-4fa7-8f99-4f18a45ee98f/roadmap_diagram_1776333497781.png)

**Gambar 1.** Peta jalan penelitian menunjukkan progresi dari fondasi riset grup (analisis geokimia LIBS di Aceh & tanah vulkanik Indonesia) menuju riset AI spektral LIBS mutakhir, hingga kontribusi tesis ini berupa dekomposisi mono–poliatomik eksplisit via CNN–Transformer Encoder–Decoder.

---

## PROFIL MAHASISWA MAGISTER

| Item | Detail |
|------|--------|
| Nama Mahasiswa | Birrul Walidain |
| NIM | 250820201100015 |
| Program Studi | Magister Fisika, FMIPA Universitas Syiah Kuala |
| Judul Proposal | CNN–Transformer Encoder–Decoder untuk Analisis Kuantitatif LIBS: Dekomposisi Spektral Mono–Poliatomik Eksplisit pada Karakterisasi Tanah Vulkanik |

**Perkembangan Tesis:**
Mahasiswa telah menyelesaikan desain arsitektur CNN–Transformer encoder–decoder dan implementasi modul pembangkitan data sintetis berbasis forward model Saha–Boltzmann dengan data transisi dari NIST Atomic Spectra Database. Tahap saat ini meliputi pre-training model pada dataset sintetis (10.000 spektrum, 7 elemen mayor) dan preparasi sampel tanah vulkanik dari lereng Gunung Seulawah Agam, Aceh. Sebuah manuskrip jurnal sedang disusun untuk publikasi di IOP Machine Learning: Science and Technology. Studi pendahulu berupa klasifikasi spektral LIBS menggunakan arsitektur Informer telah dipublikasikan (Walidain et al., 2026).

---

## METODE PENELITIAN

### Tempat dan Waktu Penelitian
Penelitian dilaksanakan di:
- **Laboratorium Optika dan Aplikasi Laser, FMIPA USK** — akuisisi data LIBS eksperimental.
- **Laboratorium Material, FMIPA** — preparasi sampel pellet tanah vulkanik.
- **Workstation komputasi (GPU Server)** — training dan inferensi model CNN–Transformer.

Rentang waktu penelitian: **10 bulan** (Februari 2025 – November 2025).

### Alat dan Bahan Penelitian

| No | Alat/Bahan | Spesifikasi | Jumlah |
|----|-----------|-------------|--------|
| 1 | Laser Nd:YAG | Q-switched, λ = 1064 nm, pulse energy 114 mJ | 1 unit |
| 2 | Lensa bikonveks | f = 155 mm (pemfokusan laser) | 1 unit |
| 3 | Spektrometer Echelle + detektor OMA | Rentang spektral 200–900 nm | 1 unit |
| 4 | Kabel serat optik | Penghantar sinyal emisi plasma | 1 unit |
| 5 | Mesin Hydraulic Press | Tekanan hingga 7 ton | 1 unit |
| 6 | Mortar dan pestel | Penggerusan sampel tanah | 1 set |
| 7 | Ayakan 40 mesh (420 µm) | Penyeragaman ukuran partikel | 1 unit |
| 8 | Cetakan sampel | Pembuatan pelet (ketebalan ~4 mm) | 1 unit |
| 9 | Masker pelindung | Keselamatan kerja | Secukupnya |
| 10 | GPU Compute Server | Pelatihan deep learning (CNN–Transformer) | 1 unit |
| 11 | Instrumen XRF | Ground-truth konsentrasi elemen | 1 unit |
| 12 | Sampel tanah vulkanik | Gunung Seulawah Agam, Aceh (4 arah × 3 kedalaman) | **24 sampel** |

### Prosedur Penelitian

Prosedur mengikuti two-stage workflow:

#### Tahap 1 — Offline: Pembangkitan Data Sintetis (Pre-training)

1. **Pengumpulan data transisi atomik** dari NIST Atomic Spectra Database: panjang gelombang transisi (λ_ki), koefisien Einstein (A_ki), degenerasi (g_i), dan energi level (E_i) untuk 7 elemen mayor: **Si, Al, Fe, Ca, Mg, Na, K**.
2. **Penghitungan populasi** tiap tingkat ionisasi menggunakan Persamaan Saha dan distribusi keadaan tereksitasi menggunakan Distribusi Boltzmann. Parameter plasma di-sampling secara acak seragam: T_e ∈ [6.000–15.000] K, n_e ∈ [10^16 – 10^17] cm^{-3}.
3. **Pembangkitan spektrum monoatomik** S_mono^(z)(λ) per elemen dengan menjumlahkan koefisien emisi spektral menggunakan profil Voigt (konvolusi pelebaran Doppler + Stark).
4. **Konstruksi spektrum poliatomik** S_poly(λ) = Σ_z c_z · S_mono^(z)(λ) — superposisi terbobot konsentrasi fraksional.
5. **Konvolusi instrumental** dengan fungsi Gaussian (FWHM = 0,02 nm) untuk mencocokkan resolusi spektrometer echelle.
6. Total dataset: **10.000 pasangan spektrum sintetis**, dibagi 80% training / 20% test.

#### Tahap 2 — Online: Akuisisi Eksperimental dan Training Model

**2a. Preparasi dan Akuisisi Sampel**
1. Pengumpulan sampel tanah vulkanik dari lereng **Gunung Seulawah Agam, Aceh** — 4 arah mata angin (Utara, Barat, Selatan, Timur) × 3 kedalaman (0–20, 20–40, 40–60 cm) = **24 sampel**.
2. Preparasi sampel: penggerusan menggunakan mortar → pengayakan 40 mesh → pengepresan hidrolik 7 ton → pembentukan pelet (∅ ~4 mm).
3. Ablasi dengan laser Nd:YAG (1064 nm, 114 mJ). 3 tembakan laser per sampel, dirata-ratakan untuk meningkatkan rasio sinyal-terhadap-noise (SNR).
4. Perekaman spektrum emisi plasma dengan spektrometer Echelle + detektor OMA.
5. Pengukuran XRF independen sebagai **ground-truth konsentrasi unsur** untuk supervisi fine-tuning.

**2b. Arsitektur Model CNN–Transformer Encoder–Decoder**
1. **1D-CNN Feature Extractor** (shared encoder–decoder): 3 lapisan konvolusi 1D (kernel: 7, 5, 3; filter: 64, 128, 128) + Batch Normalization + ReLU. Menangkap profil puncak emisi lokal.
2. **Transformer Encoder** (4 layer, 8 attention heads): menerima fitur CNN dari **seluruh** spektrum monoatomik (concatenated) + positional & element-type embedding → menghasilkan representasi laten spesifik-elemen via self-attention.
3. **Transformer Decoder** (4 layer, 8 heads): menerima fitur CNN spektrum campuran poliatomik → **masked self-attention** → **cross-attention** terhadap output encoder → mempelajari aturan komposisi spektral.
4. **Regression Head**: Global Average Pooling → Linear projection → vektor konsentrasi ĉ ∈ ℝ^Z.
5. **Tiga output head**: (i) konsentrasi elemen (aktivasi Softplus), (ii) rekonstruksi spektrum (Sigmoid), (iii) parameter plasma T_e, n_e (Softplus).

**2c. Strategi Pelatihan**
1. **Pre-training** pada data sintetis: 200 epoch, learning rate 10^{-4}, optimizer Adam + cosine annealing. Loss = MSE konsentrasi + α × MSE rekonstruksi spektral (α = 0,1).
2. **Fine-tuning** pada data eksperimental LIBS: 50 epoch, learning rate 10^{-5}, dengan target ground-truth XRF.
3. **Domain adaptation**: Gradient Reversal Layer (GRL) + domain classifier (sintetis vs eksperimental) → mendorong encoder menghasilkan representasi domain-invariant.

**2d. Protokol Evaluasi**
- **3 metrik per elemen**: RMSE, R², MAPE.
- **4 model baseline**: PLS (Partial Least Squares), CNN-only, Transformer-only, Informer (Walidain et al., 2026).
- **Validasi**: stratified 5-fold cross-validation pada dataset eksperimental.
- **Interpretabilitas**: analisis bobot cross-attention → identifikasi apakah model menemukan region spektral spesifik elemen yang konsisten dengan garis emisi NIST yang diketahui.

---

## JADWAL PENELITIAN

| Fase | Bulan | Kegiatan Utama |
|------|-------|---------------|
| 1 — Persiapan Data | 1–3 | Pembangkitan spektrum sintetis (Saha–Boltzmann + NIST ASD), pengumpulan & preparasi sampel tanah vulkanik, pengukuran XRF |
| 2 — Pengembangan Model | 3–5 | Desain & implementasi arsitektur CNN–Transformer encoder–decoder |
| 3 — Pelatihan & Optimasi | 5–8 | Pre-training pada data sintetis, fine-tuning pada data eksperimental, optimasi hyperparameter, domain adaptation |
| 4 — Evaluasi & Penulisan | 8–10 | Analisis performa, perbandingan baseline, analisis interpretabilitas cross-attention, penulisan tesis & manuskrip jurnal |

---

## DAFTAR PUSTAKA UTAMA

1. Mitaphonna, R., Ramli, M., Ismail, N., Hartadi, B.S., & Idris, N. (2023). Identification of possible preserved 2004 Indian Ocean tsunami deposits collected from Pulot Village in Aceh Besar Regency, Indonesia. *J. Phys.: Conf. Ser.*, 2582(1), 012033.
2. Mitaphonna, R., Ramli, M., Ismail, N., & Idris, N. (2024). Qualitative Geochemical Analysis of the 2004 Indian Ocean Giant Tsunami Deposits Excavated at Seungko Mulat Located in Aceh Besar of Indonesia Using Laser-Induced Breakdown Spectroscopy. *Indonesian Journal of Chemistry*, 24(3).
3. Khumaeni, A., Indriana, R.D., Jonathan, F., Fiantis, D., Ginting, F.I., Idris, N., & Kurniawan, H. (2025). Analysis of geochemical and mineral compositions of volcanic soil affected by Merapi eruption in Central Java Indonesia using laser-induced breakdown spectroscopy with calibration-free. *Talanta*, 295, 128376.
4. Walidain, B. et al. (2026). Informer-based classification of LIBS spectra. *(Published)*.
5. Liu, Y. et al. (2026). Remote LIBS quantitative analysis using hybrid CNN–Transformer. *(Published)*.
6. Favre, Y. et al. (2025). MERLIN: Adaptive radiative-transfer model for CF-LIBS. *(Published)*.
7. Wang, S. et al. (2024). Simulation-assisted deep learning for LIBS spectral analysis. *(Published)*.
