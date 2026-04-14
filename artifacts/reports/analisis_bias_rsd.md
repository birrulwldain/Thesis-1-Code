# Analisis Deteksi Ekstrem Pembentuk Bias (RSD)

Laporan ini menguraikan daftar elemen yang menjadi biang keladi lonjakan *Relative Standard Deviation* (RSD) akibat eksistensi parsial.

Eksistensi parsial terjadi ketika pendeteksian di suatu sampel **tidak utuh (misal hanya muncul 1 atau 2 kali dari total 3 tembakan iterasi)**. Ketidak-utuhan ini memaksa algoritma memasukkan nilai `0` pada iterasi yang hilang, yang secara matematis melambungkan Standar Deviasi dan RSD menjadi di atas batas toleransi.

Dianalisis dari total **24 Kelompok Sampel (S1 - S24)**.

---

## 🟢 1. Kelompok Stabil: Spektrum Konsisten (Bebas Bias)
Elemen-elemen ini sangat presisi. Ketika mereka menampakkan diri di sebuah sampel, mereka konsisten muncul di ke-3 iterasi (*3/3*). Jika tidak ada, mereka mutlak hilang (*0/3*). Tidak pernah menyebabkan bias.

| Tipe Konsistensi | Daftar Elemen |
| :--- | :--- |
| **Sempurna Mutlak** | `Al`, `Ca`, `Cu`, `K`, `Li`, `Sr`, `Ti`, `Zr` (Hadir di 72 dari 72 spektrum tanpa celah) |
| **Sempurna Kondisional**| `Na`, `Ni`, `Re` (Cukup stabil. Jika sampel memilikinya, ia pasti muncul di ke-3 iterasi) |

---

## 🟡 2. Kelompok Bias Minor (Kasus Bolong di 1-2 Sampel)
Elemen-elemen ini rata-rata tepercaya, namun terkadang "berkedip" (terlewat deteksi) pada sebagian kecil sampel.

| Elemen | Jumlah Sampel Bias | Rincian Sampel (Status Kelengkapan) |
| :---: | :---: | :--- |
| **Ba** | 1 | S22 (hanya 2/3 iterasi) |
| **Br** | 1 | S15 (hanya 2/3 iterasi) |
| **Eu** | 1 | S14 (hanya 1/3 iterasi) |
| **In** | 1 | S21 (hanya 1/3 iterasi) |
| **Ir** | 1 | S6 (hanya 2/3 iterasi) |
| **Rb** | 1 | S23 (hanya 2/3 iterasi) |
| **Si** | 1 | S10 (hanya 1/3 iterasi) |
| **Ag** | 2 | S11 (2/3), S21 (2/3) |
| **Fe** | 2 | S12 (2/3), S14 (1/3) |
| **V** | 2 | S9 (2/3), S10 (1/3) |
| **Zn** | 2 | S20 (2/3), S21 (2/3) |

---

## 🔴 3. Kelompok Bias Moderat-Tinggi (Kasus Bolong di 3-6 Sampel)
Elemen-elemen pengacau ini seringkali terdeteksi secara fluktuatif di banyak sampel. Sangat disarankan untuk **mengosongkan nilai rata-ratanya** di sampel terkait (jangan pakai angka nol rata-rata), atau telusuri garis puncaknya di GUI untuk `Exclude Line`.

| Elemen | Jumlah Sampel Bias | Rincian Sampel (Status Kelengkapan) |
| :---: | :---: | :--- |
| **Mn** | **6** | S6 (2/3), S7 (2/3), S10 (2/3), S18 (2/3), S19 (2/3), S21 (2/3) |
| **Cr** | 4 | S3 (2/3), S4 (1/3), S18 (2/3), S23 (2/3) |
| **Ga** | 4 | S13 (2/3), S19 (2/3), S23 (1/3), S24 (2/3) |
| **Pb** | 4 | S1 (2/3), S2 (2/3), S19 (2/3), S21 (2/3) |
| **Y**  | 4 | S12 (1/3), S14 (2/3), S16 (2/3), S17 (2/3) |
| **As** | 3 | S17 (1/3), S19 (2/3), S21 (2/3) |
| **Cl** | 3 | S5 (2/3), S7 (2/3), S23 (2/3) |
| **Mg** | 3 | S11 (2/3), S12 (2/3), S22 (2/3) |
| **P**  | 3 | S2 (2/3), S6 (2/3), S22 (2/3) |

---

> [!TIP]
> **Rekomendasi Penanganan untuk Tesis:**
> Untuk mendapatkan korelasi model (e.g., PCA atau Regresi) yang bagus:
> 1. Abaikan saja iterasi parsial dari kelompok **Merah** di sampel spesifik tersebut. Anda bisa memanfaatkan format *Scientific Notation* di *Detail Garis* GUI untuk me-*reject* garis jika dirasa hanya mem-fit *background noise*.
> 2. Elemen inti di kelompok **Hijau** adalah ujung tombak riset Anda, memiliki kredibilitas spektral tertinggi.
