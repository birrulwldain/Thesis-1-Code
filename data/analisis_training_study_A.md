# Analisis Hasil Training Study A

Sumber laporan:

- `model_inversi_plain_pi_A_L_report.txt`
- `model_inversi_plain_pi_A_M_report.txt`
- `model_inversi_plain_pi_A_H_report.txt`
- `model_inversi_mrmr_miq_pi_A_L_report.txt`
- `model_inversi_mrmr_miq_pi_A_M_report.txt`
- `model_inversi_mrmr_miq_pi_A_H_report.txt`

## Ringkasan metrik

| Dataset | Pipeline | Dimensi input | RMSE T_e (%) | RMSE n_e (%) | RMSE komposisi (%) |
|---|---|---:|---:|---:|---:|
| A_L | Plain PI | 4000 | 13.16 | 70.24 | 2.37 |
| A_L | mRMR(MIQ) + PI | 126 | 10.90 | 70.29 | 2.46 |
| A_M | Plain PI | 8000 | 9.71 | 62.60 | 2.36 |
| A_M | mRMR(MIQ) + PI | 251 | 10.30 | 62.69 | 2.33 |
| A_H | Plain PI | 16000 | 13.89 | 61.36 | 2.63 |
| A_H | mRMR(MIQ) + PI | 256 | 10.24 | 61.29 | 2.52 |

## Temuan utama

1. Target komposisi sudah stabil pada semua eksperimen.
   Nilai RMSE komposisi rata-rata selalu berada di kisaran 2.33% hingga 2.63%, dan seluruh laporan memberi verdict `baik`.

2. Target `n_e_core` adalah bottleneck utama.
   Semua eksperimen berada pada RMSE relatif sekitar 61% hingga 70%, sehingga seluruh verdict `n_e` tetap `lemah` baik dengan maupun tanpa mRMR.

3. mRMR memberi pengurangan dimensi yang sangat agresif tanpa menghancurkan performa.
   - A_L: 4000 -> 126 fitur
   - A_M: 8000 -> 251 fitur
   - A_H: 16000 -> 256 fitur
   Meskipun dimensi turun drastis, RMSE komposisi dan `T_e` tetap sebanding dengan baseline plain.

4. Dampak mRMR terhadap `T_e_core` bergantung pada tier spektral.
   - A_L: membaik dari 13.16% ke 10.90%
   - A_M: sedikit memburuk dari 9.71% ke 10.30%
   - A_H: membaik dari 13.89% ke 10.24%
   Ini menunjukkan seleksi fitur paling membantu pada tier rendah dan tinggi, tetapi tidak memberi keuntungan pada tier menengah.

5. Tier `A_M` adalah kompromi terbaik secara umum.
   - Plain PI pada A_M memberi RMSE `T_e` terbaik: 9.71%
   - mRMR pada A_M memberi RMSE komposisi terbaik: 2.33%
   - RMSE `n_e` pada A_M juga lebih baik daripada A_L, walau tetap belum memadai

## Interpretasi untuk proyek

- Untuk estimasi komposisi, pipeline saat ini sudah cukup meyakinkan.
- Untuk estimasi `T_e_core`, model sudah usable, terutama pada `A_M plain` dan `A_H/A_L` dengan mRMR.
- Untuk estimasi `n_e_core`, masalah tampaknya bukan sekadar jumlah fitur. Karena plain dan mRMR memberi hasil hampir identik, akar masalah lebih mungkin berasal dari:
  - informasi `n_e` yang memang lemah dalam spektrum sintetis yang digunakan,
  - arsitektur model Phase 1 yang lebih mudah mempelajari `T_e` dan komposisi dibanding `n_e`,
  - jumlah sampel yang masih kecil (`200` dengan split train/test `160/40`),
  - training yang masih singkat (`10` epoch).

## Implikasi praktis

- Jika prioritasnya efisiensi input, `mRMR(MIQ)+PI` layak dipertahankan karena kompresi fitur sangat besar dengan degradasi performa yang kecil.
- Jika prioritasnya akurasi `T_e`, baseline terbaik saat ini adalah `A_M plain`.
- Jika prioritasnya model ringkas dengan performa tetap kompetitif, kandidat terbaik adalah `A_H mRMR` atau `A_M mRMR`.
- Jika prioritasnya `n_e_core`, eksperimen berikutnya sebaiknya fokus pada desain data dan objective training, bukan hanya pada seleksi fitur.

## Rekomendasi eksperimen lanjutan

1. Naikkan jumlah sampel di atas 200 untuk melihat apakah `n_e_core` memang data-limited.
2. Naikkan epoch di atas 10 dan pantau kurva loss/validasi agar jelas apakah model masih underfit.
3. Simpan metrik per-target yang lebih rinci untuk komponen komposisi dan distribusi error, bukan hanya rata-rata.
4. Uji loss weighting khusus agar error `n_e_core` tidak kalah dominan dari target lain.
5. Bandingkan `mrmr_score_mode=miq` dengan `mifs` pada dataset yang sama.
6. Validasi apakah tier spektral `A_M` memang sweet spot tetap ketika jumlah sampel dan epoch ditingkatkan.
