# Struktur Repo Model-Agnostic untuk Fase Eksperimen

## Summary

Struktur yang disarankan adalah **cukup modular untuk ganti ide model dengan cepat**, tetapi **tidak terlalu formal** sehingga setiap perubahan hipotesis memaksa refactor lagi. Prinsip utamanya: stabilkan `data`, `metrics`, `runner`, dan `artifact layout`; biarkan keluarga model tetap fleksibel.

Default yang dipilih:
- repo tetap berfungsi sebagai **research lab**
- `SVR`, `PI`, `CNN`, dan `CNN Transformer` hidup berdampingan
- perubahan model tidak boleh memaksa perubahan format dataset atau lokasi hasil

## Key Changes

- Gunakan struktur tingkat atas berikut:
  - `src/` untuk kode reusable
  - `scripts/` untuk entrypoint eksperimen
  - `configs/` untuk konfigurasi eksperimen
  - `data/` untuk input dataset
  - `artifacts/` untuk output run
  - `Manuskrip/` dibiarkan terpisah dan tidak ikut arus eksperimen

- Di dalam `src/`, bagi berdasarkan fungsi yang relatif stabil:
  - `src/core/` untuk kontrak umum: dataset bundle, split, metrics, serialization, report writer
  - `src/data/` untuk loader HDF5, preprocessing, normalisasi, feature selection
  - `src/models/` untuk implementasi model per keluarga
  - `src/training/` untuk trainer dan evaluasi
  - `src/physics/` untuk `libs_physics` dan komponen fisika domain
  - `src/inference/` opsional untuk load model dan prediksi pada data baru

- Di dalam `src/models/`, pisahkan per keluarga model tanpa memaksa warisan yang rumit:
  - `src/models/svr.py`
  - `src/models/pi.py`
  - `src/models/cnn.py`
  - `src/models/cnn_transformer.py`
  - `src/models/registry.py`
  `registry.py` bertugas memetakan nama model seperti `svr`, `pi`, `cnn`, `cnn_transformer` ke class/factory yang sesuai.

- Semua eksperimen diluncurkan dari `scripts/` sebagai runner tipis:
  - `scripts/generate_dataset.py`
  - `scripts/train_model.py`
  - `scripts/evaluate_model.py`
  - `scripts/hpc/run-study-a.sh`
  `train_model.py` menjadi runner generik yang memilih model dari config/CLI, bukan skrip khusus per model.

- Konfigurasi dipisah agar ide baru tidak merusak config lama:
  - `configs/base.yaml`
  - `configs/data/study_a.yaml`
  - `configs/model/pi.yaml`
  - `configs/model/cnn_transformer.yaml`
  - `configs/experiment/study_a_plain.yaml`
  - `configs/experiment/study_a_cnn_transformer.yaml`
  Satu file experiment hanya menyatakan kombinasi data, model, training, dan output naming.

- Artefak hasil dipindahkan keluar dari `data/`:
  - `data/raw/` untuk data sumber
  - `data/processed/` untuk dataset HDF5 siap training
  - `artifacts/models/` untuk model tersimpan
  - `artifacts/reports/` untuk report training dan analisis
  - `artifacts/logs/` untuk log HPC dan log run
  - `artifacts/runs/<run_name>/` opsional jika ingin satu folder per eksperimen

## Public Interfaces / Contracts

- Stabilkan kontrak dataset:
  loader selalu mengembalikan objek dengan `spectra`, target termodinamika, target komposisi opsional, metadata kolom, dan split tidak ditentukan di model.

- Stabilkan kontrak model:
  setiap keluarga model harus menyediakan antarmuka minimal yang sama:
  - `fit(train_dataset)`
  - `predict(spectra)`
  - `save(path)`
  - `load(path)`
  - metadata model termasuk `model_type`, config, metrics, target names

- Stabilkan kontrak report:
  semua model menghasilkan report dengan format yang sama untuk:
  - metadata eksperimen
  - dataset code
  - model type
  - dimensi input
  - metrik per target
  - verdict
  Dengan begitu `SVR`, `PI`, dan `CNN Transformer` bisa dibandingkan langsung.

- Stabilkan kontrak runner:
  `scripts/train_model.py` menerima minimal:
  - `--dataset`
  - `--model`
  - `--config`
  - `--out`
  - `--epochs` atau override training lain bila perlu
  CLI lama boleh dipertahankan sementara sebagai compatibility wrapper yang memanggil runner baru.

## Implementation Notes

- Jangan refactor semua modul historis sekaligus.
  Tahap pertama cukup:
  - ekstrak util stabil ke `src/core`, `src/data`, `src/training`
  - pindahkan model existing ke `src/models/pi.py` dan nanti `src/models/svr.py`
  - buat registry model
  - buat runner training generik

- Jangan memaksakan inheritance base class yang kompleks.
  Cukup gunakan protokol sederhana atau pola factory agar model baru mudah ditambah.

- `mRMR` diperlakukan sebagai bagian dari pipeline data, bukan identitas model.
  Jadi nanti `plain`, `mrmr`, atau preprocessing lain hidup di `src/data/feature_selection.py` atau modul serupa, sehingga `PI` dan `CNN Transformer` bisa memakai atau mengabaikannya tanpa desain bercabang.

- `config.yaml` root yang sekarang dipecah secara bertahap.
  Untuk transisi, boleh ada compatibility loader yang masih membaca file lama lalu memetakan ke struktur config baru.

- HPC script tetap dipertahankan, tetapi hanya memanggil runner generik dengan kombinasi config yang eksplisit.
  Itu akan mengurangi ledakan jumlah skrip saat model bertambah.

## Test Plan

- Smoke test dataset generation tetap menghasilkan HDF5 dengan field yang sama seperti saat ini.
- Smoke test training generik berhasil untuk `plain PI` dengan dataset kecil.
- Smoke test registry bisa memilih model `pi` dan nanti `cnn_transformer` tanpa perubahan runner.
- Test report writer memastikan semua model mengeluarkan field report yang konsisten.
- Test serialization memastikan metadata `model_type`, metrics, dan target names tersimpan konsisten.
- Test backward compatibility minimal untuk membuka model lama atau setidaknya gagal dengan pesan yang jelas.

## Assumptions

- Fokus repo untuk beberapa bulan ke depan adalah eksperimen arsitektur, bukan packaging produksi.
- `SVR`, `PI`, dan model deep learning baru akan hidup paralel untuk perbandingan.
- Struktur manuskrip dan dokumen akademik tidak diubah pada tahap ini.
- Output eksperimen akan terus bertambah, jadi pemisahan `data/` dan `artifacts/` dianggap wajib.
- Desain dipilih agar implementer bisa menambah `CNN Transformer` tanpa mengubah loader dataset, metrik, atau format report.

