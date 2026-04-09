#!/bin/bash
#SBATCH --job-name=libs_generate
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/bwalidain/thesis/logs/generate_%j.out
#SBATCH --error=/home/bwalidain/thesis/logs/generate_%j.err

BASE_DIR=/home/bwalidain/thesis
CONDA_SH=/home/bwalidain/miniconda3/etc/profile.d/conda.sh
ENV_NAME=bw
SAMPLES=${SAMPLES:-200}
DATASET_GROUP=${DATASET_GROUP:-A}
SPECTRAL_TIER=${SPECTRAL_TIER:-L}
DATASET_TAG="${DATASET_GROUP}_${SPECTRAL_TIER}"
SCRATCH_OUT=${SCRATCH_OUT:-/home/bwalidain/_scratch/dataset_synthetic_${DATASET_TAG}.h5}
FINAL_OUT=${FINAL_OUT:-$BASE_DIR/data/dataset_synthetic_${DATASET_TAG}.h5}

mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/data"

source "$CONDA_SH"
conda activate "$ENV_NAME"

export LIBS_BASE_DIR="$BASE_DIR"
export MKL_VERBOSE=0
export OMP_NUM_THREADS=8
export PYTHONPATH="${PYTHONPATH:-}:$BASE_DIR"

echo "[Generate] Group=$DATASET_GROUP Tier=$SPECTRAL_TIER Samples=$SAMPLES"
echo "[Generate] Scratch=$SCRATCH_OUT"
echo "[Generate] Final=$FINAL_OUT"

python "$BASE_DIR/scripts/generate_dataset.py" \
  --samples "$SAMPLES" \
  --dataset-group "$DATASET_GROUP" \
  --spectral-tier "$SPECTRAL_TIER" \
  --out "$SCRATCH_OUT"
mv "$SCRATCH_OUT" "$FINAL_OUT"

conda deactivate
