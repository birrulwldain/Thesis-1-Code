#!/bin/bash
#SBATCH --job-name=libs_train
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/bwalidain/thesis/logs/train_%j.out
#SBATCH --error=/home/bwalidain/thesis/logs/train_%j.err

BASE_DIR=/home/bwalidain/thesis
CONDA_SH=/home/bwalidain/miniconda3/etc/profile.d/conda.sh
ENV_NAME=bw
DATASET_GROUP=${DATASET_GROUP:-A}
SPECTRAL_TIER=${SPECTRAL_TIER:-L}
DATASET_TAG="${DATASET_GROUP}_${SPECTRAL_TIER}"
DATASET_PATH=${DATASET_PATH:-$BASE_DIR/data/dataset_synthetic_${DATASET_TAG}.h5}
MODEL_OUT=${MODEL_OUT:-$BASE_DIR/data/model_inversi_pi_${DATASET_TAG}.pkl}
EPOCHS=${EPOCHS:-3}
USE_MRMR=${USE_MRMR:-1}

mkdir -p "$BASE_DIR/logs"
mkdir -p "$BASE_DIR/data"

source "$CONDA_SH"
conda activate "$ENV_NAME"

export LIBS_BASE_DIR="$BASE_DIR"
export MKL_VERBOSE=0
export OMP_NUM_THREADS=8
export PYTHONPATH="${PYTHONPATH:-}:$BASE_DIR"
export PYTHONUNBUFFERED=1

echo "[Train] Group=$DATASET_GROUP Tier=$SPECTRAL_TIER"
echo "[Train] Dataset=$DATASET_PATH"
echo "[Train] Model=$MODEL_OUT"

TRAIN_ARGS=(
  "$BASE_DIR/scripts/train_inversion_model.py"
  --dataset "$DATASET_PATH"
  --out "$MODEL_OUT"
  --epochs "$EPOCHS"
)

if [ "$USE_MRMR" = "1" ]; then
  TRAIN_ARGS+=(
    --mrmr
    --mrmr-features 256
    --mrmr-pool 1024
    --mrmr-sample 500
    --mrmr-prefilter-stride 32
    --mrmr-score-mode miq
  )
fi

python -u "${TRAIN_ARGS[@]}"

conda deactivate
