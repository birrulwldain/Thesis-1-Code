#!/bin/bash
#SBATCH --job-name=libs_study_A
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/bwalidain/thesis/artifacts/logs/study_A_%j.out
#SBATCH --error=/home/bwalidain/thesis/artifacts/logs/study_A_%j.err

BASE_DIR=/home/bwalidain/thesis
CONDA_SH=/home/bwalidain/miniconda3/etc/profile.d/conda.sh
ENV_NAME=bw
DATASET_GROUP=A
SAMPLES=${SAMPLES:-2000}
EPOCHS=${EPOCHS:-10}
MRMR_SCORE_MODE=${MRMR_SCORE_MODE:-miq}
GENERATE_DATA=${GENERATE_DATA:-1}
TRAIN_PLAIN=${TRAIN_PLAIN:-1}
TRAIN_MRMR=${TRAIN_MRMR:-0}

mkdir -p "$BASE_DIR/artifacts/logs"
mkdir -p "$BASE_DIR/artifacts/models"
mkdir -p "$BASE_DIR/artifacts/reports"
mkdir -p "$BASE_DIR/data/processed"
mkdir -p /home/bwalidain/_scratch

source "$CONDA_SH"
conda activate "$ENV_NAME"

export LIBS_BASE_DIR="$BASE_DIR"
export MKL_VERBOSE=0
export OMP_NUM_THREADS=8
export PYTHONPATH="${PYTHONPATH:-}:$BASE_DIR"
export PYTHONUNBUFFERED=1

echo "[Study-A] SAMPLES=$SAMPLES EPOCHS=$EPOCHS"
echo "[Study-A] GENERATE_DATA=$GENERATE_DATA TRAIN_PLAIN=$TRAIN_PLAIN TRAIN_MRMR=$TRAIN_MRMR"

for SPECTRAL_TIER in L M H; do
  DATASET_TAG="${DATASET_GROUP}_${SPECTRAL_TIER}"
  DATASET_PATH="$BASE_DIR/data/processed/dataset_synthetic_${DATASET_TAG}.h5"
  SCRATCH_OUT="/home/bwalidain/_scratch/dataset_synthetic_${DATASET_TAG}.h5"

  if [ "$GENERATE_DATA" = "1" ]; then
    echo "[Study-A][Generate] $DATASET_TAG -> $DATASET_PATH"
    python "$BASE_DIR/scripts/generate_dataset.py" \
      --samples "$SAMPLES" \
      --dataset-group "$DATASET_GROUP" \
      --spectral-tier "$SPECTRAL_TIER" \
      --out "$SCRATCH_OUT"
    mv "$SCRATCH_OUT" "$DATASET_PATH"
  fi

  if [ "$TRAIN_PLAIN" = "1" ]; then
    MODEL_OUT="$BASE_DIR/artifacts/models/model_inversi_plain_pi_${DATASET_TAG}.pkl"
    REPORT_OUT="$BASE_DIR/artifacts/reports/model_inversi_plain_pi_${DATASET_TAG}_report.txt"
    echo "[Study-A][Train][plain] $DATASET_TAG -> $MODEL_OUT"
    python -u "$BASE_DIR/scripts/train_model.py" \
      --dataset "$DATASET_PATH" \
      --model pi \
      --out "$MODEL_OUT" \
      --report-out "$REPORT_OUT" \
      --epochs "$EPOCHS"
  fi

  if [ "$TRAIN_MRMR" = "1" ]; then
    MODEL_OUT="$BASE_DIR/artifacts/models/model_inversi_mrmr_${MRMR_SCORE_MODE}_pi_${DATASET_TAG}.pkl"
    REPORT_OUT="$BASE_DIR/artifacts/reports/model_inversi_mrmr_${MRMR_SCORE_MODE}_pi_${DATASET_TAG}_report.txt"
    echo "[Study-A][Train][mrmr] $DATASET_TAG -> $MODEL_OUT"
    python -u "$BASE_DIR/scripts/train_model.py" \
      --dataset "$DATASET_PATH" \
      --model pi \
      --out "$MODEL_OUT" \
      --report-out "$REPORT_OUT" \
      --epochs "$EPOCHS" \
      --mrmr \
      --mrmr-features 256 \
      --mrmr-pool 1024 \
      --mrmr-sample 500 \
      --mrmr-prefilter-stride 32 \
      --mrmr-score-mode "$MRMR_SCORE_MODE"
  fi
done

conda deactivate
