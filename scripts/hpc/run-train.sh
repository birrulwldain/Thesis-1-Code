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

mkdir -p "$BASE_DIR/logs"

source "$CONDA_SH"
conda activate "$ENV_NAME"

export LIBS_BASE_DIR="$BASE_DIR"
export MKL_VERBOSE=0
export OMP_NUM_THREADS=8
export PYTHONPATH="${PYTHONPATH:-}:$BASE_DIR"
export PYTHONUNBUFFERED=1

python -u "$BASE_DIR/scripts/train_inversion_model.py" \
  --mrmr \
  --mrmr-features 256 \
  --mrmr-pool 1024 \
  --mrmr-sample 500 \
  --mrmr-score-mode miq \
  --epochs 3

conda deactivate
