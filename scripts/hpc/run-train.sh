#!/bin/bash
#SBATCH --job-name=libs_train
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/bwalidain/birrulwldain/logs/train_%j.out
#SBATCH --error=/home/bwalidain/birrulwldain/logs/train_%j.err

set -euo pipefail

BASE_DIR=/home/bwalidain/birrulwldain
CONDA_SH=/home/bwalidain/miniconda3/etc/profile.d/conda.sh
ENV_NAME=bw

source "$CONDA_SH"
conda activate "$ENV_NAME"

export LIBS_BASE_DIR="$BASE_DIR"
export MKL_VERBOSE=0
export OMP_NUM_THREADS=8
export PYTHONPATH="$PYTHONPATH:$BASE_DIR"

python "$BASE_DIR/scripts/train_inversion_model.py" \
  --mrmr \
  --mrmr-features 1024 \
  --mrmr-pool 4096 \
  --mrmr-sample 2000 \
  --mrmr-score-mode miq

conda deactivate
