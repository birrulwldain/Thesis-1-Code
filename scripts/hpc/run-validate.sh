#!/bin/bash
#SBATCH --job-name=libs_validate
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/bwalidain/birrulwldain/logs/validate_%j.out
#SBATCH --error=/home/bwalidain/birrulwldain/logs/validate_%j.err

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

python "$BASE_DIR/scripts/empirical_validation.py"

conda deactivate
