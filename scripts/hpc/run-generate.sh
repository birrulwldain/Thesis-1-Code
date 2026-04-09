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
SCRATCH_OUT=${SCRATCH_OUT:-/home/bwalidain/_scratch/dataset_synthetic.h5}
FINAL_OUT=${FINAL_OUT:-$BASE_DIR/data/dataset_synthetic.h5}

mkdir -p "$BASE_DIR/logs"

source "$CONDA_SH"
conda activate "$ENV_NAME"

export LIBS_BASE_DIR="$BASE_DIR"
export MKL_VERBOSE=0
export OMP_NUM_THREADS=8
export PYTHONPATH="${PYTHONPATH:-}:$BASE_DIR"

python "$BASE_DIR/scripts/generate_dataset.py" --samples "$SAMPLES" --out "$SCRATCH_OUT"
mv "$SCRATCH_OUT" "$FINAL_OUT"

conda deactivate
