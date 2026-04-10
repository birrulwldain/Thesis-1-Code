#!/bin/bash

set -u

BASE_DIR=${BASE_DIR:-/home/bwalidain/thesis}
RUNNER=${RUNNER:-$BASE_DIR/scripts/hpc/run-train.sh}
EPOCHS=${EPOCHS:-10}

for DATASET_GROUP in A B; do
  for SPECTRAL_TIER in L M H; do
    for FEATURE_MODE in plain mrmr; do
      echo "[Submit] Group=$DATASET_GROUP Tier=$SPECTRAL_TIER FeatureMode=$FEATURE_MODE Epochs=$EPOCHS"
      DATASET_GROUP="$DATASET_GROUP" \
      SPECTRAL_TIER="$SPECTRAL_TIER" \
      FEATURE_MODE="$FEATURE_MODE" \
      EPOCHS="$EPOCHS" \
      sbatch "$RUNNER"
    done
  done
done
