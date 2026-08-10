#!/bin/bash
# The Melon-trained reverse-direction model (Section 5.4): Baseline x seeds 77 / 78 / 79.
#
#   bash scripts/train_melon.sh [GPU_ID]
#
# ~45 min per run on an RTX A5000. Pin a single GPU: these checkpoints were trained
# without DataParallel, so leaving both GPUs visible would halve the effective
# BatchNorm batch and change the results.
set -eu
GPU="${1:-0}"
cd "$(dirname "$0")/.."

for SEED in 77 78 79; do
  echo "=== Melon Baseline seed=${SEED} on GPU ${GPU} :: $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="$GPU" python train.py \
    --config-name=packed_melon_clean model=Baseline train.seed="$SEED"
done
echo "=== done :: $(date '+%F %T') ==="
