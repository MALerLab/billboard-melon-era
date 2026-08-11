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

# Use the uv-managed venv when present, so `bash scripts/...` works right after `uv sync`
# without activating anything. Override with PYTHON=/path/to/python.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=".venv/bin/python"
  elif command -v python >/dev/null 2>&1; then PY="python"
  else PY="python3"; fi
fi


for SEED in 77 78 79; do
  echo "=== Melon Baseline seed=${SEED} on GPU ${GPU} :: $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py \
    --config-name=packed_melon_clean model=Baseline train.seed="$SEED"
done
echo "=== done :: $(date '+%F %T') ==="
