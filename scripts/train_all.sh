#!/bin/bash
# The 18 Billboard runs of the paper: 6 architectures x seeds 77 / 78 / 79.
#
#   bash scripts/train_all.sh [GPU_ID]
#
# One run at a time on one GPU. Each run is ~1-5 h on an RTX A5000 depending on the
# architecture (Musicnn is by far the slowest), so the full grid is a multi-day job.
# Checkpoints land in best_model/<MMDD_HHMM>_<Arch>_s<seed>/.
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


# Cheapest first so failures surface early; Musicnn last.
for ARCH in Baseline ShortChunkCNN ShortChunkCNN_Res FCN CRNN Musicnn; do
  for SEED in 77 78 79; do
    echo "=== ${ARCH} seed=${SEED} on GPU ${GPU} :: $(date '+%F %T') ==="
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py \
      --config-name=packed model="$ARCH" train.seed="$SEED"
  done
done
echo "=== done :: $(date '+%F %T') ==="
