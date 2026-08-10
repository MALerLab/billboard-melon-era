#!/bin/bash
# Regenerate every published table and figure from the released .npz artifacts.
# No GPU, no audio, no checkpoints needed. Takes a couple of minutes.
#
#   bash scripts/reproduce_tables_and_figures.sh [BEST_MODEL_DIR]
set -eu
cd "$(dirname "$0")/.."
BASE="${1:-best_model}"

echo "== Table 2 (in-domain Billboard accuracy), as evaluated in the paper =="
python final_tables.py --base "$BASE" --tables 2A \
  --eval-manifest csv/billboard_eval_asrun_2437.json --format both --out-dir tables_out

echo "== Table 2, clean artist-disjoint test split (see README, Billboard evaluation set) =="
python final_tables.py --base "$BASE" --tables 2A \
  --eval-manifest csv/billboard_eval_test_2421.json --format markdown --out-dir tables_out

echo "== Table 3 (median era offset per decade) =="
python final_tables.py --base "$BASE" --tables 3A --format both --out-dir tables_out

echo "== Table 4 (per-artist case study) =="
python case_study_table.py --base "$BASE"

echo "== Seed-pooled ensembles for the figures =="
python make_ensemble_runs.py --base "$BASE" --criterion macro --mode pool
python make_ensemble_runs.py --base "$BASE" --criterion macro --mode pool --melon

echo "== Figures 3-7 =="
python paper_figs.py --source ensemble --criterion macro

echo "== done: tables in tables_out/, figures in figs_regen/ =="
