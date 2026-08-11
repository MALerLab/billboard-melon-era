# Measuring Cross-Cultural Style Diffusion Through Era Classification

Code, chart-entry labels, artist-aware splits and training configurations for the ISMIR 2026
paper *Measuring Cross-Cultural Style Diffusion Through Era Classification*.

A CNN is trained to predict the chart-entry era of Billboard Hot 100 recordings, then applied
to Korean Melon chart recordings. The systematic gap between predicted and actual era — the
**era offset** — measures how far Korean popular music sits from the contemporaneous US
style axis, and how that distance narrows over time.

| | |
|---|---|
| Billboard | 22,002 tracks, 1958–2024, artist-disjoint 17,161 / 2,420 / 2,421 split |
| Melon | 2,771 tracks, 1964–2009, artist-disjoint 1,662 / 304 / 305 split |
| Models | 6 CNNs (baseline CNN, FCN, ShortChunkCNN, ShortChunkCNN_Res, Musicnn, CRNN) |
| Runs | 3 seeds × 6 architectures = 18 Billboard runs, plus 3 Melon reverse runs |
| Headline | Median era offset −4.7 / −4.5 / −4.1 / −2.4 / −2.7 years for the 1960s–2000s |

**No audio is distributed here.** The CSVs carry chart entries with the YouTube video IDs
used to obtain each recording; see [Data layout](#data-layout).

## Quick start — reproduce every table and figure without a GPU

The published numbers are all derived from per-segment year-probability `.npz` files. Those
are attached to the [latest release](https://github.com/malerlab/billboard-melon-era/releases),
together with the 21 selected checkpoints (~133 MB total), so nothing below needs audio, a
GPU, or retraining.

```bash
git clone https://github.com/malerlab/billboard-melon-era
cd billboard-melon-era
uv sync                                    # creates .venv and installs everything
source .venv/bin/activate                  # or prefix each command with `uv run`

# fetch the released artifacts into best_model/
gh release download -R malerlab/billboard-melon-era --pattern 'best_model.tar.gz'
tar xzf best_model.tar.gz

bash scripts/reproduce_tables_and_figures.sh
```

Tables land in `tables_out/`, figures in `figs_regen/`.

The `scripts/` drivers find `.venv/bin/python` on their own, so they work without
activating; override with `PYTHON=/path/to/python`.

Without `uv`, any environment with `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`,
`scipy` and `scikit-learn` works; see `pyproject.toml` for the pinned set. The analysis half
of the pipeline never imports CUDA.

## What produces what

```
chart CSVs + audio ──preprocess.py──►  whole-track .pt caches
                   ──make_slices.py─►  30-second segment caches

train.py --config-name=packed model=<Arch> train.seed=<77|78|79>
     └─► best_model/<MMDD_HHMM>_<Arch>_s<seed>/{config.yaml, best_selection.json, macro_*.pt}

infer_all.py --weights best_model --criteria macro
     └─► <run>/macro_{bill,kpop,kpop_heldout}_segment_year_probs.npz   ← the only interface
                                                                        between GPU work and
                                                                        paper numbers
final_tables.py       ──► Table 2, Table 3
case_study_table.py   ──► Table 4
make_ensemble_runs.py ──► seed-pooled npz ──► paper_figs.py ──► Figures 3–7
```

| Paper artifact | Command |
|---|---|
| Table 2 (in-domain accuracy) | `python final_tables.py --base best_model --tables 2A --eval-manifest csv/billboard_eval_asrun_2437.json` |
| Table 3 (era offset per decade) | `python final_tables.py --base best_model --tables 3A` |
| Table 4 (per-artist case study) | `python case_study_table.py --base best_model` |
| Figures 3–7 | `make_ensemble_runs.py` then `paper_figs.py --source ensemble --criterion macro` |

Figure 1 (`overview.png` in the paper) is a hand-drawn schematic with no generator.

## Training from scratch

```bash
bash scripts/train_all.sh 0        # 18 Billboard runs, one GPU — a multi-day job
bash scripts/train_melon.sh 0      # 3 Melon reverse runs, ~45 min each
python infer_all.py --weights best_model --criteria macro
```

Protocol, as reported in the paper and as implemented in `config/packed.yaml`: Adam,
lr 1e-4, batch 64, 30,000 iterations, validation every 500, the checkpoint with the highest
validation macro accuracy retained. Each decade is undersampled to the size of the smallest
class (2000s, 1,585 tracks), re-randomized every epoch, and a random 30-second crop is drawn
per track per epoch. Audio is mono 16 kHz; spectrogram settings follow each architecture's
original paper (`config/model/*.yaml`).

`train.py` seeds Python, NumPy, torch and CUDA from `train.seed`. Weights & Biases defaults
to `WANDB_MODE=offline`; set `WANDB_MODE=online` to log to your own project.

## Data layout

Audio is not redistributed. `csv/billboard_hot100_chosen.csv` and
`csv/korean_melon_meta_chosen.csv` give every chart entry with its YouTube video ID;
`data_prep/resolve_youtube_ids.py` documents how those IDs were resolved from the charts.
Obtain the recordings yourself under whatever terms apply to you, then place them as:

```
data/       {YYYY}_{Song}_{Artist}.mp3      # Billboard, 22,002 files
kpop_data/  {YYYY}_{Song}_{Artist}.wav      # Melon, 2,771 files
```

The literal braces are part of the filename and are load-bearing: the leading `{YYYY}` *is*
the chart-entry label. Point elsewhere with `BILLBOARD_AUDIO_ROOT` / `MELON_AUDIO_ROOT`.

Then build the caches (~200 GB, and training additionally wants a large amount of RAM
because whole tracks are held resident):

```bash
python preprocess.py                                            # whole-track .pt caches
python make_slices.py --audio-root data --ext .mp3 \
    --manifest csv/billboard_eval_asrun_2437.json --out-subdir test
python make_slices.py --audio-root kpop_data --ext .wav \
    --manifest csv/melon_eval_all_2771.json --out-subdir test
python make_slices.py --audio-root kpop_data --ext .wav \
    --manifest csv/melon_eval_heldout_305.json --out-subdir test_org
```

### Files in `csv/`

| File | Role |
|---|---|
| `billboard_hot100_chosen.csv` | 22,002 Billboard chart entries with YouTube video IDs |
| `korean_melon_meta_chosen.csv` | 2,771 Melon chart entries with YouTube video IDs |
| `billboard_artist_split.json` | Billboard artist-disjoint split (17,161 / 2,420 / 2,421) |
| `kpop_artist_split.json` | Melon artist-disjoint split (1,662 / 304 / 305) |
| `billboard_eval_asrun_2437.json` | Billboard tracks actually scored in the paper — see below |
| `billboard_eval_test_2421.json` | the clean test split, for the corrected numbers |
| `billboard_eval_leaked_16.json` | the 16 tracks that separate the two |
| `melon_eval_all_2771.json` | the whole Melon corpus, the cross-domain target |
| `melon_eval_heldout_305.json` | Melon held-out split, identical to `kpop_artist_split.json['test']` |
| `billboard_hot100_failed.csv` | chart entries with no retrievable audio (30,993 → 22,002) |
| `K_POP.csv` | K-pop acts that charted on the Hot 100 |

Splits are by *artist*, not by track: a collaboration graph is built over chart entries and
each connected component is assigned whole to train, validation or test, so no artist appears
on both sides. `artist_split.py` regenerates them.

## Known gaps

These are documented rather than silently fixed, because the released code is the code that
produced the published numbers.

**Billboard evaluation set.** The cached Billboard evaluation set holds 2,437 tracks: the
2,421 test tracks plus 13 train and 3 validation tracks, left behind by an earlier split.
Every published Table 2 and Table 3 number was computed over the 2,437. We measured the
effect: decade macro accuracy moves by at most 0.10 pp and micro by at most 0.18 pp. Both
manifests ship, so either number is reproducible:

| | CNN | FCN | SCNN | SCNNR | Musicnn | CRNN |
|---|---|---|---|---|---|---|
| macro, as published (2,437) | 67.0 | 69.6 | 71.2 | 68.1 | 70.1 | 69.0 |
| macro, clean test set (2,421) | 67.0 | 69.6 | 71.1 | 68.0 | 70.0 | 68.9 |

**Table 4 spread column.** The Median column of Table 4 reproduces exactly. The Spread
column does not: a sweep of 60 plausible definitions matched at best 2 of 5 rows, so the
published spread was computed ad hoc and its convention is not recoverable.
`case_study_table.py` states the definition it uses and should be read as a recomputation.
The caption's two excluded Sanullim tracks are not recorded anywhere; the two obvious
candidates (predicted ~20 years late, both later re-recordings) are named in the script.

**Undocumented loss term.** Training minimizes cross-entropy summed over the four hierarchy
levels plus a hierarchical consistency loss at weight 1.0 (`train.high_to_low_weight`,
`train.low_to_high_weight`). The paper describes the consistency loss but not that it is
computed over only the first 4 samples of each 64-sample batch (`trainer.py`, the
`len(labels[0])` slice). This is in the released code because it is in the published runs.

**Inert domain head.** `ClassificationHead` always builds a gradient-reversal domain
classifier. It is unused — `adversarial_loss_weight` is 0 everywhere and no adversarial
config ships — but its output width is included in the hierarchical-consistency
normalization, so it perturbs the gradient slightly. Present in every released checkpoint.

**Cross-machine variation.** The seed-77 runs were trained on two GPUs with `DataParallel`
(effective BatchNorm batch 32/GPU); the seed-78/79 runs on one GPU (batch 64). Their
checkpoints therefore differ in whether keys carry a `module.` prefix; `infer_all.py` handles
both. Retraining on different hardware will not reproduce checkpoints bit for bit — this is
the main reason the trained weights are released.

**Crop-consistency figures.** The paper reports within-song crop standard deviations of 3.3
years on Billboard and 6.0 on Melon; recomputing from the released runs gives 2.99 and 5.13.
`infer_all.py` writes these per run as `crop_consistency` in `{criterion}_summary.json`.

## Repository layout

```
train.py trainer.py dataset.py model_zoo.py modules.py   training pipeline
preprocess.py make_slices.py                             audio -> tensor caches
infer_all.py                                             checkpoints -> .npz predictions
final_tables.py case_study_table.py                      Tables 2, 3, 4
make_ensemble_runs.py paper_figs.py                       Figures 3-7
artist_split.py data_prep/resolve_youtube_ids.py         dataset construction
config/                                                  Hydra training configurations
csv/                                                     chart labels, splits, manifests
scripts/                                                 end-to-end driver scripts
```

`data_prep/resolve_youtube_ids.py` resolves chart entries to YouTube video IDs and stops
there; the bulk audio download step used to build the corpus is deliberately not distributed.

## Citation

```bibtex
@inproceedings{billboardmelonera2026,
  title     = {Measuring Cross-Cultural Style Diffusion Through Era Classification},
  booktitle = {Proceedings of the 27th International Society for Music Information
               Retrieval Conference (ISMIR)},
  year      = {2026}
}
```

## License

Code is MIT licensed (`LICENSE`). The model architectures are ported from
[sota-music-tagging-models](https://github.com/minzwon/sota-music-tagging-models) (MIT); see
`NOTICE` for attribution and for the provenance of the chart data.
