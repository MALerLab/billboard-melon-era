#!/usr/bin/env python3
"""Regenerate the five code-generated figures of the ISMIR 2026 paper
*Measuring Cross-Cultural Style Diffusion Through Era Classification*.

Reads only the per-segment year-probability `.npz` files written by infer_all.py,
pooled across seeds by make_ensemble_runs.py. No audio, no checkpoints, no GPU.

  fig3  confusion matrix, segment-level year predictions, Billboard test set
        -> paper Figure 3 (figs/confusion_matrix.png)
  fig4  quarter-decade error barplot grid, Billboard vs Melon, shared y per decade
        -> paper Figure 4 (figs/barplot2.png); note the paper uses the *sharey* variant
  fig5  Billboard->Melon KDE by decade
        -> paper Figure 5 (figs/final_kde_bill_to_kpop.png)
  fig6  per-architecture 1970s->2000s mode shift, one arrow per model
        -> paper Figure 6 (figs/kde_plot_every_model.pdf); the paper uses the
           `fig6_kde_every_modelb.pdf` arrow variant
  fig7  Melon-trained model on the Billboard test set (reverse direction)
        -> paper Figure 7 (figs/final_kde_kpop_to_bill.png)

Usage:
  python paper_figs.py --source ensemble --criterion macro
  python paper_figs.py --source ensemble --figs 5 6          # subset of figures
  python paper_figs.py --run-dir <run> --criterion macro     # fig3/4/5 for one run

--arch picks the architecture for the single-model figures (fig3/4/5); paper aliases
(CNN, SCNN, SCNNR) are accepted. fig6 always draws all six architectures; fig7 uses
the Melon-trained ensemble. --run-dir points at any directory holding
{criterion}_{bill,kpop}_segment_year_probs.npz and draws fig3/4/5 from it.

Figures render in Times New Roman when available; on a machine without it matplotlib
silently falls back to DejaVu, which changes the typography but not the data.
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*findfont.*')
pd.options.mode.chained_assignment = None

ROOT = Path(__file__).resolve().parent
YEAR0 = 1958
N_YEARS = 67  # 1958..2024, Hierarchy3 class indices

ARCHS = ['Baseline', 'ShortChunkCNN', 'ShortChunkCNN_Res', 'FCN', 'CRNN', 'Musicnn']
ARCH_LABEL = {'Baseline': 'CNN', 'ShortChunkCNN': 'SCNN', 'ShortChunkCNN_Res': 'SCNNR',
              'FCN': 'FCN', 'CRNN': 'CRNN', 'Musicnn': 'Musicnn'}
ARCH_ALIAS = {**{a.lower(): a for a in ARCHS},
              **{lbl.lower(): a for a, lbl in ARCH_LABEL.items()}}
# Seed-ensemble of the 18 dproto runs: per-segment year distributions pooled over
# seeds 77/78/79 by make_ensemble_runs.py. Every figure then rests on the same
# 18-run basis the tables report, instead of one arbitrary seed.
# Set by main() from --ens-dir (default: figs_regen/_ensemble_runs).
ENS_BASE = Path('figs_regen/_ensemble_runs')
ENS_MELON_RUN = 'MELON_ENSEMBLE_s77'


def _ens_run(arch):
  return ENS_BASE / f'ENSEMBLE_{arch}_s77'


# fig3/4/5 describe "the model" as a whole, and sit next to tables computed over
# all 18 runs, so they are drawn from the pooled 18-run group rather than one
# architecture; fig6 keeps per-architecture groups, which is its whole point.
def _ens_all():
  return ENS_BASE / 'ENSEMBLE_ALL_s77'
FIG6_ORDER = ['Baseline', 'FCN', 'ShortChunkCNN', 'ShortChunkCNN_Res', 'Musicnn', 'CRNN']


# ---------------------------------------------------------------- data loading

def quarter_class(year):
  """Year -> Hierarchy2 quarter-decade class index, verbatim dataset.py:_get_quarter."""
  if year < 1970:
    if year < 1964:
      return 0 if year < 1961 else 1
    return 2 if year < 1967 else 3
  if year >= 2010:
    if year < 2017:
      return 20 if year < 2014 else 21
    return 22 if year < 2021 else 23
  base = 4 * ((year - (year % 10)) - 1960) // 10
  d = year % 10
  if d < 5:
    return base + (0 if d < 3 else 1)
  return base + (2 if d < 8 else 3)


def frame_from_csv(path):
  """Paper-era CSV already has the notebook schema."""
  return pd.read_csv(path)


def frame_from_npz(path):
  """infer_all.py segment npz -> notebook CSV schema (Hierarchy2 + Hierarchy3 rows).

  npz keys: files (N,) '{YYYY}_{title}_{artist}_{seg}.pt', probs (N, 67) float16
  over year classes 1958-2024. Predictions are per-segment argmax, matching the
  paper's segment-level Hierarchy3 semantics.
  """
  z = np.load(path, allow_pickle=True)
  files = [str(f) for f in z['files']]
  pred_idx = z['probs'].argmax(axis=1).astype(int)
  true_year = np.array([int(re.match(r'\{(\d{4})\}', f).group(1)) for f in files])
  pred_year = YEAR0 + pred_idx
  h3 = pd.DataFrame({'Hierarchy': 'Hierarchy3', 'Filename': files,
                     'TrueLabel': true_year - YEAR0, 'PredictedLabel': pred_idx,
                     'Loss': np.nan})
  h2 = pd.DataFrame({'Hierarchy': 'Hierarchy2', 'Filename': files,
                     'TrueLabel': [quarter_class(y) for y in true_year],
                     'PredictedLabel': [quarter_class(y) for y in pred_year],
                     'Loss': np.nan})
  return pd.concat([h2, h3], ignore_index=True)


def load_frame(spec):
  if spec is None:
    return None
  spec = Path(spec)
  return frame_from_npz(spec) if spec.suffix == '.npz' else frame_from_csv(spec)


def build_source(source, criterion, arch='Baseline'):
  """Return dict of input specs for the camera-ready source condition.

  Only one source ships: the seed-pooled ensemble of the 18 dproto runs, built by
  make_ensemble_runs.py. arch selects the model behind the single-model figures
  (fig3/4/5); kpop_by_model (fig6) and bill_from_melon (fig7) are arch-independent.
  """
  if source != 'ensemble':
    raise ValueError(source)
  suffix = '' if arch == 'Baseline' else f'-{ARCH_LABEL[arch].lower()}'
  melon = ENS_BASE / ENS_MELON_RUN / f'{criterion}_bill_segment_year_probs.npz'
  return {
      'name': f'ensemble-{criterion}' + suffix,
      'bill_baseline': _ens_all() / f'{criterion}_bill_segment_year_probs.npz',
      'kpop_baseline': _ens_all() / f'{criterion}_kpop_segment_year_probs.npz',
      'kpop_by_model': {a: _ens_run(a) / f'{criterion}_kpop_segment_year_probs.npz'
                        for a in ARCHS},
      'bill_from_melon': melon if melon.exists() else None,
  }


def build_custom(run_dir, criterion, melon_run_dir=None):
  """Spec for an arbitrary run directory (a future retrain, an ablation, ...)."""
  run_dir = Path(run_dir)
  melon = (Path(melon_run_dir) / f'{criterion}_bill_segment_year_probs.npz'
           if melon_run_dir else None)
  return {
      'name': f'custom-{run_dir.name}-{criterion}',
      'bill_baseline': run_dir / f'{criterion}_bill_segment_year_probs.npz',
      'kpop_baseline': run_dir / f'{criterion}_kpop_segment_year_probs.npz',
      'kpop_by_model': None,  # fig6 needs the six named architectures of a source
      'bill_from_melon': melon,
  }


# ------------------------------------------------------------- notebook pieces

def calculate_kde_metrics(data):
  """plot.ipynb cell 19 verbatim (Scott-bandwidth scipy KDE on a 1000-pt grid)."""
  kde = gaussian_kde(data)
  x = np.linspace(min(data), max(data), 1000)
  y = kde(x)
  return {'max_density_x': float(x[np.argmax(y)]), 'max_density_y': float(max(y)),
          'kde_mean': float((x * y).sum() / y.sum())}


def calculate_decades(data):
  """plot.ipynb cell 21 verbatim."""
  decades = []
  for filename in data['Filename']:
    year = int(filename.split('_')[0][1:-1])
    if year < 1970:
      decades.append('1960s')
    elif year < 1980:
      decades.append('1970s')
    elif year < 1990:
      decades.append('1980s')
    elif year < 2000:
      decades.append('1990s')
    elif year < 2010:
      decades.append('2000s')
    else:
      decades.append('2010s')
  return decades


def serif_theme():
  """plot.ipynb cell 20 (as executed; retina display -> we save at dpi=200 instead)."""
  sns.set_theme({'font.family': 'serif', 'font.serif': ['Times New Roman']},
                rc={'axes.unicode_minus': False}, style=None)


def reset_style():
  sns.reset_defaults()
  matplotlib.rcParams.update(matplotlib.rcParamsDefault)
  matplotlib.use('Agg')


# --------------------------------------------------------------------- figures

def fig3_confusion(df, out_png):
  """plot.ipynb cell 16 with the cell-17 argument swap corrected.

  Ran before cell 20's theme in the paper session -> default matplotlib style.
  """
  reset_style()
  h3 = df[df['Hierarchy'] == 'Hierarchy3']
  labels = h3['TrueLabel'].to_numpy()
  preds = h3['PredictedLabel'].to_numpy()
  class_name = [str(y) for y in range(YEAR0, YEAR0 + N_YEARS)]

  cm = confusion_matrix(labels, preds, labels=range(N_YEARS), normalize='true')
  cm = np.nan_to_num(cm)

  fig, ax = plt.subplots(figsize=(14, 14))
  sns.heatmap(cm, annot=False, cmap='Oranges', cbar=True, square=True,
              xticklabels=class_name, yticklabels=class_name,
              cbar_kws={'shrink': 0.8, 'aspect': 20}, ax=ax)
  ax.set_title('One year Confusion Matrix', size=25, pad=12)
  ax.set_ylabel('True Label', size=20, labelpad=10)
  ax.set_xlabel('Predicted Label', size=20, labelpad=10)
  tick_indices = list(range(2, len(class_name), 5))  # 1960, 1965, ... 2020
  tick_labels = [class_name[i] for i in tick_indices]
  ax.set_xticks(tick_indices)
  ax.set_yticks(tick_indices)
  ax.set_xticklabels(tick_labels, fontsize=14)
  ax.set_yticklabels(tick_labels, fontsize=14)
  ax.set_xticks(np.arange(len(class_name)) + 0.5, minor=True)
  ax.set_yticks(np.arange(len(class_name)) + 0.5, minor=True)
  ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
  ax.tick_params(which='minor', bottom=False, left=False)
  ax.invert_yaxis()
  fig.savefig(out_png, dpi=100, bbox_inches='tight')
  plt.close(fig)

  err = (preds - labels)
  dec = np.array(calculate_decades(h3))
  stats = {'n_segments': int(len(h3)),
           'micro_acc_year': float((preds == labels).mean()),
           'within_2yr': float((np.abs(err) <= 2).mean()),
           'mean_error_yr': float(err.mean()),
           'mean_abs_error_yr': float(np.abs(err).mean()),
           'per_decade_within_2yr': {d: float((np.abs(err[dec == d]) <= 2).mean())
                                     for d in ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s']}}
  return stats


def fig4_barplot(df_bill, df_kpop, out_png, sharey=False):
  """plot.ipynb cell 28 verbatim (cell 27 theme); sharey='row' is the R1 fix."""
  reset_style()
  sns.set_theme(style='white')
  bill = df_bill.copy()
  kpop = df_kpop.copy()
  bill['Decade'] = calculate_decades(bill)
  kpop['Decade'] = calculate_decades(kpop)

  stats = {}
  if sharey:
    fig, axgrid = plt.subplots(5, 2, figsize=(10, 13), sharey='row')
  else:
    fig = plt.figure(figsize=(10, 13))
  for i, decade in enumerate(['1960s', '1970s', '1980s', '1990s', '2000s']):
    for j, (dataset, name, color) in enumerate(zip([bill, kpop], ['Billboard', 'Melon'],
                                                   ['orangered', 'limegreen'])):
      ax = axgrid[i][j] if sharey else plt.subplot(5, 2, 2 * i + j + 1)
      subset = dataset[(dataset['Hierarchy'] == 'Hierarchy2') & (dataset['Decade'] == decade)].copy()
      subset['Difference_hierarchy2'] = subset['PredictedLabel'] - subset['TrueLabel']
      difference_counts = subset['Difference_hierarchy2'].value_counts().sort_index()
      raw_total = int(difference_counts.sum())
      difference_counts = difference_counts.reindex(range(-10, 10), fill_value=0)
      total_counts = difference_counts.sum()
      normalized_counts = difference_counts / total_counts if total_counts != 0 else difference_counts
      sns.barplot(x=difference_counts.index, y=normalized_counts.values, color=color, ax=ax)
      ax.set_title(f'{name} {decade}')
      ax.grid(True)
      ax.axvline(10, color='k', linestyle='--', linewidth=1)
      ax.set_xlabel('')
      stats[f'{name}_{decade}'] = {
          'n_segments': raw_total,
          'in_window': int(total_counts),
          'dropped_frac': float(1 - total_counts / raw_total) if raw_total else 0.0,
          'mean_diff_quarter': float(subset['Difference_hierarchy2'].mean()),
          'hist': {str(k): float(v) for k, v in normalized_counts.items()},
      }
  fig.supxlabel('Predicted - True(Quarter Decade)')
  fig.supylabel('Normalized Count')
  fig.tight_layout()
  fig.savefig(out_png, dpi=200, bbox_inches='tight')
  plt.close(fig)
  return stats


def kde_by_decade(df, out_png, include_2010s=False, inset_2010s=False):
  """plot.ipynb cell 22 (fig5) / cell 24 paper-cosmetics variant (fig7).

  fig5: include_2010s=False. fig7: include_2010s=True, inset stays 5-decade
  (git blob 5b88385 state), no title, xlabel 'Predicted - True'.
  """
  reset_style()
  serif_theme()
  results = df.copy()
  results['Decade'] = calculate_decades(results)
  sns.set_theme(style='whitegrid')

  final_h = results[results['Hierarchy'] == 'Hierarchy3'].copy()
  final_h['LabelDifference'] = final_h['PredictedLabel'] - final_h['TrueLabel']
  plt.figure(figsize=(8, 5))
  ax = plt.gca()
  colors = {'1960s': 'red', '1970s': 'orange', '1980s': 'green', '1990s': 'blue',
            '2000s': 'purple', '2010s': 'black'}
  decades = ['1960s', '1970s', '1980s', '1990s', '2000s'] + (['2010s'] if include_2010s else [])

  stats = {}
  for decade in decades:
    subset = final_h[final_h['Decade'] == decade]['LabelDifference']
    color = colors[decade]
    sns.kdeplot(subset, bw_adjust=1, label=decade, fill=True, color=color)
    metrics = calculate_kde_metrics(subset)
    metrics['n_segments'] = int(len(subset))
    metrics['mean_diff'] = float(subset.mean())
    stats[decade] = metrics
    print(f'  {decade}: mode {metrics["max_density_x"]:+.2f} '
          f'(density {metrics["max_density_y"]:.4f}), mean {metrics["mean_diff"]:+.2f}, '
          f'n={metrics["n_segments"]}')
    plt.scatter(metrics['max_density_x'], metrics['max_density_y'], s=50, color=color)

  ax.set_xlim(-15, 15)
  ax.axvline(0, color='k', linestyle='--', linewidth=1)
  ax.set_xlabel('Predicted - True')
  ax.set_ylabel('Density')
  ax.legend(loc='upper left', fontsize=8)

  sns.set_theme(style='white')
  axins = inset_axes(ax, width='30%', height='30%', loc='upper right')
  inset_decades = decades if (include_2010s and inset_2010s) else \
      ['1960s', '1970s', '1980s', '1990s', '2000s']
  for decade in inset_decades:
    subset = final_h[final_h['Decade'] == decade]['LabelDifference']
    sns.kdeplot(subset, bw_adjust=1.5, label=decade, fill=True, color=colors[decade], ax=axins)
  axins.axvline(0, color='k', linestyle='--', linewidth=1)
  axins.xaxis.set_visible(True)
  axins.set_xlabel('')
  axins.yaxis.set_visible(True)
  axins.set_ylabel('')
  axins.legend().set_visible(False)
  plt.tight_layout()
  plt.savefig(out_png, dpi=200, bbox_inches='tight')
  plt.close()
  return stats


def fig6_models(frames, out_base):
  """frames: {arch: notebook-df}. Reconstruction of the published fig6
  (figs/kde_plot_every_model.png), matched against the image extracted from the
  paper PDF (figs_regen/paper_original/): one axes on the fig5 canvas (8x5,
  whitegrid, dpi=200 tight -> 1565x964), 1970s curves in Oranges and 2000s in
  Blues (light->dark in FIG6_ORDER), mode dots (cell-19 KDE metrics, s=50),
  legend '{model} - {decade}': 1970s box upper right, 2000s box lower right.
  """
  order = [a for a in FIG6_ORDER if a in frames]
  palettes = {'1970s': sns.color_palette('Oranges', len(FIG6_ORDER)),
              '2000s': sns.color_palette('Blues', len(FIG6_ORDER))}
  data = {}
  stats = {}
  for arch in order:
    res = frames[arch].copy()
    res['Decade'] = calculate_decades(res)
    h3 = res[res['Hierarchy'] == 'Hierarchy3'].copy()
    h3['LabelDifference'] = h3['PredictedLabel'] - h3['TrueLabel']
    data[arch] = {d: h3[h3['Decade'] == d]['LabelDifference'] for d in ['1970s', '2000s']}
    stats[ARCH_LABEL[arch]] = {}
    for d in ['1970s', '2000s']:
      m = calculate_kde_metrics(data[arch][d])
      m['n_segments'] = int(len(data[arch][d]))
      m['mean_diff'] = float(data[arch][d].mean())
      stats[ARCH_LABEL[arch]][d] = m
      print(f'  {ARCH_LABEL[arch]:8s} {d}: mode {m["max_density_x"]:+.2f}, '
            f'mean {m["mean_diff"]:+.2f}')

  reset_style()
  serif_theme()
  sns.set_theme(style='whitegrid')
  plt.figure(figsize=(8, 5))
  ax = plt.gca()
  for dec in ['1970s', '2000s']:
    for arch in order:
      color = palettes[dec][FIG6_ORDER.index(arch)]
      sns.kdeplot(data[arch][dec], bw_adjust=1, label=f'{ARCH_LABEL[arch]} - {dec}',
                  color=color, linewidth=2)
      m = stats[ARCH_LABEL[arch]][dec]
      plt.scatter(m['max_density_x'], m['max_density_y'], s=50, color=color, zorder=5)
  ax.set_xlim(-15, 15)
  ax.axvline(0, color='k', linestyle='--', linewidth=1)
  ax.set_xlabel('Predicted - True(One Year)')
  ax.set_ylabel('Density')
  handles, labels = ax.get_legend_handles_labels()
  n = len(order)
  leg1 = ax.legend(handles[:n], labels[:n], loc='upper right', fontsize=8)
  ax.add_artist(leg1)
  ax.legend(handles[n:], labels[n:], loc='lower right', fontsize=8)
  plt.tight_layout()
  plt.savefig(f'{out_base}.png', dpi=200, bbox_inches='tight')
  plt.close()

  fig6b_mode_shift(order, data, stats, palettes, f'{out_base}b.png')
  return stats


def fig6b_mode_shift(order, data, stats, palettes, out_png):
  """fig6b variant: same canvas, curves faded to background, one arrow per model
  from its 1970s mode point (Oranges dot) to its 2000s mode point (Blues dot).
  """
  reset_style()
  serif_theme()
  sns.set_theme(style='whitegrid')
  plt.figure(figsize=(8, 5))
  ax = plt.gca()
  for dec in ['1970s', '2000s']:
    for arch in order:
      color = palettes[dec][FIG6_ORDER.index(arch)]
      sns.kdeplot(data[arch][dec], bw_adjust=1, color=color, linewidth=2, alpha=0.45)
  placed = []
  for arch in order:
    shade = FIG6_ORDER.index(arch)
    m70 = stats[ARCH_LABEL[arch]]['1970s']
    m00 = stats[ARCH_LABEL[arch]]['2000s']
    p70 = (m70['max_density_x'], m70['max_density_y'])
    p00 = (m00['max_density_x'], m00['max_density_y'])
    ax.scatter(*p70, s=50, color=palettes['1970s'][shade], zorder=5)
    ax.scatter(*p00, s=50, color=palettes['2000s'][shade], zorder=5)
    short = abs(p00[0] - p70[0]) < 1.5  # barely-moved models: arc out so the arrow survives
    ax.annotate('', xy=p00, xytext=p70, zorder=4,
                arrowprops=dict(arrowstyle='-|>', color='0.35', lw=1.4,
                                connectionstyle=f'arc3,rad={0.55 if short else 0.18}',
                                shrinkA=4 if short else 7, shrinkB=4 if short else 7))
    # label the 1970s end; flip to the other side of the dot when a previous
    # label already sits within collision distance
    near = any(abs(p70[0] - x) < 2.2 and abs(p70[1] - y) < 0.004 for x, y in placed)
    off, ha = ((8, -13), 'left') if near else ((-7, 5), 'right')
    placed.append(p70)
    ax.annotate(ARCH_LABEL[arch], p70, textcoords='offset points', xytext=off,
                ha=ha, fontsize=9, color='0.2', zorder=6)
  ax.set_xlim(-15, 15)
  ax.axvline(0, color='k', linestyle='--', linewidth=1)
  ax.set_xlabel('Predicted - True(One Year)')
  ax.set_ylabel('Density')
  from matplotlib.lines import Line2D
  ax.legend(handles=[Line2D([], [], marker='o', ls='', color=palettes['1970s'][4], label='1970s mode'),
                     Line2D([], [], marker='o', ls='', color=palettes['2000s'][4], label='2000s mode')],
            loc='upper left', fontsize=8)
  plt.tight_layout()
  plt.savefig(out_png, dpi=200, bbox_inches='tight')
  plt.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches='tight')  # R3: vector
  plt.close()


# ------------------------------------------------------------------ driver

def maybe_frame(path, what):
  if path is None:
    return None
  if not Path(path).exists():
    print(f'  !! {what}: missing {path} — skipping dependent figures')
    return None
  return load_frame(path)


def run_spec(spec, outroot, figs=(3, 4, 5, 6, 7), arch='Baseline', model_label=None,
             sharey_variant=True):
  label = model_label or ARCH_LABEL[arch]
  outdir = outroot / spec['name']
  outdir.mkdir(parents=True, exist_ok=True)
  print(f'== source {spec["name"]} -> {outdir}  (figs {sorted(figs)}, model {label})')
  stats = {'source': spec['name'], 'model': label,
           'inputs': {k: ({a: str(p) for a, p in v.items()} if isinstance(v, dict) else str(v))
                      for k, v in spec.items() if k != 'name'}}

  bill = maybe_frame(spec['bill_baseline'], 'Billboard predictions') \
      if {3, 4} & set(figs) else None
  kpop = maybe_frame(spec['kpop_baseline'], 'Melon predictions') \
      if {4, 5} & set(figs) else None

  if 3 in figs and bill is not None:
    print(f' fig3 (confusion matrix, Billboard test, {label})')
    stats['fig3'] = fig3_confusion(bill, outdir / 'fig3_confusion_matrix.png')
    print(f'   year-acc {stats["fig3"]["micro_acc_year"]:.3f}, '
          f'|err| {stats["fig3"]["mean_abs_error_yr"]:.2f} yr')

  if 4 in figs and bill is not None and kpop is not None:
    print(' fig4 (quarter-decade barplot grid)')
    stats['fig4'] = fig4_barplot(bill, kpop, outdir / 'fig4_barplot2.png', sharey=False)
    if sharey_variant:
      fig4_barplot(bill, kpop, outdir / 'fig4_barplot2_sharey.png', sharey=True)

  if 5 in figs and kpop is not None:
    print(f' fig5 (Billboard->Melon KDE by decade, {label})')
    stats['fig5'] = kde_by_decade(kpop, outdir / 'fig5_kde_bill_to_kpop.png', include_2010s=False)

  if 6 in figs:
    if spec['kpop_by_model'] is None:
      print(' fig6 skipped (custom run dir has no six-architecture set)')
    else:
      frames = {a: f for a, p in spec['kpop_by_model'].items()
                if (f := maybe_frame(p, f'fig6 {ARCH_LABEL[a]}')) is not None}
      if frames:
        print(f' fig6 ({len(frames)}/6 architectures, 1970s vs 2000s)')
        stats['fig6'] = fig6_models(frames, outdir / 'fig6_kde_every_model')

  if 7 in figs:
    if spec['bill_from_melon'] is None:
      print(' fig7 skipped (no Melon-trained model in this condition)')
    else:
      rev = maybe_frame(spec['bill_from_melon'], 'Melon-trained reverse predictions')
      if rev is not None:
        print(' fig7 (Melon->Billboard reverse KDE)')
        # 2010s dropped: the Melon-trained model has no 2010s labels (output caps
        # at 2009), so Billboard-2010s offsets are pure truncation artifacts.
        stats['fig7'] = kde_by_decade(rev, outdir / 'fig7_kde_kpop_to_bill.png',
                                      include_2010s=False, inset_2010s=False)

  # merge so a partial --figs run doesn't wipe stats of figures drawn earlier
  stats_path = outdir / 'stats.json'
  if stats_path.exists():
    stats = {**json.load(open(stats_path)), **stats}
  with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=1)
  print(f' stats -> {stats_path}')
  return stats


def compare(outroot):
  """Montages + KDE-mode tables across whichever source dirs exist."""
  import matplotlib.image as mpimg
  dirs = sorted([d for d in outroot.iterdir() if (d / 'stats.json').exists()])
  if not dirs:
    print('nothing to compare')
    return
  cmpdir = outroot / 'comparison'
  cmpdir.mkdir(exist_ok=True)
  allstats = {d.name: json.load(open(d / 'stats.json')) for d in dirs}

  figures = ['fig3_confusion_matrix.png', 'fig4_barplot2.png', 'fig5_kde_bill_to_kpop.png',
             'fig6_kde_every_modelb.png', 'fig7_kde_kpop_to_bill.png']
  reset_style()
  for figname in figures:
    avail = [(d.name, d / figname) for d in dirs if (d / figname).exists()]
    if len(avail) < 2:
      continue
    fig, axes = plt.subplots(1, len(avail), figsize=(7 * len(avail), 7))
    for ax, (name, path) in zip(np.atleast_1d(axes), avail):
      ax.imshow(mpimg.imread(path))
      ax.set_title(name, fontsize=16)
      ax.axis('off')
    fig.suptitle(figname.replace('.png', ''), fontsize=18)
    fig.tight_layout()
    fig.savefig(cmpdir / f'cmp_{figname}', dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'montage -> {cmpdir / ("cmp_" + figname)}')

  lines = ['# KDE mode comparison (Predicted - True, years; segment-level)', '']
  lines += ['## fig5 Billboard->Melon, baseline CNN — paper Table 3 = '
            '-5.81 / -8.34 / -4.41 / -3.32 / -0.57', '']
  header = '| source | 1960s | 1970s | 1980s | 1990s | 2000s |'
  lines += [header, '|---' * 6 + '|']
  for name, st in allstats.items():
    if st.get('fig5'):
      row = [f'{st["fig5"][d]["max_density_x"]:+.2f}' if d in st['fig5'] else '—'
             for d in ['1960s', '1970s', '1980s', '1990s', '2000s']]
      lines.append('| ' + name + ' | ' + ' | '.join(row) + ' |')
  lines += ['', '## fig6 modes per architecture (1970s / 2000s)', '']
  archs = ['CNN', 'SCNN', 'SCNNR', 'FCN', 'CRNN', 'Musicnn']
  lines += ['| source | ' + ' | '.join(archs) + ' |', '|---' * 7 + '|']
  for name, st in allstats.items():
    if st.get('fig6'):
      row = [f'{st["fig6"][a]["1970s"]["max_density_x"]:+.1f} / '
             f'{st["fig6"][a]["2000s"]["max_density_x"]:+.1f}'
             if a in st['fig6'] else '—' for a in archs]
      lines.append('| ' + name + ' | ' + ' | '.join(row) + ' |')
  lines += ['', '## fig7 Melon->Billboard modes', '']
  lines += ['| source | 1960s | 1970s | 1980s | 1990s | 2000s | 2010s |', '|---' * 7 + '|']
  for name, st in allstats.items():
    if st.get('fig7'):
      row = [f'{st["fig7"][d]["max_density_x"]:+.2f}' if d in st['fig7'] else '—'
             for d in ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s']]
      lines.append('| ' + name + ' | ' + ' | '.join(row) + ' |')
  lines += ['', '## fig3 Billboard year-level accuracy', '']
  lines += ['| source | year acc | within ±2yr | mean |err| |', '|---|---|---|---|']
  for name, st in allstats.items():
    f3 = st.get('fig3')
    if f3:
      lines.append(f'| {name} | {f3["micro_acc_year"]:.3f} | {f3["within_2yr"]:.3f} | '
                   f'{f3["mean_abs_error_yr"]:.2f} |')
  (cmpdir / 'comparison.md').write_text('\n'.join(lines) + '\n')
  print(f'tables -> {cmpdir / "comparison.md"}')


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--source', nargs='+', default=[], choices=['ensemble'],
                  help='seed-pooled ensemble of the 18 runs, built by make_ensemble_runs.py')
  ap.add_argument('--ens-dir', default='figs_regen/_ensemble_runs',
                  help='directory holding ENSEMBLE_*/ and MELON_ENSEMBLE_s77/ '
                       '(output of make_ensemble_runs.py)')
  ap.add_argument('--criterion', nargs='+', default=['macro'], choices=['macro', 'loss'],
                  help="checkpoint-selection criterion; the paper uses 'macro'")
  ap.add_argument('--arch', default='Baseline',
                  help='model for the single-model figures fig3/4/5: '
                       'Baseline|ShortChunkCNN|ShortChunkCNN_Res|FCN|CRNN|Musicnn '
                       '(paper aliases CNN/SCNN/SCNNR accepted)')
  ap.add_argument('--figs', nargs='+', type=int, default=[3, 4, 5, 6, 7],
                  choices=[3, 4, 5, 6, 7], help='which paper figures to draw')
  ap.add_argument('--run-dir', default=None,
                  help='any run directory with {criterion}_{bill,kpop}_segment_year_probs.npz '
                       '(fig3/4/5; fig6 needs a named source)')
  ap.add_argument('--melon-run-dir', default=None,
                  help='Melon-trained run directory for fig7 when using --run-dir')
  ap.add_argument('--outdir', default='figs_regen')
  ap.add_argument('--compare', action='store_true')
  args = ap.parse_args()

  arch = ARCH_ALIAS.get(args.arch.lower())
  if arch is None:
    raise SystemExit(f'unknown --arch {args.arch!r}; choose from '
                     f'{sorted(set(ARCHS) | set(ARCH_LABEL.values()))}')

  global ENS_BASE
  ENS_BASE = Path(args.ens_dir)
  if not ENS_BASE.is_absolute():
    ENS_BASE = ROOT / ENS_BASE

  outroot = ROOT / args.outdir
  for source in args.source:
    for crit in args.criterion:
      run_spec(build_source(source, crit, arch), outroot, figs=set(args.figs), arch=arch)
  if args.run_dir:
    for crit in args.criterion:
      run_spec(build_custom(args.run_dir, crit, args.melon_run_dir), outroot,
               figs=set(args.figs), arch=arch, model_label=Path(args.run_dir).name)
  if args.compare:
    compare(outroot)


if __name__ == '__main__':
  main()
