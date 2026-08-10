#!/usr/bin/env python3
"""Table 4 — year-level era offsets for five representative Korean artists.

Reads the same per-segment year-probability `.npz` files as `final_tables.py`, so it needs
no audio, no checkpoints and no GPU.

For each artist and each run: every track's offset is (predicted year - chart-entry year),
where the predicted year is the argmax of the track's mean segment softmax. The reported
median is the mean +/- SD of the per-run per-artist median across runs; the spread is the
10th-90th percentile of the per-song offsets pooled over runs.

    python case_study_table.py --base <best_model> [<best_model> ...]

Reproduction status (measured, see README "Known gaps"):
  * The Median column reproduces the published table exactly for all five artists
    (-6.1+/-0.7, -5.2+/-2.0, +2.8+/-2.2, -0.1+/-0.8, -1.0+/-1.1).
  * The Spread column does NOT reproduce. A sweep of 60 plausible definitions
    (18-run vs seed-77-only x pooled/per-track-median/per-track-mean x five percentile
    interpolations x 10-90 and 5-95, with and without the Sanullim exclusion) matched at
    best 2 of 5 rows. The published spread was computed ad hoc and its exact convention
    is not recoverable. This script uses the definition stated above; treat its Spread
    column as a recomputation, not a reproduction.

The caption's "two Sanullim entries whose audio turned out to be non-original recordings"
are not recorded anywhere, but two tracks stand out unambiguously at roughly +20 years
(SANULLIM_NON_ORIGINAL below); pass them via --exclude to drop them from the spread.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

YEAR_MIN, YEAR_MAX = 1958, 2024
YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1, dtype=np.float64)
RUN_DIR_RE = re.compile(r'^(\d{4})_(\d{4})_(.+)_s(\d+)$')
TRACK_RE = re.compile(r'^\{(\d{4})\}_\{(.*)\}_\{(.*)\}$')

# Artist label -> the exact artist strings as they appear in the cached filenames.
# Verified counts against the paper: 12 / 13 / 10 / 14 / 6.
ARTISTS = [
    ('Choi Hee-jun', '1964--69', ['최희준']),
    ('Sanullim',     '1978--88', ['산울림']),
    ('Seo Taiji',    '1992--03', ['서태지와 아이들', '서태지']),
    ('BigBang',      '2007--09', ['Bigbang', 'Bigbang, 2ne1']),
    ("Girls' Gen.",  '2007--09', ['소녀시대']),
]


# The two Sanullim tracks whose predictions sit ~20 years late, i.e. the likely subjects of
# the published caption's exclusion. Inferred from the offsets, not from any record.
SANULLIM_NON_ORIGINAL = [
    '{1986}_{그대 떠나는 날 비가 오는가}_{산울림}',
    '{1985}_{너의 의미}_{산울림}',
]


def track_of(seg_name):
    return re.sub(r'_\d+$', '', re.sub(r'\.pt$', '', str(seg_name)))


def artist_of(track):
    m = TRACK_RE.match(track)
    return m.group(3) if m else None


def chart_year(track):
    m = TRACK_RE.match(track)
    return int(m.group(1)) if m else None


def run_offsets(npz_path):
    """{track: offset} for one run on the Melon corpus."""
    z = np.load(npz_path, allow_pickle=True)
    files = [str(f) for f in z['files']]
    probs = z['probs'].astype(np.float64)
    idx = defaultdict(list)
    for i, f in enumerate(files):
        idx[track_of(f)].append(i)
    out = {}
    for t, ii in idx.items():
        ty = chart_year(t)
        if ty is None:
            continue
        p = probs[ii].mean(axis=0)
        out[t] = float(YEARS[int(p.argmax())] - ty)
    return out


def discover(bases, criterion):
    runs = []
    for base in bases:
        base = Path(base)
        if not base.is_dir():
            print(f'[warn] base not found: {base}', file=sys.stderr)
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or not RUN_DIR_RE.match(d.name):
                continue
            cfg = d / 'config.yaml'
            if cfg.exists():
                import yaml
                dt = yaml.safe_load(cfg.read_text()).get('train_set', {}).get('data_type')
                if dt != 'billboard':          # never mix the Melon-trained reverse models in
                    continue
            npz = d / f'{criterion}_kpop_segment_year_probs.npz'
            if npz.exists():
                runs.append(npz)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', nargs='+', default=['best_model'])
    ap.add_argument('--criterion', default='macro', choices=['macro', 'loss'])
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='track names to drop from the spread (see the docstring caveat)')
    ap.add_argument('--sd-ddof', type=int, default=0, help='0 = population SD, as in the paper')
    ap.add_argument('--format', default='both', choices=['markdown', 'latex', 'both'])
    args = ap.parse_args()

    npzs = discover(args.base, args.criterion)
    if not npzs:
        raise SystemExit('no Billboard-trained runs with a kpop npz found under --base')
    print(f'[runs] {len(npzs)}', file=sys.stderr)

    per_run = [run_offsets(p) for p in npzs]
    excluded = set(args.exclude)

    rows = []
    for label, period, names in ARTISTS:
        tracks = sorted({t for t in per_run[0] if artist_of(t) in names})
        if not tracks:
            print(f'[warn] no tracks for {label}', file=sys.stderr)
            continue
        medians = [np.median([r[t] for t in tracks if t in r]) for r in per_run]
        pooled = [r[t] for r in per_run for t in tracks if t in r and t not in excluded]
        lo, hi = np.percentile(pooled, [10, 90])
        rows.append((label, period, len(tracks),
                     float(np.mean(medians)), float(np.std(medians, ddof=args.sd_ddof)),
                     float(lo), float(hi)))

    if args.format in ('markdown', 'both'):
        print('\n| Artist | Period | n | Median (yr) | Spread |')
        print('|---|---|---|---|---|')
        for lab, per, n, m, sd, lo, hi in rows:
            print(f'| {lab} | {per.replace("--", "–")} | {n} | {m:+.1f}±{sd:.1f} | '
                  f'{lo:+.0f} to {hi:+.0f} |')

    if args.format in ('latex', 'both'):
        print('\n\\begin{tabular}{lcrcc}')
        print('\\toprule')
        print('Artist & Period & $n$ & Median (yr) & Spread \\\\')
        print('\\midrule')
        for lab, per, n, m, sd, lo, hi in rows:
            nn = f'\\phantom{{0}}{n}' if n < 10 else str(n)
            print(f'{lab} & {per} & {nn} & ${m:+.1f}{{\\pm}}{sd:.1f}$ & '
                  f'${lo:+.0f}$ to ${hi:+.0f}$ \\\\'.replace('+', ''))
        print('\\bottomrule')
        print('\\end{tabular}')


if __name__ == '__main__':
    main()
