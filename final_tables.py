"""Camera-ready Table 2/3 generator — recomputed directly from the stored segment-year-probability npz files.

Reads the `{criterion}_{domain}_segment_year_probs.npz` files (+ track csv) left by
`infer_all.py` and emits the paper's Table 2 and Table 3 (plus supporting variants)
(2A/2B/2C, 3A/3B/3C/3D) as LaTeX and Markdown. Every number is recomputed from the
files, so the tables always agree with the artifacts they were built from.

Aggregation conventions
  - Default checkpoint criterion: macro (`--criterion`)
  - Track prediction: mean of segment softmax -> point estimate (`--estimator`, default argmax)
  - Decade accuracy: fold the predicted year into a decade and compare. macro = mean of per-era accuracies
  - median offset: per-era median of (pred_year - true_year)
  - on-era share: fraction of tracks with |offset| < thr (default 2.5)
  - mean±SD: mean over the 6 architecture values with **population SD (ddof=0)** — change with `--sd-ddof`

Ensembling (soft voting across seeds/architectures)
  npz files from the same domain share an identical segment file list, so the
  probabilities can simply be averaged.
    --ensemble arch   merge architectures (separately per seed)
    --ensemble seed   merge seeds (separately per architecture)
    --ensemble all    merge everything into one

Usage:
    python3 final_tables.py --base /path/to/best_model --tables 2A 3A
    python3 final_tables.py --base ... --tables all --out-dir tables_out
    python3 final_tables.py --base ... --ensemble arch --tables 3A
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
DECADES = ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s']
KPOP_DECADES = DECADES[:5]          # the Melon corpus has no 2010s

# Column order used in the paper's tables (same as the original Table 2)
ARCH_ORDER = ['CNN', 'FCN', 'SCNN', 'SCNNR', 'Musicnn', 'CRNN']
ARCH_LABEL = {
    'Baseline': 'CNN',
    'FCN': 'FCN',
    'ShortChunkCNN': 'SCNN',
    'ShortChunkCNN_Res': 'SCNNR',
    'Musicnn': 'Musicnn',
    'CRNN': 'CRNN',
}
RUN_DIR_RE = re.compile(r'^(\d{4})_(\d{4})_(.+)_s(\d+)$')


# --------------------------------------------------------------------------
# Run discovery
# --------------------------------------------------------------------------
class Run:
    """A single run directory under best_model."""

    def __init__(self, path):
        self.path = Path(path)
        m = RUN_DIR_RE.match(self.path.name)
        if not m:
            raise ValueError(f'cannot parse run directory name: {self.path.name}')
        self.date, self.time, self.arch_raw, seed = m.groups()
        self.seed = int(seed)
        self.arch = ARCH_LABEL.get(self.arch_raw, self.arch_raw)

    def npz(self, criterion, domain):
        return self.path / f'{criterion}_{domain}_segment_year_probs.npz'

    @property
    def trained_on(self):
        """'billboard' or 'kpop', read from the run's saved config.

        The directory name cannot tell them apart: the Melon-trained reverse models are
        also called '..._Baseline_s77'. Mixing them into Table 2/3 silently corrupts both.
        """
        cfg = self.path / 'config.yaml'
        if not cfg.exists():
            return None
        import yaml
        return yaml.safe_load(cfg.read_text()).get('train_set', {}).get('data_type')

    def summary(self, criterion):
        return self.path / f'{criterion}_summary.json'

    def __repr__(self):
        return f'<Run {self.arch} s{self.seed} {self.path.name}>'


def discover_runs(bases, runs=None, archs=None, seeds=None, criterion='macro', domain='kpop',
                  trained_on='billboard'):
    """Find the runs under --base and apply the (arch, seed, training-domain) filters."""
    found = []
    for base in bases:
        base = Path(base)
        if not base.is_dir():
            print(f'[warn] base not found: {base}', file=sys.stderr)
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or not RUN_DIR_RE.match(d.name):
                continue
            try:
                r = Run(d)
            except ValueError:
                continue
            if runs and d.name not in runs:
                continue
            if archs and r.arch not in archs:
                continue
            if seeds and r.seed not in seeds:
                continue
            if trained_on != 'any' and r.trained_on not in (None, trained_on):
                print(f'[skip] {d.name}: trained on {r.trained_on}, not {trained_on}',
                      file=sys.stderr)
                continue
            if not r.npz(criterion, domain).exists():
                print(f'[skip] {d.name}: no {criterion}_{domain} npz', file=sys.stderr)
                continue
            found.append(r)
    return found


# --------------------------------------------------------------------------
# Loading and point estimation
# --------------------------------------------------------------------------
def decade_of(year):
    if year < 1960:
        return 0
    return min((int(year) - 1960) // 10, 5)


def track_of(name):
    return re.sub(r'_\d+$', '', re.sub(r'\.pt$', '', name))


def chart_year(track):
    m = re.match(r'\{(\d{4})\}', track)
    return int(m.group(1)) if m else None


def load_probs(npz_path):
    """(files, probs) — probs is normalized so that each row sums to 1."""
    z = np.load(npz_path, allow_pickle=True)
    files = [str(f) for f in z['files']]
    probs = z['probs'].astype(np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return files, probs


def point_estimate(p, estimator):
    """Derive a single year from the averaged distribution p (1D, sums to 1)."""
    if estimator == 'argmax':
        return float(YEARS[int(p.argmax())])
    if estimator == 'expect':
        return float((YEARS * p).sum())
    if estimator == 'qmed':
        return float(YEARS[int(np.searchsorted(np.cumsum(p), 0.5))])
    raise ValueError(f'unknown estimator: {estimator}')


class Predictions:
    """Segment/track predictions for one model (or ensemble) on one domain."""

    def __init__(self, files, probs, estimator='argmax', keep_tracks=None):
        self.estimator = estimator
        idx = defaultdict(list)
        for i, f in enumerate(files):
            idx[track_of(f)].append(i)
        if keep_tracks is not None:
            idx = {t: ii for t, ii in idx.items() if t in keep_tracks}

        # Precompute the segment argmax years, then walk the tracks in a single pass.
        seg_year = YEARS[probs.argmax(axis=1)]
        tracks = []
        seg_pred_list, seg_true_list = [], []
        trk_true, trk_pred, trk_std, trk_nseg = [], [], [], []
        for t, ii in idx.items():
            ty = chart_year(t)
            if ty is None:
                continue
            tracks.append(t)
            sp = seg_year[ii]
            seg_pred_list.append(sp)
            seg_true_list.append(np.full(len(ii), ty, dtype=np.float64))

            p = probs[ii].mean(axis=0)
            p = p / p.sum()
            trk_true.append(ty)
            trk_pred.append(point_estimate(p, estimator))
            trk_nseg.append(len(ii))
            trk_std.append(float(np.std(sp, ddof=1)) if len(ii) > 1 else np.nan)

        self.tracks = tracks
        self.seg_pred = np.concatenate(seg_pred_list) if seg_pred_list else np.array([])
        self.seg_true = np.concatenate(seg_true_list) if seg_true_list else np.array([])
        self.trk_true = np.array(trk_true, dtype=np.float64)
        self.trk_pred = np.array(trk_pred, dtype=np.float64)
        self.trk_std = np.array(trk_std, dtype=np.float64)
        self.trk_nseg = np.array(trk_nseg, dtype=int)

    # -- accuracy ---------------------------------------------------------
    def decade_accuracy(self, level='track'):
        """Per-era accuracy (%) plus macro/micro. level: track | segment."""
        if level == 'track':
            true, pred = self.trk_true, self.trk_pred
        else:
            true, pred = self.seg_true, self.seg_pred
        td = np.array([decade_of(y) for y in true])
        pd_ = np.array([decade_of(round(min(max(y, YEAR_MIN), YEAR_MAX))) for y in pred])
        out = {}
        correct = total = 0
        for d in range(6):
            m = td == d
            if not m.any():
                continue
            c, n = int((pd_[m] == d).sum()), int(m.sum())
            out[DECADES[d]] = c / n * 100
            correct += c
            total += n
        out['macro'] = float(np.mean([out[k] for k in DECADES if k in out]))
        out['micro'] = correct / total * 100
        out['n'] = total
        return out

    # -- offset ------------------------------------------------------------
    def offset_stats(self, thresholds=(2.5,), level='track'):
        """Per-era median/mean offset, n, and on-era share (%) per threshold.

        level='track'   : mean of segment softmax -> point estimate (the paper's track level)
        level='segment' : aggregate the per-segment argmax values directly
        n always reports the track count (to match the Tracks row of the table).
        """
        if level == 'segment':
            off = self.seg_pred - self.seg_true
            td = np.array([decade_of(y) for y in self.seg_true])
            n_src = np.array([decade_of(y) for y in self.trk_true])
        else:
            off = self.trk_pred - self.trk_true
            td = np.array([decade_of(y) for y in self.trk_true])
            n_src = td
        out = {}
        for d in range(6):
            m = td == d
            if not m.any():
                continue
            o = off[m]
            out[DECADES[d]] = {
                'n': int((n_src == d).sum()),
                'n_units': int(m.sum()),
                'median': float(np.median(o)),
                'mean': float(o.mean()),
                'on_era': {f'{t}': float((np.abs(o) < t).mean() * 100) for t in thresholds},
            }
        return out

    def offsets_are_integral(self, level='track'):
        """Whether every offset is an integer — used to detect on-era threshold degeneracy."""
        off = (self.seg_pred - self.seg_true) if level == 'segment' \
            else (self.trk_pred - self.trk_true)
        return bool(np.all(off == np.round(off)))

    def crop_sd(self):
        """Mean within-track SD of the segment year predictions (multi-segment tracks only)."""
        v = self.trk_std[~np.isnan(self.trk_std)]
        return float(v.mean()) if len(v) else float('nan')


def build_predictions(runs, criterion, domain, estimator, keep_tracks=None):
    """Merge runs by soft voting (probability averaging) into a single Predictions."""
    files_ref, acc = None, None
    for r in runs:
        files, probs = load_probs(r.npz(criterion, domain))
        if files_ref is None:
            files_ref, acc = files, probs.copy()
        else:
            if files != files_ref:
                raise SystemExit(
                    f'segment file lists differ — cannot ensemble:\n'
                    f'  {runs[0].path.name} vs {r.path.name}')
            acc += probs
    acc /= len(runs)
    return Predictions(files_ref, acc, estimator, keep_tracks)


# --------------------------------------------------------------------------
# Building the columns (one column of the table)
# --------------------------------------------------------------------------
def make_columns(runs, criterion, estimator, ensemble, domains=('bill', 'kpop'),
                 keep_tracks=None):
    """Build the table's column list. Returns: [(label, {domain: Predictions}, [Run,...]), ...]"""
    if ensemble == 'none':
        groups = [(r.arch if len({x.seed for x in runs}) == 1 else f'{r.arch} s{r.seed}', [r])
                  for r in runs]
    elif ensemble == 'arch':
        by_seed = defaultdict(list)
        for r in runs:
            by_seed[r.seed].append(r)
        groups = [(f'ENS-arch s{s}', rs) for s, rs in sorted(by_seed.items())]
    elif ensemble == 'seed':
        by_arch = defaultdict(list)
        for r in runs:
            by_arch[r.arch].append(r)
        groups = [(a, rs) for a, rs in sorted(by_arch.items(),
                                              key=lambda kv: _arch_key(kv[0]))]
    elif ensemble == 'all':
        groups = [('ENS-all', list(runs))]
    else:
        raise ValueError(ensemble)

    cols, seen_labels = [], defaultdict(int)
    for label, rs in groups:
        seen_labels[label] += 1
        if seen_labels[label] > 1:          # keep colliding labels from silently overwriting in the JSON dump
            label = f'{label} #{seen_labels[label]}'
            print(f'[warn] duplicate column label; disambiguating as "{label}".', file=sys.stderr)
        preds = {}
        for dom in domains:
            rs_dom = [r for r in rs if r.npz(criterion, dom).exists()]
            if not rs_dom:
                print(f'[warn] {label}: no {dom} npz at all, leaving this column empty.',
                      file=sys.stderr)
                continue
            if len(rs_dom) != len(rs):
                missing = [r.path.name for r in rs if r not in rs_dom]
                print(f'[warn] {label}: the {dom} ensemble is built from only {len(rs_dom)}/{len(rs)} '
                      f'models (missing: {", ".join(missing)}) — composition differs across domains.',
                      file=sys.stderr)
            preds[dom] = build_predictions(rs_dom, criterion, dom, estimator,
                                          (keep_tracks or {}).get(dom))
        cols.append((label, preds, rs))

    if ensemble in ('none', 'seed'):
        cols.sort(key=lambda c: _arch_key(c[0]))
    return cols


def _arch_key(label):
    base = label.split(' s')[0]
    return (ARCH_ORDER.index(base) if base in ARCH_ORDER else 99, label)


ENS_PREFIX = 'ENS'


def is_ens(label):
    """Is this an ensemble column — it must be excluded from the mean±SD aggregate (avoids double counting)."""
    return label.startswith(ENS_PREFIX)


def add_ensemble_columns(cols, runs, criterion, estimator, axis, domains=('bill', 'kpop'),
                         keep_tracks=None):
    """Keep the existing columns and append the ensemble columns after them."""
    if axis == 'none':
        return cols
    if axis == 'all':
        groups = [(f'{ENS_PREFIX}-all ({len(runs)})', list(runs))]
    elif axis == 'seed':
        by = defaultdict(list)
        for r in runs:
            by[r.arch].append(r)
        groups = [(f'{ENS_PREFIX}-seed:{a}', rs)
                  for a, rs in sorted(by.items(), key=lambda kv: _arch_key(kv[0]))]
    elif axis == 'arch':
        by = defaultdict(list)
        for r in runs:
            by[r.seed].append(r)
        groups = [(f'{ENS_PREFIX}-arch:s{s}', rs) for s, rs in sorted(by.items())]
    else:
        raise ValueError(axis)

    extra = []
    for label, rs in groups:
        preds = {}
        for dom in domains:
            rs_dom = [r for r in rs if r.npz(criterion, dom).exists()]
            if rs_dom:
                preds[dom] = build_predictions(rs_dom, criterion, dom, estimator,
                                              (keep_tracks or {}).get(dom))
        extra.append((label, preds, rs))
    return list(cols) + extra


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def agg(values, ddof=0):
    """(mean, sd) — population SD by default. With too few samples, sd is NaN (i.e. not printed)."""
    v = np.array([x for x in values if x is not None and not np.isnan(x)], dtype=np.float64)
    if not len(v):
        return float('nan'), float('nan')
    if len(v) < 2 or len(v) - ddof <= 0:
        # With a single column there is no spread to measure. Do not disguise it as 0.0.
        return float(v.mean()), float('nan')
    return float(v.mean()), float(np.std(v, ddof=ddof))


def pm(mean, sd, prec=1):
    """The mean±sd string. If sd is NaN, the ± term is omitted."""
    if sd is None or np.isnan(sd):
        return f'{mean:.{prec}f}'
    return f'{mean:.{prec}f}±{sd:.{prec}f}'


def pm_tex(mean, sd, prec=1):
    if sd is None or np.isnan(sd):
        return f'${mean:.{prec}f}$'
    return f'${mean:.{prec}f}{{\\pm}}{sd:.{prec}f}$'


def axis_desc(ensemble, n_cols, n_runs, n_seeds=1, tex=True):
    """What the table's columns range over — makes the caption state the actual aggregation axis."""
    pms = r'mean $\pm$ SD' if tex else 'mean ± SD'
    if n_cols < 2:
        if ensemble in ('arch', 'all'):
            return f'a single soft-voting ensemble of {n_runs} models'
        return 'a single model'
    if ensemble == 'arch':
        return f'{pms} across {n_cols} seeds (each an architecture ensemble)'
    if ensemble == 'seed':
        return f'{pms} across {n_cols} architectures (each a seed ensemble)'
    if n_seeds > 1:      # the columns are individual architecture x seed runs
        return f'{pms} across {n_cols} architecture--seed runs'
    return f'{pms} across {n_cols} architectures'


def md_table(header, rows, align=None):
    align = align or ['---'] * len(header)
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(align) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def tex_table(colspec, header_lines, body_rows, mid_before=None, caption=None):
    """A simple booktabs table. Each entry of body_rows is a list of cells."""
    lines = [f'\\begin{{tabular}}{{{colspec}}}', '  \\toprule']
    for h in header_lines:
        lines.append('  ' + ' & '.join(h) + r' \\')
    lines.append('  \\midrule')
    for i, r in enumerate(body_rows):
        if mid_before and i in mid_before:
            lines.append('  \\midrule')
        lines.append('  ' + ' & '.join(str(x) for x in r) + r' \\')
    lines.append('  \\bottomrule')
    lines.append('\\end{tabular}')
    if caption:
        lines.append('% caption: ' + caption)
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Table 2 — Billboard accuracy
# --------------------------------------------------------------------------
def require_domain(cols, domain):
    """If any column lacks predictions for the domain, report it clearly and stop."""
    bad = [lab for lab, preds, _ in cols if domain not in preds]
    if bad:
        raise SystemExit(
            f'[error] Table 2 needs {domain} inference results, but these columns have none: '
            + ', '.join(bad)
            + f'\n        Check that each run directory contains <criterion>_{domain}_segment_year_probs.npz, '
              'or rerun with 2A/2B/2C dropped from --tables.')


def table2(cols, level='track', fmt='both'):
    """2A (track) / 2B (segment): era x model accuracy."""
    require_domain(cols, 'bill')
    labels = [c[0] for c in cols]
    accs = [c[1]['bill'].decade_accuracy(level=level) for c in cols]
    short = {'1960s': '60s', '1970s': '70s', '1980s': '80s',
             '1990s': '90s', '2000s': '00s', '2010s': '10s'}

    rows = []
    for d in DECADES:
        rows.append([short[d]] + [f'{a[d]:.1f}' if d in a else '--' for a in accs])
    macro = ['Macro'] + [f'{a["macro"]:.1f}' for a in accs]
    micro = ['Micro'] + [f'{a["micro"]:.1f}' for a in accs]

    out = {}
    if fmt in ('markdown', 'both'):
        md_rows = list(rows)
        md_rows.append(['**Macro**'] + [f'**{a["macro"]:.1f}**' for a in accs])
        md_rows.append(['**Micro**'] + [f'**{a["micro"]:.1f}**' for a in accs])
        out['markdown'] = md_table(['Decade'] + labels, md_rows)
    if fmt in ('latex', 'both'):
        lvl = 'track-level predictions; segment softmax averaged per track' \
            if level == 'track' else 'segment-level predictions'
        out['latex'] = tex_table(
            '@{}l*{%d}{c}@{}' % len(labels),
            [['Decade'] + labels],
            rows + [macro, micro],
            mid_before={len(rows)},
            caption=('Model-wise era classification accuracy (%) on the Billboard test '
                     f'set at the decade level ({lvl}).'))
    out['data'] = {lab: a for lab, a in zip(labels, accs)}
    return out


def table2c(cols, fmt='both'):
    """2C: Billboard/Melon macro·micro + crop SD."""
    require_domain(cols, 'bill')
    labels = [c[0] for c in cols]
    rows_md, rows_tex, data = [], [], {}
    for lab, preds, _ in cols:
        b = preds['bill'].decade_accuracy('track')
        k = preds['kpop'].decade_accuracy('track') if 'kpop' in preds else None
        sd = preds['bill'].crop_sd()
        vals = [f'{b["macro"]:.1f}', f'{b["micro"]:.1f}',
                f'{k["macro"]:.1f}' if k else '--',
                f'{k["micro"]:.1f}' if k else '--',
                f'{sd:.2f}']
        rows_md.append([lab] + vals)
        rows_tex.append([lab] + vals)
        data[lab] = {'bill_macro': b['macro'], 'bill_micro': b['micro'],
                     'kpop_macro': k['macro'] if k else None,
                     'kpop_micro': k['micro'] if k else None, 'crop_sd': sd}

    out = {'data': data}
    if fmt in ('markdown', 'both'):
        out['markdown'] = md_table(
            ['Model', 'Billboard Macro', 'Billboard Micro',
             'Melon Macro', 'Melon Micro', 'Crop SD (yr)'], rows_md)
    if fmt in ('latex', 'both'):
        out['latex'] = tex_table(
            '@{}lccccc@{}',
            [['', r'\multicolumn{2}{c}{Billboard test}',
              r'\multicolumn{2}{c}{Melon (cross)}', 'Crop'],
             ['Model', 'Macro', 'Micro', 'Macro', 'Micro', 'SD (yr)']],
            rows_tex,
            caption=('Decade-level accuracy (%, track-level) on the in-domain Billboard '
                     'test set and the cross-domain Melon corpus, with within-track '
                     'prediction spread (SD of segment-level year predictions).'))
    return out


# --------------------------------------------------------------------------
# Table 3 — Melon era offset
# --------------------------------------------------------------------------
def collect_offsets(cols, thresholds, ddof=0, level='track', domain='kpop'):
    """Collect the per-column era statistics plus the across-column mean±SD.

    Ensemble columns (ENS-*) are kept in per_col but dropped from the mean±SD
    aggregate — their member runs already appear as individual columns, so the
    same data would be counted twice.
    """
    per_col = {}
    integral = True
    for lab, preds, _ in cols:
        if domain not in preds:
            continue
        per_col[lab] = preds[domain].offset_stats(thresholds, level)
        integral &= preds[domain].offsets_are_integral(level)

    # If the offsets are integral, thresholds that cut the same integer set (e.g. <2.5 and <3.0) are indistinguishable.
    if integral:
        buckets = defaultdict(list)
        for t in thresholds:
            buckets[int(np.ceil(t)) if t != int(t) else int(t)].append(t)
        dup = [v for v in buckets.values() if len(v) > 1]
        if dup:
            print('[warn] the offsets are integral, so these thresholds give identical results: '
                  + '; '.join('/'.join(str(x) for x in v) for v in dup)
                  + ' — the on-era threshold robustness check (3C) is meaningless in this setup.',
                  file=sys.stderr)

    base = {k: v for k, v in per_col.items() if not is_ens(k)}   # aggregate excluding ensembles
    summary = {}
    for d in DECADES:
        if not any(d in s for s in per_col.values()):
            continue
        meds = [s[d]['median'] for s in base.values() if d in s]
        summary[d] = {
            'n': next((s[d]['n'] for s in per_col.values() if d in s), 0),
            'n_units': next((s[d]['n_units'] for s in per_col.values() if d in s), 0),
            'median_mean': agg(meds, ddof)[0],
            'median_sd': agg(meds, ddof)[1],
            'on_era': {},
        }
        for t in thresholds:
            vals = [s[d]['on_era'][f'{t}'] for s in base.values() if d in s]
            m, sd = agg(vals, ddof)
            summary[d]['on_era'][f'{t}'] = {'mean': m, 'sd': sd}
    return per_col, summary


def decs_of(summary):
    """The decades that actually appear in the table (2010s is present or not depending on the domain)."""
    return [d for d in DECADES if d in summary]


def table3a(summary, thr=2.5, fmt='both', level='track', axis='across architectures'):
    """3A: n / median±SD / on-era±SD."""
    unit = 'tracks' if level == 'track' else 'segments'
    n_key = 'n' if level == 'track' else 'n_units'
    n_lab_md = 'Tracks (n)' if level == 'track' else 'Segments (n)'
    n_lab_tex = r'Tracks ($n$)' if level == 'track' else r'Segments ($n$)'

    DECS = decs_of(summary)
    n_row = [f'{summary[d][n_key]}' for d in DECS]
    med_md = [pm(summary[d]['median_mean'], summary[d]['median_sd']) for d in DECS]
    on_md = [pm(summary[d]['on_era'][f'{thr}']['mean'],
                summary[d]['on_era'][f'{thr}']['sd']) for d in DECS]
    med_tex = [pm_tex(summary[d]['median_mean'], summary[d]['median_sd'])
               for d in DECS]
    on_tex = [pm_tex(summary[d]['on_era'][f'{thr}']['mean'],
                     summary[d]['on_era'][f'{thr}']['sd']) for d in DECS]

    out = {}
    if fmt in ('markdown', 'both'):
        out['markdown'] = md_table(
            [''] + DECS,
            [[n_lab_md] + n_row,
             ['**Median offset (yr)**'] + med_md,
             [f'**On-era share (%, \\|Δ\\|<{thr})**'] + on_md])
    if fmt in ('latex', 'both'):
        out['latex'] = tex_table(
            '@{}l' + 'c' * len(DECS) + '@{}', [[''] + DECS],
            [[n_lab_tex] + n_row,
             ['Median offset (yr)'] + med_tex,
             [r'On-era share (\%)'] + on_tex],
            caption=('Cross-domain era statistics per decade for Melon songs under '
                     f'Billboard-trained models ({axis}). '
                     f'Median offset is the per-decade median of {level}-level year '
                     f'offsets; on-era share is the fraction of {unit} dated within {thr} '
                     'years of their chart-entry year.'))
    return out


def table3b(per_col, summary, thr=2.5, fmt='both'):
    """3B: per-run medians laid out, plus Mean and On-era rows. Ensemble rows are appended separately after Mean."""
    DECS = decs_of(summary)
    labels = [l for l in per_col if not is_ens(l)]
    ens_labels = [l for l in per_col if is_ens(l)]
    rows = []
    for lab in labels:
        rows.append([lab] + [f'{per_col[lab][d]["median"]:.2f}' if d in per_col[lab]
                             else '--' for d in DECS])
    mean_row = ['Mean'] + [f'{summary[d]["median_mean"]:.2f}' for d in DECS]
    on_row = ['On-era share (%)'] + [f'{summary[d]["on_era"][f"{thr}"]["mean"]:.1f}'
                                     for d in DECS]
    ens_rows = [[l] + [f'{per_col[l][d]["median"]:.2f}' if d in per_col[l] else '--'
                       for d in DECS] for l in ens_labels]
    ens_on = [[f'{l} on-era (%)'] + [f'{per_col[l][d]["on_era"][f"{thr}"]:.1f}'
                                     if d in per_col[l] else '--' for d in DECS]
              for l in ens_labels]
    out = {}
    if fmt in ('markdown', 'both'):
        out['markdown'] = md_table(
            ['Median offset (yr)'] + DECS,
            rows + [['**Mean**'] + mean_row[1:], on_row] + ens_rows + ens_on)
    if fmt in ('latex', 'both'):
        tex_rows = [[r[0]] + [f'${v}$' for v in r[1:]] for r in rows]
        tex_rows.append(['Mean'] + [f'${v}$' for v in mean_row[1:]])
        tex_rows.append([r'On-era share (\%)'] + on_row[1:])
        for r in ens_rows:
            tex_rows.append([r[0]] + [f'${v}$' for v in r[1:]])
        for r in ens_on:
            tex_rows.append(r)
        out['latex'] = tex_table('@{}l' + 'c' * len(DECS) + '@{}',
                                 [['Median offset (yr)'] + DECS],
                                 tex_rows,
                                 mid_before={len(rows), len(rows) + 2} if ens_rows
                                 else {len(rows)})
    return out


def table3c(summary, thresholds, fmt='both'):
    """3C: on-era threshold robustness."""
    DECS = decs_of(summary)
    med = [f'{summary[d]["median_mean"]:.1f}' for d in DECS]
    rows = [['Median offset (yr)'] + med]
    for t in thresholds:
        rows.append([f'On-era |Δ|<{t} (%)'] +
                    [f'{summary[d]["on_era"][f"{t}"]["mean"]:.1f}' for d in DECS])
    out = {}
    if fmt in ('markdown', 'both'):
        md_rows = [rows[0]] + [[r[0].replace('|', '\\|')] + r[1:] for r in rows[1:]]
        out['markdown'] = md_table([''] + DECS, md_rows)
    if fmt in ('latex', 'both'):
        tex_rows = [['Median offset (yr)'] + [f'${v}$' for v in med]]
        for t in thresholds:
            tex_rows.append([f'On-era $|\\Delta|{{<}}{t}$ (\\%)'] +
                            [f'{summary[d]["on_era"][f"{t}"]["mean"]:.1f}'
                             for d in DECS])
        out['latex'] = tex_table('@{}l' + 'c' * len(DECS) + '@{}',
                                 [[''] + DECS], tex_rows)
    return out


def table3d(summary, fmt='both', axis='across architectures'):
    """3D: the medians on a single row."""
    DECS = decs_of(summary)
    med = [f'{summary[d]["median_mean"]:.1f}' for d in DECS]
    # 3D carries no ± term, so rewrite the caption's "mean ± SD across ..." as "averaged across ...".
    axis_avg = axis.replace(r'mean $\pm$ SD across', 'averaged across')
    out = {}
    if fmt in ('markdown', 'both'):
        out['markdown'] = md_table([''] + DECS, [['Era offset (median)'] + med])
    if fmt in ('latex', 'both'):
        out['latex'] = tex_table(
            '@{}l' + 'c' * len(DECS) + '@{}', [[''] + DECS],
            [['Era offset (median)'] + [f'${v}$' for v in med]],
            caption=('Median era offset (years) per decade for Melon songs, '
                     f'{axis_avg} (Billboard-trained).'))
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
ALL_TABLES = ['2A', '2B', '2C', '3A', '3B', '3C', '3D']


def main():
    ap = argparse.ArgumentParser(
        description='Camera-ready Table 2/3 generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--base', nargs='+', default=['best_model'],
                    help='path to the best_model directory holding the run directories (several allowed)')
    ap.add_argument('--runs', nargs='*', default=None, help='use only these run directory names')
    ap.add_argument('--archs', nargs='*', default=None,
                    help=f'architecture filter (labels: {" ".join(ARCH_ORDER)})')
    ap.add_argument('--eval-manifest', default=None,
                    help='JSON with a "test" list restricting which Billboard tracks are '
                         'scored. csv/billboard_eval_asrun_2437.json reproduces the published '
                         'numbers exactly; csv/billboard_eval_test_2421.json is the clean '
                         'artist-disjoint test split (see README, "Billboard evaluation set")')
    ap.add_argument('--trained-on', default='billboard', choices=['billboard', 'kpop', 'any'],
                    help="keep only runs trained on this domain; the paper's Tables 2 and 3 "
                         'are Billboard-trained. Melon-trained reverse models can share a '
                         'best_model directory and must not be mixed in')
    ap.add_argument('--seeds', nargs='*', type=int, default=None, help='seed filter')
    ap.add_argument('--criterion', default='macro', choices=['macro', 'loss'],
                    help='checkpoint selection criterion')
    ap.add_argument('--estimator', default='argmax', choices=['argmax', 'expect', 'qmed'],
                    help='point estimator for the track year')
    ap.add_argument('--ensemble', default='none', choices=['none', 'arch', 'seed', 'all'],
                    help='soft-voting ensemble axis')
    ap.add_argument('--tables', nargs='+', default=['all'],
                    help=f'tables to generate: {" ".join(ALL_TABLES)} or all')
    ap.add_argument('--thresholds', nargs='+', type=float, default=[2.0, 2.5, 3.0],
                    help='on-era thresholds (2.5 is the default reported one)')
    ap.add_argument('--on-era-thr', type=float, default=2.5, help='threshold reported in 3A/3B')
    ap.add_argument('--sd-ddof', type=int, default=0, help='ddof for mean±SD (0 = population)')
    ap.add_argument('--offset-level', default='track', choices=['track', 'segment'],
                    help='aggregation unit for the Table 3 offsets')
    ap.add_argument('--offset-domain', default='kpop', choices=['kpop', 'bill', 'kpop_heldout'],
                    help='target domain for Table 3 (bill = in-domain control)')
    ap.add_argument('--add-ensemble', default='none',
                    choices=['none', 'arch', 'seed', 'all'],
                    help='keep the individual columns and append ensemble columns (excluded from the mean±SD aggregate)')
    ap.add_argument('--format', default='both', choices=['latex', 'markdown', 'both'])
    ap.add_argument('--out-dir', default=None, help='directory to write the results to')
    args = ap.parse_args()

    if 'all' in [t.lower() for t in args.tables]:
        tables = list(ALL_TABLES)
    else:
        canon = {t.upper(): t for t in ALL_TABLES}
        unknown = [t for t in args.tables if t.upper() not in canon]
        if unknown:
            raise SystemExit(f'[error] unknown --tables value: {", ".join(unknown)}\n'
                             f'        available: {" ".join(ALL_TABLES)} or all')
        tables = [canon[t.upper()] for t in args.tables]
    thresholds = sorted(set(args.thresholds) | {args.on_era_thr})

    runs = discover_runs(args.base, args.runs, args.archs, args.seeds,
                         args.criterion, 'kpop', args.trained_on)
    if not runs:
        raise SystemExit('no runs match the given filters.')
    print(f'[runs] {len(runs)}: ' + ', '.join(f'{r.arch}(s{r.seed})' for r in runs),
          file=sys.stderr)

    keep_tracks = None
    if args.eval_manifest:
        manifest = json.loads(Path(args.eval_manifest).read_text(encoding='utf-8'))
        keep_tracks = {'bill': set(manifest['test'])}
        print(f'[eval] Billboard restricted to {len(keep_tracks["bill"])} tracks '
              f'from {args.eval_manifest}', file=sys.stderr)

    cols = make_columns(runs, args.criterion, args.estimator, args.ensemble,
                        keep_tracks=keep_tracks)
    cols = add_ensemble_columns(cols, runs, args.criterion, args.estimator,
                                args.add_ensemble, keep_tracks=keep_tracks)
    print(f'[cols] {len(cols)} columns: ' + ', '.join(c[0] for c in cols), file=sys.stderr)

    per_col, summary = collect_offsets(cols, thresholds, args.sd_ddof,
                                       args.offset_level, args.offset_domain)

    results, data_dump = {}, {'config': vars(args),
                              'runs': [r.path.name for r in runs],
                              'columns': [c[0] for c in cols]}
    t2a = t2b = t2c = None
    if '2A' in tables:
        t2a = table2(cols, 'track', args.format)
        results['2A'] = t2a
    if '2B' in tables:
        t2b = table2(cols, 'segment', args.format)
        results['2B'] = t2b
    if '2C' in tables:
        t2c = table2c(cols, args.format)
        results['2C'] = t2c
    n_seeds = len({r.seed for r in runs})
    n_base = sum(1 for c in cols if not is_ens(c[0]))
    axis_tex = axis_desc(args.ensemble, n_base, len(runs), n_seeds, tex=True)
    if '3A' in tables:
        results['3A'] = table3a(summary, args.on_era_thr, args.format,
                                args.offset_level, axis_tex)
    if '3B' in tables:
        results['3B'] = table3b(per_col, summary, args.on_era_thr, args.format)
    if '3C' in tables:
        results['3C'] = table3c(summary, thresholds, args.format)
    if '3D' in tables:
        results['3D'] = table3d(summary, args.format, axis_tex)

    data_dump['table2'] = {k: results[k]['data'] for k in ('2A', '2B', '2C')
                           if k in results and 'data' in results[k]}
    data_dump['table3'] = {'per_column': per_col, 'summary': summary}

    # Output
    chunks = []
    head = (f'# Table 2·3 — criterion={args.criterion}, estimator={args.estimator}, '
            f'ensemble={args.ensemble}, offset-level={args.offset_level}, '
            f'SD ddof={args.sd_ddof}\n'
            f'\nAggregation axis: {axis_desc(args.ensemble, n_base, len(runs), n_seeds, tex=False)} '
            f'({len(runs)} runs -> {n_base} columns'
            + (f' + {len(cols) - n_base} ensemble column(s), excluded from the aggregate' if len(cols) > n_base else '')
            + f'), Table 3 domain={args.offset_domain}')
    chunks.append(head)
    for key in ALL_TABLES:
        if key not in tables or key not in results:
            continue
        chunks.append(f'\n## Table {key}\n')
        r = results[key]
        if 'markdown' in r:
            chunks.append(r['markdown'])
        if 'latex' in r:
            chunks.append('\n```latex\n' + r['latex'] + '\n```')
    body = '\n'.join(chunks)
    print(body)

    # The cross-check is an extra. Even if it fails here, the table outputs must already be saved.
    out = tag = None
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = f'{args.criterion}-{args.estimator}' + (
            f'-ens{args.ensemble}' if args.ensemble != 'none' else '') + (
            f'-add{args.add_ensemble}' if args.add_ensemble != 'none' else '') + (
            '-seg' if args.offset_level == 'segment' else '') + (
            f'-{args.offset_domain}' if args.offset_domain != 'kpop' else '')


    if out is not None:
        (out / f'tables-{tag}.md').write_text(body + '\n', encoding='utf-8')
        (out / f'tables-{tag}.json').write_text(
            json.dumps(data_dump, ensure_ascii=False, indent=2, default=float),
            encoding='utf-8')
        for key, r in results.items():
            if 'latex' in r:
                (out / f'table{key}-{tag}.tex').write_text(r['latex'] + '\n',
                                                          encoding='utf-8')
        print(f'\n[saved] {out}/', file=sys.stderr)


if __name__ == '__main__':
    main()
