"""Build seed-ensemble segment-probability files so the paper figures can be drawn
from all three seeds of each architecture instead of a single run.

Every run of a given domain shares the same segment file list in the same order
(verified below), so the ensemble is a plain average of the per-segment year
distributions -- the same soft-voting final_tables.py uses for its ensemble
columns. Output mimics a best_model/ directory so paper_figs.py can consume it:

    <out>/ENSEMBLE_<Arch>_s77/<criterion>_<domain>_segment_year_probs.npz

The _s77 suffix is only there to satisfy paper_figs.py's run glob; these files
are averages over seeds 77/78/79.

    python make_ensemble_runs.py --base best_model              # macro, all six archs
    python make_ensemble_runs.py --base best_model --melon      # Melon-trained reverse model

--base accepts several roots if the 18 runs are split across directories.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARCHS = ['Baseline', 'ShortChunkCNN', 'ShortChunkCNN_Res', 'FCN', 'CRNN', 'Musicnn']
DEFAULT_OUT = Path('figs_regen/_ensemble_runs')
# Directories holding the run dirs (set from --base). The 18 camera-ready runs may live
# under more than one root, so this is a list.
BASES = [Path('best_model')]


def trained_on(run_dir):
    """'billboard' or 'kpop' -- Melon-trained runs share the Baseline name pattern,
    so the training domain has to come from the config, not the directory name."""
    import yaml
    cfg = run_dir / 'config.yaml'
    if not cfg.exists():
        return None
    return yaml.safe_load(cfg.read_text()).get('train_set', {}).get('data_type')


def runs_for(arch, seeds, domain='billboard'):
    """All run dirs for one architecture trained on `domain`, keyed by seed."""
    found = {}
    for base in BASES:
        for d in sorted(base.glob(f'*_{arch}_s*')):
            seed = int(d.name.rsplit('_s', 1)[1])
            if seed in seeds and trained_on(d) == domain:
                found.setdefault(seed, d)
    return found


def average(paths):
    """Soft-vote: mean of row-normalized per-segment year distributions.

    This builds a NEW predictor that is better than any of its members, so its
    accuracy and its spread are not representative of the individual runs the
    tables describe. Prefer pool() for anything that sits next to a per-run
    statistic.
    """
    files_ref, acc = None, None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        files = [str(f) for f in z['files']]
        probs = z['probs'].astype(np.float64)
        probs /= probs.sum(axis=1, keepdims=True)
        if files_ref is None:
            files_ref, acc = files, probs
        else:
            if files != files_ref:
                raise SystemExit(f'segment file lists differ -- cannot ensemble:\n  {p}')
            acc += probs
    return files_ref, (acc / len(paths)).astype(np.float32)


def pool(paths):
    """Stack every run's per-segment distributions as separate rows.

    The result describes the distribution of predictions *across* runs, which is
    the same basis the tables report (mean over runs), and it carries run-to-run
    variation instead of averaging it away. Row count is len(paths) x N.
    """
    all_files, all_probs = [], []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        probs = z['probs'].astype(np.float64)
        probs /= probs.sum(axis=1, keepdims=True)
        all_files.extend(str(f) for f in z['files'])
        all_probs.append(probs.astype(np.float32))
    return all_files, np.concatenate(all_probs, axis=0)


def main():
    global BASES
    ap = argparse.ArgumentParser()
    ap.add_argument('--criterion', default='macro', choices=['macro', 'loss'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[77, 78, 79])
    ap.add_argument('--domains', nargs='+', default=['bill', 'kpop'])
    ap.add_argument('--base', nargs='+', default=[str(b) for b in BASES],
                    help='one or more directories containing the run dirs '
                         '(e.g. best_model/); accepts several roots')
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--melon', action='store_true',
                    help='ensemble the Melon-trained reverse models instead '
                         '(reads pop-era-dproto Melon runs, writes MELON_ENSEMBLE)')
    ap.add_argument('--melon-runs', nargs='*', default=None,
                    help='explicit Melon-trained run dirs (default: auto-detect)')
    ap.add_argument('--mode', default='pool', choices=['pool', 'average'],
                    help='pool = stack runs as extra rows (matches the per-run '
                         'statistics the tables report); average = soft-vote into '
                         'a single stronger predictor')
    args = ap.parse_args()
    BASES = [Path(b) for b in args.base]

    combine = pool if args.mode == 'pool' else average
    out = Path(args.out)
    seeds = set(args.seeds)

    if args.melon:
        if args.melon_runs:
            dirs = [Path(d) for d in args.melon_runs]
        else:
            # Melon-trained runs are Baseline runs whose config trains on kpop.
            import yaml
            dirs = []
            for base in BASES:
                for d in sorted(base.glob('*_Baseline_s*')):
                    cfg = d / 'config.yaml'
                    if not cfg.exists():
                        continue
                    c = yaml.safe_load(cfg.read_text())
                    if c.get('train_set', {}).get('data_type') == 'kpop':
                        dirs.append(d)
        if not dirs:
            raise SystemExit('no Melon-trained runs found')
        print(f'Melon-trained runs ({len(dirs)}):')
        for d in dirs:
            print('   ', d)
        dest = out / 'MELON_ENSEMBLE_s77'
        dest.mkdir(parents=True, exist_ok=True)
        for dom in ['bill']:
            paths = [d / f'{args.criterion}_{dom}_segment_year_probs.npz' for d in dirs]
            missing = [p for p in paths if not p.exists()]
            if missing:
                print(f'  [skip] {dom}: missing {len(missing)} file(s), '
                      f'e.g. {missing[0]}')
                continue
            files, probs = combine(paths)
            o = dest / f'{args.criterion}_{dom}_segment_year_probs.npz'
            np.savez_compressed(o, files=np.array(files), probs=probs)
            print(f'  wrote {o}  ({len(files)} rows, {len(paths)} runs, {args.mode})')
        return

    all_paths = defaultdict(list)
    for arch in ARCHS:
        found = runs_for(arch, seeds)
        if len(found) < len(seeds):
            print(f'[warn] {arch}: only seeds {sorted(found)} present, '
                  f'ensembling those')
        if not found:
            print(f'[skip] {arch}: no runs')
            continue
        dest = out / f'ENSEMBLE_{arch}_s77'
        dest.mkdir(parents=True, exist_ok=True)
        for dom in args.domains:
            paths = [found[s] / f'{args.criterion}_{dom}_segment_year_probs.npz'
                     for s in sorted(found)]
            missing = [p for p in paths if not p.exists()]
            if missing:
                print(f'  [skip] {arch}/{dom}: missing {len(missing)} file(s)')
                continue
            files, probs = combine(paths)
            o = dest / f'{args.criterion}_{dom}_segment_year_probs.npz'
            np.savez_compressed(o, files=np.array(files), probs=probs)
            print(f'{arch:20} {dom:5} <- {len(paths)} seeds {sorted(found)} '
                  f'-> {o.name} ({len(files)} rows)')
            all_paths[dom].extend(paths)


    # the all-architecture, all-seed group: the basis Table 3 reports
    dest = out / 'ENSEMBLE_ALL_s77'
    dest.mkdir(parents=True, exist_ok=True)
    for dom, paths in all_paths.items():
        files, probs = combine(paths)
        o = dest / f'{args.criterion}_{dom}_segment_year_probs.npz'
        np.savez_compressed(o, files=np.array(files), probs=probs)
        print(f'{"ALL (18 runs)":20} {dom:5} <- {len(paths)} runs '
              f'-> {o.name} ({len(files)} rows)')


if __name__ == '__main__':
    main()
