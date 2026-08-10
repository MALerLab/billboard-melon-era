"""Evaluate trained era classifiers the way the paper describes.

For every checkpoint under best_model/, this runs the Billboard test split and the Melon
corpus through the model, keeps the *year-level softmax of every 30-second segment*, and
averages those distributions within a track to get one predicted year per track. The old
pipeline stored a per-segment argmax and never aggregated, so decade metrics and era
offsets were segment statistics rather than the track statistics the paper reports.

Keeping the per-segment distributions also answers the meta-reviewer's request for the
consistency of predictions across crops of the same song: the per-track standard
deviation of segment-level predicted years falls out of the same pass.

Usage:
    python3 infer_all.py                    # every run dir, both selection criteria
    python3 infer_all.py --runs 0802_0026_Baseline --criteria macro
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_zoo

ROOT = Path(__file__).resolve().parent
YEAR_MIN, YEAR_MAX = 1958, 2024
DECADE_OF_YEAR = [min((y - 1960) // 10, 5) if y >= 1960 else 0
                  for y in range(YEAR_MIN, YEAR_MAX + 1)]

DOMAINS = {
    'bill': dict(slice_dir=ROOT / 'data/pt_files/mono_16000/test/slices_30_sec',
                 meta=ROOT / 'csv/billboard_hot100_chosen.csv', ext='.mp3'),
    'kpop': dict(slice_dir=ROOT / 'kpop_data/pt_files/mono_16000/test/slices_30_sec',
                 meta=ROOT / 'csv/korean_melon_meta_chosen.csv', ext='.wav'),
    # The artist-disjoint Melon test split (305 songs). This is the only Melon set a
    # Melon-trained model can be scored on honestly: the 'kpop' cache above covers the
    # whole corpus, which such a model has seen.
    'kpop_heldout': dict(slice_dir=ROOT / 'kpop_data/pt_files/mono_16000/test_org/slices_30_sec',
                         meta=ROOT / 'csv/korean_melon_meta_chosen.csv', ext='.wav'),
}


def track_of(slice_name):
    """'{1964}_{Song}_{Artist}_3.pt' -> '{1964}_{Song}_{Artist}'"""
    return re.sub(r'_\d+$', '', re.sub(r'\.pt$', '', slice_name))


def chart_year(track_name):
    """Chart-entry year is the leading {YYYY} of the cached filename."""
    m = re.match(r'\{(\d{4})\}', track_name)
    return int(m.group(1)) if m else None


def load_model(ckpt, cfg):
    model = getattr(model_zoo, cfg.model.cls)(**cfg.model.cfg)
    state = torch.load(ckpt, map_location='cpu')
    if all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to('cuda').eval()


def segment_year_probs(model, slice_dir, batch_size=64):
    """Year-level softmax for every cached segment. Returns (names, probs[N, 67])."""
    files = sorted(p.name for p in slice_dir.glob('*.pt'))
    probs = np.empty((len(files), YEAR_MAX - YEAR_MIN + 1), dtype=np.float32)
    with torch.inference_mode():
        for i in tqdm(range(0, len(files), batch_size), desc=slice_dir.parts[-3], leave=False):
            chunk = files[i:i + batch_size]
            auds = []
            for name in chunk:
                a = torch.load(slice_dir / name, map_location='cpu')
                auds.append(a.unsqueeze(0) if a.dim() == 1 else a)
            length = min(a.shape[-1] for a in auds)
            x = torch.stack([a[..., :length] for a in auds]).to('cuda')
            probs[i:i + len(chunk)] = model(x)[3].float().cpu().numpy()
    return files, probs


def aggregate_to_tracks(files, probs):
    """Average segment softmax within a track -> one predicted year per track.

    Also returns the spread of the per-segment argmax years, which is the crop-consistency
    statistic the meta-reviewer asked for.
    """
    by_track = defaultdict(list)
    for i, name in enumerate(files):
        by_track[track_of(name)].append(i)

    seg_year = YEAR_MIN + probs.argmax(axis=1)
    rows = []
    for track, idxs in by_track.items():
        true_year = chart_year(track)
        if true_year is None:
            continue
        mean_probs = probs[idxs].mean(axis=0)
        pred_year = YEAR_MIN + int(mean_probs.argmax())
        seg_years = seg_year[idxs]
        rows.append({
            'track': track,
            'true_year': true_year,
            'pred_year': pred_year,
            'offset': pred_year - true_year,
            'n_segments': len(idxs),
            'segment_pred_mean': float(seg_years.mean()),
            'segment_pred_std': float(seg_years.std(ddof=1)) if len(idxs) > 1 else 0.0,
            'segment_pred_min': int(seg_years.min()),
            'segment_pred_max': int(seg_years.max()),
        })
    return rows


def decade_metrics(rows):
    """Decade-level accuracy from the track-level predictions."""
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        if not (YEAR_MIN <= r['true_year'] <= YEAR_MAX):
            continue
        t = DECADE_OF_YEAR[r['true_year'] - YEAR_MIN]
        p = DECADE_OF_YEAR[min(max(r['pred_year'], YEAR_MIN), YEAR_MAX) - YEAR_MIN]
        per[t][1] += 1
        per[t][0] += int(t == p)
    out = {f'{1960 + 10 * d}s': round(c / n * 100, 2) for d, (c, n) in sorted(per.items()) if n}
    present = [v for v in out.values()]
    total_c = sum(c for c, _ in per.values())
    total_n = sum(n for _, n in per.values())
    out['macro'] = round(sum(present) / len(present), 2) if present else None
    out['micro'] = round(total_c / total_n * 100, 2) if total_n else None
    out['n_tracks'] = total_n
    return out


def crop_consistency(rows):
    multi = [r for r in rows if r['n_segments'] > 1]
    if not multi:
        return {}
    stds = np.array([r['segment_pred_std'] for r in multi])
    return {'n_tracks_multi_segment': len(multi),
            'mean_segment_std_years': round(float(stds.mean()), 3),
            'median_segment_std_years': round(float(np.median(stds)), 3),
            'mean_segment_sem_years': round(float((stds / np.sqrt(
                [r['n_segments'] for r in multi])).mean()), 3)}


def run_one(run_dir, criteria):
    sel_path = run_dir / 'best_selection.json'
    if not sel_path.exists():
        print(f'  skip {run_dir.name}: no best_selection.json')
        return
    selection = json.loads(sel_path.read_text())
    cfg = OmegaConf.load(run_dir / 'config.yaml')

    for criterion in criteria:
        if criterion not in selection:
            continue
        ckpt = run_dir / f"{criterion}_{selection[criterion]['iteration']}.pt"
        if not ckpt.exists():
            print(f'  skip {run_dir.name}/{criterion}: {ckpt.name} missing')
            continue
        model = load_model(ckpt, cfg)
        print(f'  {run_dir.name} [{criterion}] {ckpt.name}')

        summary = {'checkpoint': ckpt.name, 'selection': selection[criterion]}
        for domain, spec in DOMAINS.items():
            if not spec['slice_dir'].is_dir():
                continue
            files, probs = segment_year_probs(model, spec['slice_dir'])
            rows = aggregate_to_tracks(files, probs)

            out_csv = run_dir / f'{criterion}_{domain}_track_predictions.csv'
            with open(out_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            np.savez_compressed(run_dir / f'{criterion}_{domain}_segment_year_probs.npz',
                                files=np.array(files), probs=probs.astype(np.float16))

            summary[domain] = {'decade': decade_metrics(rows),
                               'crop_consistency': crop_consistency(rows),
                               'mean_offset_years': round(float(np.mean([r['offset'] for r in rows])), 3)}
            d = summary[domain]['decade']
            print(f'    {domain}: macro={d["macro"]} micro={d["micro"]} '
                  f'n={d["n_tracks"]} mean_offset={summary[domain]["mean_offset_years"]}')

        with open(run_dir / f'{criterion}_summary.json', 'w') as f:
            json.dump(summary, f, indent=1)
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None, help='run dir names under best_model/')
    ap.add_argument('--criteria', nargs='*', default=['macro', 'loss'])
    ap.add_argument('--weights', default='best_model')
    args = ap.parse_args()

    base = ROOT / args.weights
    dirs = sorted(d for d in base.iterdir() if d.is_dir())
    if args.runs:
        dirs = [d for d in dirs if d.name in args.runs]
    print(f'{len(dirs)} run dir(s)')
    for d in dirs:
        run_one(d, args.criteria)


if __name__ == '__main__':
    main()
