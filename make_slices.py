#!/usr/bin/env python3
"""Build the non-overlapping 30-second segment caches that `infer_all.py` scores.

Inference never touches source audio: it reads pre-cut fp16 tensors from

    <audio_root>/pt_files/<channel>_<sr>/<out_subdir>/slices_30_sec/{track}_{k}.pt

where `{track}` is `{YYYY}_{Song}_{Artist}` with literal braces, exactly as it appears in
the split/manifest JSON. A track's audio is centre-trimmed to a whole number of 30 s
segments and every segment is written out; tracks shorter than 30 s produce no segments.

The segmentation here is the same procedure as
`dataset.BillboardDatasetHierarchyValidTest.save_all_30_sec_pt_files`, lifted out so the
caches can be built for an arbitrary manifest and output directory. The three caches the
paper uses:

    # Billboard test set (as evaluated in the paper, 2,437 tracks)
    python make_slices.py --audio-root data --ext .mp3 \
        --manifest csv/billboard_eval_asrun_2437.json --out-subdir test

    # Melon corpus, all 2,771 tracks (cross-domain target)
    python make_slices.py --audio-root kpop_data --ext .wav \
        --manifest csv/melon_eval_all_2771.json --out-subdir test

    # Melon artist-disjoint held-out split, 305 tracks (scores the reverse model honestly)
    python make_slices.py --audio-root kpop_data --ext .wav \
        --manifest csv/melon_eval_heldout_305.json --out-subdir test_org

Existing .pt files are skipped, so the command is resumable.
"""
import argparse
import json
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


def segment_track(audio, sr, clip_len):
    """Centre-trim to a whole number of clips, then split. Mirrors dataset.py."""
    length = audio.shape[-1]
    sample_length = sr * clip_len
    n_full = length // sample_length
    if n_full == 0:
        return []
    remaining = length - n_full * sample_length
    trim_start = remaining // 2
    trim_end = remaining - trim_start
    audio = audio[:, trim_start:length - trim_end]
    return [audio[:, i * sample_length:(i + 1) * sample_length].to(dtype=torch.float16)
            for i in range(n_full)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audio-root', required=True,
                    help='directory holding the source audio files (e.g. data, kpop_data)')
    ap.add_argument('--manifest', required=True,
                    help='JSON mapping a key to a list of {YYYY}_{Song}_{Artist} track names')
    ap.add_argument('--key', default='test', help='which list in the manifest to use')
    ap.add_argument('--out-subdir', default='test',
                    help="cache subdirectory: 'test' or 'test_org'")
    ap.add_argument('--ext', default='.mp3', help='source audio extension (.mp3 or .wav)')
    ap.add_argument('--sr', type=int, default=16000)
    ap.add_argument('--channel', default='mono', choices=['mono', 'stereo'])
    ap.add_argument('--clip-len', type=int, default=30, help='segment length in seconds')
    args = ap.parse_args()

    root = Path(args.audio_root)
    tracks = json.loads(Path(args.manifest).read_text(encoding='utf-8'))[args.key]
    out_dir = root / f'pt_files/{args.channel}_{args.sr}/{args.out_subdir}/slices_30_sec'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'{len(tracks)} tracks -> {out_dir}')

    written = skipped = missing = 0
    for name in tqdm(tracks):
        src = root / f'{name}{args.ext}'
        # dataset.py keys the cache on Path.stem, i.e. the track name without the audio
        # extension: '{1958}_{A Letter To An Angel}_{Jimmy Clanton}_0.pt'.
        stem = src.stem
        if (out_dir / f'{stem}_0.pt').exists():
            skipped += 1
            continue
        if not src.exists():
            missing += 1
            continue
        audio, org_sr = torchaudio.load(src)
        if args.sr != org_sr:
            audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=args.sr)
        if args.channel == 'mono':
            audio = audio.mean(dim=0).unsqueeze(0)
        for k, seg in enumerate(segment_track(audio, args.sr, args.clip_len)):
            torch.save(seg, out_dir / f'{stem}_{k}.pt')
        written += 1

    print(f'wrote {written}, skipped {skipped} already cached, {missing} source files missing')
    if missing:
        print('  (missing audio is expected if you have not obtained every recording; '
              'see README "Data layout")')


if __name__ == '__main__':
    main()
