"""MATCH stage 3 — MFCC audio feature extractor.

Per-video MFCC features via librosa. Output:
    <dataset_root>/fea/fea_audio_mfcc.pt
as a `{video_id -> torch.Tensor(shape=[40])}` dict, matching the
upstream dataset loader contract at
`external_repos/match_hvd/src/model/MATCH/data/HateMM_MATCH.py:27`
and the per-item access pattern at line 49
(`mfcc_fea = self.mfcc_fea[vid]`).

Pipeline per video:
  1. `librosa.load(mp4_path, sr=16000)` — librosa uses soundfile +
     audioread to decode mp4 audio tracks
  2. `librosa.feature.mfcc(y, sr=16000, n_mfcc=40)` → shape `[40, T]`
  3. Mean-pool over time → `[40]` fixed-size vector
  4. Convert to `torch.Tensor(dtype=torch.float32)`
  5. Zero vector on any audio decode failure (silent track, corrupt
     file, etc.) — upstream doesn't document this edge case;
     `torch.zeros(40)` is the conservative default and keeps the
     collator's `torch.stack` from breaking.

Resume support: existing keys in `fea_audio_mfcc.pt` are preserved
and skipped. CPU-only.
"""

import argparse
import logging
import os
import sys
from typing import Dict

_OUR_METHOD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "our_method"
)
sys.path.insert(0, _OUR_METHOD)
from data_utils import (  # noqa: E402
    DATASET_ROOTS,
    SKIP_VIDEOS,
    get_media_path,
    load_annotations,
)

ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
SR = 16000
N_MFCC = 40


def fea_dir_for(dataset: str) -> str:
    return os.path.join(DATASET_ROOTS[dataset], "fea")


def output_path_for(dataset: str) -> str:
    return os.path.join(fea_dir_for(dataset), "fea_audio_mfcc.pt")


def extract_one(media_path: str, media_type: str):
    """Return a `torch.Tensor(shape=[40], dtype=float32)` MFCC vector.

    `media_type` is one of {"video", "frames"}. If the input is a
    pre-extracted frames directory (no audio), return zeros.
    """
    import torch
    import numpy as np

    if media_type == "frames":
        return torch.zeros(N_MFCC, dtype=torch.float32)
    try:
        import librosa
        y, sr = librosa.load(media_path, sr=SR, mono=True)
        if y is None or len(y) == 0:
            return torch.zeros(N_MFCC, dtype=torch.float32)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        # `mfcc` shape: (N_MFCC, T). Mean-pool over time.
        pooled = np.asarray(mfcc).mean(axis=1)
        return torch.from_numpy(pooled).float()
    except Exception:
        return torch.zeros(N_MFCC, dtype=torch.float32)


def process_dataset(dataset: str, split: str) -> Dict[str, "torch.Tensor"]:
    """Return `{vid -> torch.Tensor[40]}` for the split. Skips videos
    already present in the existing feature dict (resume).
    """
    import torch

    ann = load_annotations(dataset)
    skip_set = SKIP_VIDEOS.get(dataset, set())
    split_path = os.path.join(
        DATASET_ROOTS[dataset], "splits", f"{split}_clean.csv"
    )
    if not os.path.isfile(split_path):
        from data_utils import generate_clean_splits
        generate_clean_splits(dataset)
    with open(split_path) as f:
        vids = [line.strip() for line in f if line.strip()]

    out_path = output_path_for(dataset)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    existing: Dict[str, "torch.Tensor"] = {}
    if os.path.exists(out_path):
        try:
            existing = torch.load(out_path, weights_only=True)
        except Exception:
            existing = {}

    logging.info(
        f"[{dataset}/{split}] {len(vids)} ids, {len(existing)} already extracted"
    )
    n_new = 0
    n_fail = 0
    for vid in vids:
        if vid in existing:
            continue
        if vid in skip_set:
            existing[vid] = __import__("torch").zeros(N_MFCC, dtype=__import__("torch").float32)
            continue
        if vid not in ann:
            continue
        media = get_media_path(vid, dataset)
        if media is None:
            existing[vid] = __import__("torch").zeros(N_MFCC, dtype=__import__("torch").float32)
            n_fail += 1
            continue
        media_path, media_type = media
        vec = extract_one(media_path, media_type)
        if (vec == 0).all():
            n_fail += 1
        existing[vid] = vec
        n_new += 1
        if n_new % 50 == 0:
            torch.save(existing, out_path)
            logging.info(
                f"  [{dataset}/{split}] progress {n_new} new, {n_fail} zero"
            )
    torch.save(existing, out_path)
    logging.info(
        f"[{dataset}/{split}] done, {n_new} new extracted, {n_fail} zero, total {len(existing)}"
    )
    return existing


def main():
    parser = argparse.ArgumentParser(
        description="MATCH stage 3 MFCC extractor (librosa)"
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Both splits go to the same `fea_audio_mfcc.pt` dict",
    )
    args = parser.parse_args()
    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    datasets = ALL_DATASETS if args.all else [args.dataset]
    splits = ["train", "test"] if args.split == "both" else [args.split]
    for ds in datasets:
        for sp in splits:
            process_dataset(ds, sp)
    logging.info("All MFCC extraction done.")


if __name__ == "__main__":
    main()
