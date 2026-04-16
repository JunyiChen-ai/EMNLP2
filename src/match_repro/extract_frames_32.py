"""Pre-extract 32 uniformly-sampled frames per video (frames_32/).

Parallel to `extract_frames.py` (16 frames). The MATCH-HVD 2b.5
time-unit step (`time_unit_jina_clip.py`, ported from upstream
`preprocess/time_unit.py`) expects `frames_32/<vid>/frame_NNN.jpg`
with NNN in `000..031`, because upstream's CLIP matcher iterates 32
candidate frames per video.

Source preference order (same as `extract_frames.py`):
  1. If `<dataset_root>/frames/<vid>/` already exists with extracted
     jpgs, symlink / copy the 32 uniformly-spaced candidates into
     `frames_32/<vid>/`.
  2. Otherwise decode the mp4 via decord at 32 `np.linspace(0, n-1, 32)`
     indices.

CPU-only; uses `ProcessPoolExecutor` for parallelism. Resume-skip on
existing `frame_031.jpg`.

Run via director-submitted CPU sbatch before stage 2b.5 and stage 2c
(both consume `frames_32/`).
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
from data_utils import DATASET_ROOTS, get_media_path, load_annotations  # noqa: E402

NUM_FRAMES = 32
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]


def extract_one(vid, dataset, force=False):
    out_dir = os.path.join(DATASET_ROOTS[dataset], "frames_32", vid)
    last_frame = os.path.join(out_dir, f"frame_{NUM_FRAMES - 1:03d}.jpg")
    if os.path.exists(last_frame) and not force:
        return ("skip", vid)

    import glob as gm
    frames_dir = os.path.join(DATASET_ROOTS[dataset], "frames", vid)
    src_jpgs = sorted(gm.glob(os.path.join(frames_dir, "*.jpg")))
    if src_jpgs:
        media_path, media_type = frames_dir, "frames"
    else:
        media = get_media_path(vid, dataset)
        if media is None:
            return ("missing_media", vid)
        media_path, media_type = media

    os.makedirs(out_dir, exist_ok=True)

    if media_type == "frames":
        import glob as gm2
        src_jpgs = sorted(gm2.glob(os.path.join(media_path, "*.jpg")))
        if not src_jpgs:
            return ("no_src_frames", vid)
        if len(src_jpgs) <= NUM_FRAMES:
            indices = list(range(len(src_jpgs))) + [len(src_jpgs) - 1] * (
                NUM_FRAMES - len(src_jpgs)
            )
        else:
            import numpy as np
            indices = np.linspace(
                0, len(src_jpgs) - 1, NUM_FRAMES, dtype=int
            ).tolist()
        for i, idx in enumerate(indices):
            dst = os.path.join(out_dir, f"frame_{i:03d}.jpg")
            if not os.path.exists(dst):
                try:
                    os.symlink(src_jpgs[idx], dst)
                except OSError:
                    import shutil
                    shutil.copy(src_jpgs[idx], dst)
        return ("ok_symlink", vid)

    # mp4 path via decord
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(media_path)
        n = len(vr)
        if n == 0:
            return ("empty_video", vid)
        if n <= NUM_FRAMES:
            indices = list(range(n)) + [n - 1] * (NUM_FRAMES - n)
        else:
            import numpy as np
            indices = np.linspace(0, n - 1, NUM_FRAMES, dtype=int).tolist()
        from PIL import Image
        for i, idx in enumerate(indices):
            arr = vr[idx].asnumpy()
            Image.fromarray(arr).save(
                os.path.join(out_dir, f"frame_{i:03d}.jpg"), quality=85
            )
        return ("ok_mp4", vid)
    except Exception as e:
        return (f"err:{type(e).__name__}", vid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", default="all", choices=["train", "test", "all"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.dataset:
        parser.error("Provide --dataset or --all")

    datasets = ALL_DATASETS if args.all else [args.dataset]

    for ds in datasets:
        ann = load_annotations(ds)
        if args.split == "all":
            vids = list(ann.keys())
        else:
            split_path = os.path.join(
                DATASET_ROOTS[ds], "splits", f"{args.split}_clean.csv"
            )
            if not os.path.isfile(split_path):
                from data_utils import generate_clean_splits
                generate_clean_splits(ds)
            with open(split_path) as f:
                vids = [line.strip() for line in f if line.strip()]
        print(f"[{ds}] extracting {NUM_FRAMES} frames for {len(vids)} videos")

        from collections import Counter
        status = Counter()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(extract_one, v, ds, args.force) for v in vids
            ]
            done = 0
            for fut in as_completed(futures):
                s, vid = fut.result()
                status[s] += 1
                done += 1
                if done % 100 == 0:
                    print(f"  [{ds}] {done}/{len(vids)} {dict(status)}")
        print(f"  [{ds}] FINAL {dict(status)}")


if __name__ == "__main__":
    main()
