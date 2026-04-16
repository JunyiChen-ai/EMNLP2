"""MATCH stage 3 — ViViT 32-frame video feature extractor.

Backbone: `google/vivit-b-16x2-kinetics400` via HF transformers
`VivitModel` + `VivitImageProcessor`. Matches upstream dataset
loader contract at `HateMM_MATCH.py:28`:
    self.vivit_fea = torch.load('./data/HateMM/fea/fea_frames_32_google-vivit-b.pt', weights_only=True)
with per-item access at line 50 (`vivit_fea = self.vivit_fea[vid]`).

Output: `<dataset_root>/fea/fea_frames_32_google-vivit-b.pt` as
`{video_id -> torch.Tensor(shape=[768])}` — ViViT-base hidden_size
is 768, we use the `[CLS]` token embedding from
`outputs.last_hidden_state[:, 0]`.

Frame source: `<dataset_root>/frames_32/<vid>/frame_NNN.jpg`, already
extracted by `src/match_repro/extract_frames_32.py`.

Resume support: existing keys in the output dict are preserved.
Requires CUDA (ViViT-b at ~90M params, bf16 fine on any A100).
"""

import argparse
import glob
import logging
import os
import sys
from typing import Dict

_OUR_METHOD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "our_method"
)
sys.path.insert(0, _OUR_METHOD)
from data_utils import DATASET_ROOTS, SKIP_VIDEOS, load_annotations  # noqa: E402

ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
MODEL_ID = "google/vivit-b-16x2-kinetics400"
NUM_FRAMES = 32
HIDDEN_DIM = 768  # ViViT-b `hidden_size`


def fea_dir_for(dataset: str) -> str:
    return os.path.join(DATASET_ROOTS[dataset], "fea")


def output_path_for(dataset: str) -> str:
    return os.path.join(fea_dir_for(dataset), "fea_frames_32_google-vivit-b.pt")


def load_32_frames(dataset: str, vid: str):
    """Return a list of 32 PIL.Image frames or None if any missing."""
    from PIL import Image
    folder = os.path.join(DATASET_ROOTS[dataset], "frames_32", vid)
    if not os.path.isdir(folder):
        return None
    paths = [
        os.path.join(folder, f"frame_{i:03d}.jpg") for i in range(NUM_FRAMES)
    ]
    if not all(os.path.exists(p) for p in paths):
        return None
    return [Image.open(p).convert("RGB") for p in paths]


def load_vivit():
    """Lazy import so the module is syntax-check clean without torch."""
    import torch
    from transformers import VivitImageProcessor, VivitModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading {MODEL_ID} on {device}")
    processor = VivitImageProcessor.from_pretrained(MODEL_ID)
    model = VivitModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return model, processor, device


def extract_one(frames, model, processor, device):
    """Encode one video's 32 frames → `torch.Tensor(shape=[768])`.

    Returns the `[CLS]` token embedding from
    `outputs.last_hidden_state[:, 0]`, moved to CPU.
    """
    import torch

    inputs = processor(list(frames), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls = outputs.last_hidden_state[:, 0]  # (1, hidden_size)
    return cls.squeeze(0).float().cpu()


def process_dataset(dataset: str, split: str, model, processor, device) -> Dict[str, "torch.Tensor"]:
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
            existing[vid] = torch.zeros(HIDDEN_DIM, dtype=torch.float32)
            continue
        if vid not in ann:
            continue
        frames = load_32_frames(dataset, vid)
        if frames is None or len(frames) != NUM_FRAMES:
            existing[vid] = torch.zeros(HIDDEN_DIM, dtype=torch.float32)
            n_fail += 1
            continue
        try:
            vec = extract_one(frames, model, processor, device)
        except Exception as e:
            logging.error(f"  {vid}: ViViT forward failed: {e}")
            vec = torch.zeros(HIDDEN_DIM, dtype=torch.float32)
            n_fail += 1
        existing[vid] = vec
        n_new += 1
        if n_new % 25 == 0:
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
        description="MATCH stage 3 ViViT feature extractor"
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
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

    model, processor, device = load_vivit()
    for ds in datasets:
        for sp in splits:
            process_dataset(ds, sp, model, processor, device)
    logging.info("All ViViT extraction done.")


if __name__ == "__main__":
    main()
