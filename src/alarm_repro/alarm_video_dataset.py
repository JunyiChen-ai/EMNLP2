"""ALARM video dataset adapter.

Wraps our 4 video benchmarks in the per-item shape ALARM's upstream
runners expect. Upstream runners pull `{id, img, text, label, split}`
per meme row from `data/<dataset>/data.jsonl`
(`src/model/utils/datasets/FHM_data.py:32-43`); for video, the
equivalent is `{vid, frames: [PIL, ...8], text: transcript, label,
split}`.

All 4 stages below (Label / make_embeddings / conduct_retrieval /
Experience / Reference / InPredict) consume this adapter; none of
the ALARM stage files read upstream's meme data loaders directly.

8-frame sampling rule comes from `feedback_meme_to_video_8frames.md`:
indices `[0, 2, 4, 6, 8, 10, 12, 14]` from `frames_16/<vid>/`. The
same sampling is used by MATCH (`run_match_agents.py`) and LoReHM
(`lorehm_video_dataset.py`), so a single set of extracted jpgs
serves all three methods.
"""

import os
import sys

# Allow PIL to finish decoding JPGs with a truncated tail (hit on MHClip_ZH
# frames_16; same hardening as lorehm_video_dataset.py and procap_v3).
from PIL import ImageFile  # noqa: E402
ImageFile.LOAD_TRUNCATED_IMAGES = True

_OUR_METHOD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "our_method"
)
sys.path.insert(0, _OUR_METHOD)
from data_utils import DATASET_ROOTS, SKIP_VIDEOS, load_annotations  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]

FRAME_INDICES = (0, 2, 4, 6, 8, 10, 12, 14)
NUM_FRAMES = len(FRAME_INDICES)


def collapse_label(dataset, gt_label):
    if dataset == "HateMM":
        return 1 if gt_label == "Hate" else 0
    if dataset == "ImpliHateVid":
        return 1 if gt_label == "Hateful" else 0
    return 1 if gt_label in ("Hateful", "Offensive") else 0


def load_split_video_ids(dataset, split):
    csv_path = os.path.join(
        DATASET_ROOTS[dataset], "splits", f"{split}_clean.csv"
    )
    if not os.path.isfile(csv_path):
        from data_utils import generate_clean_splits
        generate_clean_splits(dataset)
    with open(csv_path) as f:
        return [line.strip() for line in f if line.strip()]


def frames_dir_for(dataset, vid):
    return os.path.join(DATASET_ROOTS[dataset], "frames_16", vid)


def load_8_frames(dataset, vid, indices=FRAME_INDICES):
    """Return a list of 8 PIL.Image RGB frames or None if missing."""
    from PIL import Image
    folder = frames_dir_for(dataset, vid)
    paths = [
        os.path.join(folder, f"frame_{i:03d}.jpg") for i in indices
    ]
    if not all(os.path.exists(p) for p in paths):
        return None
    return [Image.open(p).convert("RGB") for p in paths]


def build_video_items(dataset, split, include_frames=True):
    """Per-video list of dicts matching upstream's per-meme row schema.

    Returns `([items], missing_counts)` where each item is:
        {
          "id":     str,            # video_id
          "split":  "train" | "test",
          "text":   str,            # transcript (substitutes meme OCR)
          "label":  int,            # collapsed binary
          "frames": list[PIL.Image] | None,  # 8 frames if include_frames
        }

    Videos in `SKIP_VIDEOS` or missing annotation/frames are dropped;
    counts surfaced in `missing`.
    """
    ann = load_annotations(dataset)
    skip_set = SKIP_VIDEOS.get(dataset, set())
    vids = load_split_video_ids(dataset, split)
    items = []
    missing = {"skip": 0, "no_ann": 0, "no_frames": 0}
    for vid in vids:
        if vid in skip_set:
            missing["skip"] += 1
            continue
        if vid not in ann:
            missing["no_ann"] += 1
            continue
        text = (ann[vid].get("transcript", "") or "")
        label = collapse_label(dataset, ann[vid]["label"])
        entry = {
            "id": vid,
            "split": split,
            "text": text,
            "label": label,
            "frames": None,
        }
        if include_frames:
            frames = load_8_frames(dataset, vid)
            if frames is None:
                missing["no_frames"] += 1
                continue
            entry["frames"] = frames
        items.append(entry)
    return items, missing
