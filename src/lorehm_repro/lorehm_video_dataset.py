"""LoReHM video dataset adapter (rework 2026-04-15).

Supplies per-video items in the same shape upstream's `Dataset`
class (`external_repos/lorehm/utils/utils.py:14-51`) produces for
meme datasets, but for our 4 video benchmarks. The per-item
`__getitem__` tuple matches upstream's:
    (index, image, text, label, rel_sampl)

Rework: the grid composite path is removed. Frames are passed
directly as a **list of 16 PIL.Image** objects for LLaVA-Next's
native multi-image input path.

Mappings from upstream to our video setting:
  * `image` — for a meme, upstream passes a single file path to
    LLaVA. For video, this field is a list of **16** PIL.Image
    frames loaded from `frames_16/<vid>/frame_NNN.jpg` for
    NNN in 000..015 (all 16, no subsampling).
  * `text` — upstream's meme OCR → our video transcript
    (`ann["transcript"]`), per the brief adaptation.
  * `label` — collapsed binary via
    `eval_generative_predictions.collapse_label` semantics; kept
    inline here so this file has no hard import on the eval package.
  * `rel_sampl` — pre-computed retrieval result from
    `retrieval.build_rel_sampl(...)`, a tuple
    `(example, harmful_example, harmless_example)` matching
    upstream's schema in `main.py:30`. `None` when RSA is disabled.

SKIP_VIDEOS is applied; videos with missing `frames_16/<vid>/` are
dropped from the iteration (caller still sees them counted as
skipped).
"""

import os
import sys

# Allow PIL to finish decoding JPGs that were written with a truncated
# tail (~25 bytes short). Hit on MHClip_ZH test frames in retrieval job
# 8481 — frame extractor occasionally produces mostly-complete bytes
# that fail strict PIL decoding. With this flag, the decoder accepts
# the partial tail and returns whatever it managed to read.
from PIL import ImageFile  # noqa: E402
ImageFile.LOAD_TRUNCATED_IMAGES = True

_OUR_METHOD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "our_method"
)
sys.path.insert(0, _OUR_METHOD)
from data_utils import DATASET_ROOTS, SKIP_VIDEOS, load_annotations  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]

# Rework 2026-04-15: LLaVA-Next multi-image path — use all 16 frames
# from `frames_16/<vid>/frame_NNN.jpg` for NNN in 000..015. No
# subsampling. Single-tile 336x336 per frame is enforced in
# llava_runner.py to cap visual tokens at ~9216 per call.
FRAME_INDICES = tuple(range(16))
NUM_FRAMES = len(FRAME_INDICES)


def collapse_label(dataset, gt_label):
    """Mirror of `eval_generative_predictions.collapse_label`."""
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


def load_frames_16(dataset, vid, indices=FRAME_INDICES):
    """Return a list of PIL.Image RGB frames for the given indices.

    Matches `src/match_repro/run_match_agents.py`'s path resolution
    for `frames_16/<vid>/frame_NNN.jpg`. Returns None if any of the
    requested frames are missing.
    """
    from PIL import Image
    folder = frames_dir_for(dataset, vid)
    paths = [
        os.path.join(folder, f"frame_{i:03d}.jpg") for i in indices
    ]
    if not all(os.path.exists(p) for p in paths):
        return None
    return [Image.open(p).convert("RGB") for p in paths]


def build_video_items(dataset, split, rel_sampl_map=None):
    """Build a list of per-video item dicts matching upstream's tuple
    shape `(idx, image, text, label, rel_sampl)`. Here we use dicts
    with explicit keys.

    Returns a list of
        {"idx", "vid", "frames": [PIL, ...16], "text", "label", "rel_sampl"}
    in split order. Skipped videos (in SKIP_VIDEOS, missing
    annotations, or missing frames) are dropped with a count
    reported via a second return value.
    """
    ann = load_annotations(dataset)
    skip_set = SKIP_VIDEOS.get(dataset, set())
    vids = load_split_video_ids(dataset, split)
    items = []
    missing = {"skip": 0, "no_ann": 0, "no_frames": 0}
    for idx, vid in enumerate(vids):
        if vid in skip_set:
            missing["skip"] += 1
            continue
        if vid not in ann:
            missing["no_ann"] += 1
            continue
        frames = load_frames_16(dataset, vid)
        if frames is None:
            missing["no_frames"] += 1
            continue
        text = (ann[vid].get("transcript", "") or "")
        label = collapse_label(dataset, ann[vid]["label"])
        rel = None
        if rel_sampl_map is not None:
            rel = rel_sampl_map.get(vid)
        items.append({
            "idx": idx,
            "vid": vid,
            "frames": frames,
            "text": text,
            "label": label,
            "rel_sampl": rel,
        })
    return items, missing
