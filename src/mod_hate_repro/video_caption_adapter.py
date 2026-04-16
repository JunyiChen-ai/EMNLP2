"""Mod-HATE video caption adapter.

Produces the concatenated "video caption" string Mod-HATE's LLaMA-7B
pipeline sees as its `Image caption:` field. Upstream Mod-HATE
consumed a single meme-image caption; we adapt to video by reading
Pro-Cap's per-video caption output.

**Caption source — three-path with fallback (2026-04-15 V3 update)**

The default is now the **Pro-Cap V3 Qwen2-VL multi-image (16-frame)**
output (`procap_v3/<dataset>/captions_<split>.pkl`). V3 ships the
upstream Pro-Cap `{vid: {race, gender, ..., gen}}` pickle schema
unchanged, and this adapter renders the per-video record into a
single `gen . probe_1 . probe_2 ...` string equivalent to upstream
`scr/dataset.py:153-159`'s `cap` accumulation. The 1-frame and 8-frame
BLIP-2 paths remain as explicit fallbacks but are no longer default. This adapter now:

  1. Tries the 8-frame path first
     (`procap_lavis_blip2_flan_t5_xl_8frame/<ds>/<split>_procap.jsonl`).
     Each record must have a list of 8 strings under
     `per_frame_captions`. If the file exists and has 8-frame records,
     `build_video_caption` stitches them with `"Frame i+1: <cap>"`
     markers (original 8-frame rule).
  2. Falls back to the 1-frame path
     (`procap_lavis_blip2_flan_t5_xl/<ds>/<split>_procap.jsonl`). Each
     record has a single `caption` field (= `"; "`-joined 8 VQA probe
     answers from `reproduce_procap_lavis.py`). In this path the
     caption is passed through as-is **without** frame markers — it's
     already a single coherent concatenation of the Pro-Cap probe
     answers, which is exactly what upstream Mod-HATE consumed.

Either path produces the same final row shape, so `reproduce_mod_hate.py`
and `lora_compose.py` don't need to know which Pro-Cap variant
supplied the captions.

Data sources per dataset:
  * 8-frame Pro-Cap captions:
    `results/procap_lavis_blip2_flan_t5_xl_8frame/<dataset>/<split>_procap.jsonl`
    (each line: `{"video_id": ..., "per_frame_captions": [c0..c7], ...}`)
    — **currently unused, kept for future**.
  * 1-frame Pro-Cap captions (default after 2026-04-15):
    `results/procap_lavis_blip2_flan_t5_xl/<dataset>/<split>_procap.jsonl`
    (each line: `{"video_id": ..., "caption": "<single concat>", ...}`)
  * Video transcripts: `data_utils.load_annotations(dataset)[vid]["transcript"]`
    (substitutes upstream's meme-OCR text per the Mod-HATE brief).
  * Ground-truth labels: `load_annotations(...)[vid]["label"]` → collapsed
    to {0, 1} via the same rule as `eval_generative_predictions.collapse_label`.

Video caption string format:
  * 8-frame path (future): `"Frame 1: {cap_0}. Frame 2: {cap_1}. ... Frame 8: {cap_7}."`
  * 1-frame path (default): the `caption` field as-is (no frame markers,
    already a `"; "`-joined 8-probe concat per upstream Pro-Cap notebook
    + our label-free 9th-probe adapter).

This module is intentionally pure-Python / no-torch, so it can be
imported from both the LoRA composition (`lora_compose.py`) and the
main driver (`reproduce_mod_hate.py`) without dragging GPU state in.
"""

import json
import os
import sys

_OUR_METHOD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "our_method"
)
sys.path.insert(0, _OUR_METHOD)
from data_utils import DATASET_ROOTS, SKIP_VIDEOS, load_annotations  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
PROCAP_8FRAME_SUBDIR = "procap_lavis_blip2_flan_t5_xl_8frame"
PROCAP_1FRAME_SUBDIR = "procap_lavis_blip2_flan_t5_xl"
PROCAP_V3_SUBDIR = "procap_v3"
NUM_FRAMES = 8

# Field order for the Pro-Cap V3 pkl translation. Matches
# `src/procap_v3_repro/dataset_procap.py:PROBE_FIELDS` + `GEN_FIELD`,
# so the rendered string here is identical in shape to upstream
# `dataset.py:153-159`'s `cap = gen + ' . ' + ' . '.join(probe_answers)`
# and equivalent in semantics to the 1-frame Pro-Cap path's
# `"; "`-joined probe concat that Mod-HATE consumed previously.
V3_PROBE_FIELDS = [
    "race", "gender", "country", "animal",
    "what_animal", "person", "disabled", "religion",
]
V3_GEN_FIELD = "gen"


def collapse_label(dataset, gt_label):
    """Mirror of `eval_generative_predictions.collapse_label`.

    Kept here as a direct copy so this module has no hard import on
    the eval package (which lives under `src/naive_baseline/`).
    """
    if dataset == "HateMM":
        return 1 if gt_label == "Hate" else 0
    if dataset == "ImpliHateVid":
        return 1 if gt_label == "Hateful" else 0
    return 1 if gt_label in ("Hateful", "Offensive") else 0


def build_video_caption(per_frame_captions):
    """Stitch 8 per-frame Pro-Cap captions into a single "video caption".

    Format comes from the Mod-HATE brief (`docs/baseline_briefs/mod_hate.md`):
        "Frame 1: {cap_0}. Frame 2: {cap_1}. ... Frame 8: {cap_7}."

    The explicit `Frame N:` markers let LLaMA-7B see the frame
    ordering; the `.`-terminated separators preserve the upstream
    meme-caption cadence. If a per-frame caption is empty, the
    corresponding slot shows `"Frame N: ."` (empty payload) — this
    keeps the caller's `len(per_frame_captions) == 8` invariant from
    the Pro-Cap 8-frame writer path.
    """
    assert len(per_frame_captions) == NUM_FRAMES, (
        f"expected {NUM_FRAMES} per-frame captions, got {len(per_frame_captions)}"
    )
    parts = [
        f"Frame {i + 1}: {c}" for i, c in enumerate(per_frame_captions)
    ]
    return ". ".join(parts) + "."


def load_procap_8frame(dataset, split="test"):
    """Return `{video_id -> per_frame_captions}` from the 8-frame jsonl.

    Kept for the future 8-frame path. Currently unused by default
    (the 2026-04-15 plan change routes through `load_procap_1frame`
    instead). The dispatcher in `load_procap_captions` still tries
    this first and falls back to the 1-frame path.
    """
    path = os.path.join(
        PROJECT_ROOT,
        "results",
        PROCAP_8FRAME_SUBDIR,
        dataset,
        f"{split}_procap.jsonl",
    )
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = r.get("video_id")
            caps = r.get("per_frame_captions")
            if not vid or not isinstance(caps, list) or len(caps) != NUM_FRAMES:
                continue
            out[vid] = caps
    return out


def load_procap_1frame(dataset, split="test"):
    """Return `{video_id -> caption}` from the 1-frame Pro-Cap jsonl.

    Matches the schema written by `src/procap_repro/reproduce_procap_lavis.py`:
    each record has a `caption` field (a single string, `"; "`-joined
    upstream 8-probe answers) that Mod-HATE can pass through as the
    `Image caption:` field verbatim, no frame markers needed.

    The 1-frame jsonl also carries `pred` / `verdict_text` fields from
    the standalone Pro-Cap baseline's 9th probe — we ignore those here
    (Mod-HATE does its own classification downstream).
    """
    path = os.path.join(
        PROJECT_ROOT,
        "results",
        PROCAP_1FRAME_SUBDIR,
        dataset,
        f"{split}_procap.jsonl",
    )
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = r.get("video_id")
            caption = r.get("caption")
            if not vid or not isinstance(caption, str) or not caption:
                continue
            out[vid] = caption
    return out


def load_procap_v3(dataset, split="test"):
    """Return `{video_id -> caption_string}` from the V3 captioner pkl.

    Source: `results/procap_v3/<dataset>/captions_<split>.pkl`,
    written by `src/procap_v3_repro/generate_captions_qwen2vl.py`.
    Schema: `{vid: {race, gender, animal, person, country, what_animal,
                    disabled, religion, gen}}` (all str).

    Translation: render the gen caption as the spine and append the
    8 probe answers separated by `' . '`, mirroring upstream Pro-Cap
    `dataset.py:153-159`. Empty fields are dropped (matches upstream's
    `if info.startswith('no'): continue` style guards). Returns the
    final stitched string per video so downstream `build_examples`
    can pass it through as the `Image caption:` field unchanged.
    """
    import pickle
    path = os.path.join(
        PROJECT_ROOT, "results", PROCAP_V3_SUBDIR, dataset,
        f"captions_{split}.pkl",
    )
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            captions = pickle.load(f)
    except Exception:
        return {}
    if not isinstance(captions, dict):
        return {}
    out = {}
    for vid, record in captions.items():
        if not isinstance(record, dict):
            continue
        gen = (record.get(V3_GEN_FIELD, "") or "").strip()
        parts = []
        for field in V3_PROBE_FIELDS:
            val = (record.get(field, "") or "").strip()
            if val:
                parts.append(val)
        if gen and parts:
            stitched = gen + " . " + " . ".join(parts)
        elif gen:
            stitched = gen
        elif parts:
            stitched = " . ".join(parts)
        else:
            continue
        out[vid] = stitched
    return out


def load_procap_captions(dataset, split="test", prefer="v3_qwen2vl"):
    """Dispatch: return `{vid -> video_caption_string}` from whichever
    Pro-Cap variant is selected.

    Per the 2026-04-15 Pro-Cap V3 brief, **the V3 Qwen2-VL multi-image
    path is the new default** (`prefer="v3_qwen2vl"`). It reads the V3
    captioner pkl `results/procap_v3/<dataset>/captions_<split>.pkl`
    and renders each record into a single `gen . probe_1 . probe_2 ...`
    string equivalent to the upstream Pro-Cap stitch in
    `dataset.py:153-159`.

    - `prefer="v3_qwen2vl"` (default, post-2026-04-15): V3 Qwen2-VL pkl.
      Falls back to 1-frame BLIP-2 jsonl if the V3 pkl is missing or
      empty.
    - `prefer="1frame"` (2026-04-15 ~04 plan): 1-frame BLIP-2 jsonl
      `caption` field passed through as-is. Falls back to 8-frame.
    - `prefer="8frame"` (originally planned): 8-frame BLIP-2 stitch.
      Falls back to 1-frame.

    All paths return a `{vid -> str}` dict so `build_examples` can
    treat the cases uniformly.
    """
    if prefer == "v3_qwen2vl":
        primary = load_procap_v3(dataset, split)
        if primary:
            return primary
        # Fallback chain: V3 → 1-frame → 8-frame
        one = load_procap_1frame(dataset, split)
        if one:
            return one
        eight = load_procap_8frame(dataset, split)
        return {
            vid: build_video_caption(caps) for vid, caps in eight.items()
        }
    elif prefer == "1frame":
        primary = load_procap_1frame(dataset, split)
        if primary:
            return primary
        eight = load_procap_8frame(dataset, split)
        return {
            vid: build_video_caption(caps) for vid, caps in eight.items()
        }
    elif prefer == "8frame":
        eight = load_procap_8frame(dataset, split)
        if eight:
            return {
                vid: build_video_caption(caps) for vid, caps in eight.items()
            }
        return load_procap_1frame(dataset, split)
    else:
        raise ValueError(
            f"unknown prefer={prefer}, expected 'v3_qwen2vl' / '1frame' / '8frame'"
        )


def load_split_video_ids(dataset, split):
    """Return the ordered list of video_ids for `splits/<split>_clean.csv`."""
    csv_path = os.path.join(
        DATASET_ROOTS[dataset], "splits", f"{split}_clean.csv"
    )
    if not os.path.isfile(csv_path):
        from data_utils import generate_clean_splits
        generate_clean_splits(dataset)
    with open(csv_path) as f:
        return [line.strip() for line in f if line.strip()]


def build_examples(dataset, split, transcript_limit=512):
    """Return a list of Mod-HATE-ready example dicts for one split.

    Each entry matches the `Few_HM_Data` / `hfm_generation` schema
    upstream expects:
        {
          "img":         <video_id>,     # upstream calls this "img"
          "label":       0 | 1,          # collapsed binary label
          "input":       "Image caption:<video_caption>\\nMeme text:<transcript>",
          "instruction": "<upstream instruction, meme→video rewritten>",
          "output":      "Yes" | "No",   # upstream verbolizer, unchanged
        }

    The per-video rows are produced only for videos that have Pro-Cap
    captions available (8-frame path preferred, 1-frame fallback).
    Videos missing captions from both paths are dropped with a
    warning-return so the caller can report them. Upstream's
    `SKIP_VIDEOS` is always filtered out.
    """
    annotations = load_annotations(dataset)
    skip_set = SKIP_VIDEOS.get(dataset, set())
    caps_by_vid = load_procap_captions(dataset, split)
    vids = load_split_video_ids(dataset, split)

    rows = []
    missing_caps = []
    for vid in vids:
        if vid in skip_set:
            continue
        if vid not in annotations:
            continue
        if vid not in caps_by_vid:
            missing_caps.append(vid)
            continue
        # `caps_by_vid[vid]` is already the final video-caption string
        # — either the `"Frame i+1: ..."`-marked 8-frame stitch (when
        # the 8-frame path was taken) or the raw 1-frame `caption`
        # passthrough (when the 1-frame fallback path was taken).
        video_caption = caps_by_vid[vid]
        transcript = (annotations[vid].get("transcript", "") or "")[
            :transcript_limit
        ]
        label = collapse_label(dataset, annotations[vid]["label"])
        # Upstream verbolizer: `POS_WORD="No"`, `NEG_WORD="Yes"` with
        # `{0: POS_WORD, 1: NEG_WORD}` → label=1 → "Yes" (hateful).
        output = "Yes" if label == 1 else "No"
        row = {
            "img": vid,  # upstream field name
            "label": label,
            "input": (
                f"Image caption:{video_caption}\n"
                f"Meme text:{transcript}"
            ),
            "instruction": (
                "Please decide whether the video is hateful according "
                "to its video caption and video transcript."
            ),
            "output": output,
        }
        rows.append(row)
    return rows, missing_caps


def build_support_and_test(dataset, num_shots, seed=1111):
    """K-shot support set + test set pair.

    Upstream `Few_HM_Data.process_data` shuffles the train split and
    picks the first `num_shots` positive + `num_shots` negative
    examples (`counts[0]==num_shots and counts[1]==num_shots`). We
    follow the same logic: sample a balanced K/K split from our train
    split.

    Returns `(support_rows, test_rows, missing_caps_info)`.
    """
    import random

    rng = random.Random(seed)
    train_rows, missing_train = build_examples(dataset, "train")
    test_rows, missing_test = build_examples(dataset, "test")

    rng.shuffle(train_rows)
    support = []
    counts = {0: 0, 1: 0}
    for row in train_rows:
        lbl = row["label"]
        if counts[lbl] >= num_shots:
            continue
        support.append(row)
        counts[lbl] += 1
        if counts[0] == num_shots and counts[1] == num_shots:
            break

    missing_info = {
        "train": missing_train,
        "test": missing_test,
        "support_counts": counts,
    }
    return support, test_rows, missing_info
