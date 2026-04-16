"""Pro-Cap LAVIS — 8-frame variant.

Parallel to `reproduce_procap_lavis.py` (1 middle frame). This variant
samples **8 frames** uniformly per video, runs upstream's 8 VQA probes
on each of those 8 frames, and emits a per-video record whose
`per_frame_captions` field is a list of 8 concatenated probe-answer
strings — one per sampled frame.

Downstream consumers:
  1. `src/mod_hate_repro/video_caption_adapter.py` reads
     `per_frame_captions` and stitches the 8 strings into a single
     video caption with explicit frame markers ("Frame 1: ...
     Frame 2: ...") per the Mod-HATE brief at
     `docs/baseline_briefs/mod_hate.md`.
  2. This script **also** emits a standalone Pro-Cap 8-frame baseline
     prediction per video by running one additional **9th BLIP-2
     classification call** on top of the 8×8 descriptive probe calls.
     That 9th probe receives the 8 concatenated per-frame captions +
     the transcript and returns a yes/no verdict parsed via the same
     `parse_yes_no` helper used by the 1-frame variant. The 9th probe
     runs on the **middle frame** of the 8 sampled (index 3,
     zero-based) — a simple, documented choice consistent with the
     1-frame variant's single-frame classification call.

The 1-frame variant stays around (`reproduce_procap_lavis.py`) as a
comparison row. This 8-frame script is the **primary** Pro-Cap
baseline and additionally the caption source for Mod-HATE.

Upstream fidelity (unchanged vs `reproduce_procap_lavis.py`):
  * LAVIS loader: `load_model_and_preprocess(name="blip2_t5",
    model_type="caption_coco_flant5xl", is_eval=True)` +
    `model = model.float()` upcast (upstream-exact; notebook cells
    `ae5d43bd` and `29555f87`).
  * `generate_prompt_result(im, ques)` byte-for-byte port of upstream
    notebook cell `1e529ebf`, including `vis_processors["eval"](im).float().unsqueeze(0).to(device)`,
    the `"Question: %s Answer:"` prompt template, and
    `length_penalty=3.0`.
  * 8 VQA probes in upstream order (race → gender → animal → person →
    country → what_animal → disabled → religion), each string verbatim.

Changes vs `reproduce_procap_lavis.py`:
  * Replaces the single `load_middle_frame(...)` call with
    `load_eight_frames(...)` which returns a list of exactly 8 PIL
    images per video.
  * The probing loop now iterates over 8 frames × 8 probes = 64
    `generate_prompt_result` calls per video, building a
    `per_frame_captions` list of 8 strings (each string is the ";
    "-joined 8-probe answers for that frame, same concat used by the
    1-frame variant).
  * **No 9th classification probe**. This variant produces caption
    material only; the binary verdict is Mod-HATE's job.
  * Output schema: `{"video_id": vid, "per_frame_captions": [c0, ...,
    c7]}`, 8 strings per video.
  * Output path:
    `results/procap_lavis_blip2_flan_t5_xl_8frame/<dataset>/<split>_procap.jsonl`
    — distinct from the 1-frame output at
    `results/procap_lavis_blip2_flan_t5_xl/<dataset>/<split>_procap.jsonl`.

Frame sampling rule:
  * If `get_media_path(vid, ds)` returns pre-extracted `frames/<vid>/`
    (MHClip_EN / MHClip_ZH / HateMM), we sample 8 indices via
    `np.linspace(0, len(jpgs)-1, 8, dtype=int)` and open those jpgs.
  * If `get_media_path` returns an mp4 (ImpliHateVid), we decord the
    mp4 and take 8 indices via `np.linspace(0, n-1, 8, dtype=int)`.

This matches the `feedback_meme_to_video_8frames.md` rule — 8 uniform
frames per video — without constraining the variant to the 16-frame
pre-extraction used by MATCH's `frames_16/` layout (Pro-Cap doesn't
depend on MATCH's stage, and reusing `get_media_path` keeps the two
image stages independent).

The 1-frame version (`reproduce_procap_lavis.py`) is unchanged; the
director's in-flight 1-frame run 8431 continues writing to
`results/procap_lavis_blip2_flan_t5_xl/`.
"""

import argparse
import json
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))

from data_utils import (  # noqa: E402
    DATASET_ROOTS,
    SKIP_VIDEOS,
    get_media_path,
    load_annotations,
)

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
NUM_FRAMES = 8
OUTPUT_SUBDIR = "procap_lavis_blip2_flan_t5_xl_8frame"

# 8 VQA probing questions from Pro-Cap-Generation.ipynb (Cao et al.),
# in upstream order. Byte-for-byte identical to the 1-frame variant.
PROCAP_QUESTIONS = [
    ("race", "what is the race of the person in the image?"),
    ("gender", "what is the gender of the person in the image?"),
    ("animal", "is there an animal in the image?"),
    ("person", "is there a person in the image?"),
    ("country", "which country does the person in the image come from?"),
    ("what_animal", "what animal is in the image?"),
    ("disabled", "are there disabled people in the image?"),
    ("religion", "what is the religion of the person in the image?"),
]

# 9th BLIP-2 classification probe — mirrors the label-free adaptation
# used by the 1-frame variant, with a "video" framing instead of
# "image". The body (no leading "Question:" / trailing "Answer:") is
# wrapped by the upstream `generate_prompt_result` helper with
# `"Question: %s Answer:"`, so the string the model actually sees is
# byte-identical in shape to the upstream probe format.
CLASSIFICATION_PROMPT_TEMPLATE = (
    "The video has been described as follows: {captions}. "
    "The accompanying transcript says: \"{transcript}\". "
    "Is this video promoting hateful, discriminatory, or "
    "dehumanizing stereotypes against any identity group? "
    "Answer yes or no."
)


def parse_yes_no(text):
    """Identical to the 1-frame variant's parser."""
    t = (text or "").strip().lower().lstrip("\"'*`#- ")
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return -1


def load_eight_frames(media_path, media_type):
    """Return a list of 8 PIL.Image RGB frames uniformly sampled from
    the video. Returns None if the source has zero usable frames.
    """
    from PIL import Image
    import numpy as np

    if media_type == "frames":
        import glob as globmod
        jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")))
        if not jpgs:
            return None
        if len(jpgs) >= NUM_FRAMES:
            idx = np.linspace(0, len(jpgs) - 1, NUM_FRAMES, dtype=int).tolist()
        else:
            # Pad by repeating the last available frame to reach 8.
            idx = list(range(len(jpgs))) + [len(jpgs) - 1] * (
                NUM_FRAMES - len(jpgs)
            )
        return [Image.open(jpgs[i]).convert("RGB") for i in idx]

    # mp4 path via decord
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(media_path)
    n = len(vr)
    if n == 0:
        return None
    if n >= NUM_FRAMES:
        idx = np.linspace(0, n - 1, NUM_FRAMES, dtype=int).tolist()
    else:
        idx = list(range(n)) + [n - 1] * (NUM_FRAMES - n)
    frames = []
    for i in idx:
        arr = vr[i].asnumpy()
        frames.append(Image.fromarray(arr).convert("RGB"))
    return frames


def resume_done_ids(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("video_id"):
                        done.add(r["video_id"])
                except Exception:
                    pass
    return done


def verify_output_integrity(out_path):
    if not os.path.exists(out_path):
        return 0, 0
    n_total = 0
    n_ok = 0
    with open(out_path) as f:
        for line in f:
            n_total += 1
            try:
                r = json.loads(line)
                if (
                    r.get("video_id")
                    and isinstance(r.get("per_frame_captions"), list)
                    and len(r["per_frame_captions"]) == NUM_FRAMES
                    and "pred" in r
                    and "verdict_text" in r
                ):
                    n_ok += 1
            except Exception:
                pass
    return n_ok, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument(
        "--transcript-limit",
        type=int,
        default=1000,
        help="Max chars of transcript fed to the 9th classification probe",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Smoke-test helper: process only the first N videos (0 = no limit)",
    )
    args = parser.parse_args()
    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    datasets = ALL_DATASETS if args.all else [args.dataset]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info(
        f"Pro-Cap LAVIS 8-frame repro: datasets={datasets} NUM_FRAMES={NUM_FRAMES}"
    )

    import torch
    from lavis.models import load_model_and_preprocess

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading LAVIS blip2_t5 / caption_coco_flant5xl on {device}")
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="caption_coco_flant5xl",
        is_eval=True,
        device=device,
    )
    model = model.float()
    logging.info("LAVIS BLIP-2 loaded")

    def generate_prompt_result(im, ques):
        """Verbatim upstream helper (notebook cell 1e529ebf)."""
        image = vis_processors["eval"](im).float().unsqueeze(0).to(device)
        ans = model.generate(
            {"image": image, "prompt": ("Question: %s Answer:" % (ques))},
            length_penalty=3.0,
        )
        return ans[0]

    for ds in datasets:
        annotations = load_annotations(ds)
        split_path = os.path.join(
            DATASET_ROOTS[ds], "splits", f"{args.split}_clean.csv"
        )
        if not os.path.isfile(split_path):
            from data_utils import generate_clean_splits
            generate_clean_splits(ds)
        with open(split_path) as f:
            vids = [line.strip() for line in f if line.strip()]

        out_dir = os.path.join(PROJECT_ROOT, "results", OUTPUT_SUBDIR, ds)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{args.split}_procap.jsonl")
        done = resume_done_ids(out_path)
        if done:
            logging.info(f"[{ds}] resume: {len(done)} already done")
        remaining = [v for v in vids if v not in done]
        if args.limit and args.limit > 0:
            remaining = remaining[: args.limit]
            logging.info(f"[{ds}] --limit {args.limit} (smoke helper)")
        logging.info(f"[{ds}] {len(vids)} total, {len(remaining)} to process")

        t0 = time.time()
        n_processed = 0
        n_skipped = 0
        for vid in remaining:
            if vid in SKIP_VIDEOS.get(ds, set()):
                continue
            ann = annotations.get(vid)
            if ann is None:
                logging.warning(f"  {vid}: no annotation")
                n_skipped += 1
                continue
            media = get_media_path(vid, ds)
            if media is None:
                logging.warning(f"  {vid}: no media")
                n_skipped += 1
                continue
            media_path, media_type = media
            try:
                frames = load_eight_frames(media_path, media_type)
            except Exception as e:
                logging.warning(f"  {vid}: frame load failed: {e}")
                frames = None
            if frames is None or len(frames) != NUM_FRAMES:
                rec = {
                    "video_id": vid,
                    "per_frame_captions": [],
                    "pred": -1,
                    "error": "no_frames",
                }
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                n_skipped += 1
                continue

            per_frame_captions = []
            per_frame_probe_answers = []
            for frame_idx, frame in enumerate(frames):
                answers = {}
                for key, q in PROCAP_QUESTIONS:
                    try:
                        ans = generate_prompt_result(frame, q)
                    except Exception as e:
                        logging.warning(
                            f"  {vid}/frame{frame_idx}/{key}: probe failed: {e}"
                        )
                        ans = ""
                    answers[key] = ans
                # Concatenation rule matches the 1-frame variant:
                # ";"-joined answers in upstream probe order.
                concat = "; ".join(
                    answers.get(k, "") for k, _ in PROCAP_QUESTIONS
                )
                per_frame_captions.append(concat)
                per_frame_probe_answers.append(answers)

            # ----- 9th probe: single standalone classification call -----
            # Build the video-level caption block and stitch with
            # transcript, then ask BLIP-2 yes/no on the middle of the 8
            # sampled frames (index 3, zero-based). The 9th call uses
            # the same upstream `generate_prompt_result` helper so it
            # inherits `length_penalty=3.0` + the
            # `"Question: %s Answer:"` wrapper.
            video_caption = " | ".join(
                f"Frame {i+1}: {c}"
                for i, c in enumerate(per_frame_captions)
            )
            transcript = (ann.get("transcript", "") or "")[: args.transcript_limit]
            class_question = CLASSIFICATION_PROMPT_TEMPLATE.format(
                captions=video_caption,
                transcript=transcript.replace('"', "'"),
            )
            middle_idx = NUM_FRAMES // 2 - 1  # index 3 for NUM_FRAMES=8
            try:
                verdict_text = generate_prompt_result(
                    frames[middle_idx], class_question
                )
            except Exception as e:
                logging.error(f"  {vid}: classification probe failed: {e}")
                verdict_text = ""
            pred = parse_yes_no(verdict_text)

            rec = {
                "video_id": vid,
                "per_frame_captions": per_frame_captions,
                "per_frame_probe_answers": per_frame_probe_answers,
                "pred": pred,
                "verdict_text": verdict_text,
                "caption": video_caption,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            n_processed += 1

            if n_processed % 25 == 0:
                n_ok, n_total = verify_output_integrity(out_path)
                rate = (
                    n_processed / (time.time() - t0)
                    if (time.time() - t0) > 0
                    else 0
                )
                logging.info(
                    f"  [{ds}] [{n_processed}/{len(remaining)}] "
                    f"{rate:.2f} vid/s  integrity {n_ok}/{n_total}"
                )

        logging.info(
            f"[{ds}] done. {n_processed} processed, {n_skipped} skipped."
        )

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
