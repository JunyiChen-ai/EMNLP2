"""Faithful Pro-Cap reproduction adapted to video.

Upstream: Cao et al., ACM MM 2023, arxiv 2308.08088
Repo: https://github.com/Social-AI-Studio/Pro-Cap
Backbone: BLIP-2 FlanT5-XL (Salesforce/blip2-flan-t5-xl) — paper's exact
          model, loaded via `load_model_and_preprocess(name='blip2_t5',
          model_type='caption_coco_flant5xl')`. We use the HuggingFace
          transformers port `Salesforce/blip2-flan-t5-xl` which is the
          same weights exported to the `Blip2ForConditionalGeneration`
          class.

Method (faithful to the paper's "probing" stage):
  For each meme image, the paper probes the frozen BLIP-2 with 8 VQA
  questions: race, gender, country, religion, person, animal, what
  animal, disabled (all from Pro-Cap-Generation.ipynb). The answers
  are then concatenated into a "caption" string and fed to a
  downstream classifier that the paper trains supervised on the
  target labels.

Label-free adaptation for our comparison (documented in README):
  We substitute the supervised downstream classifier with a **9th
  BLIP-2 probe** that asks directly whether the content is hateful,
  using the concatenated 8-answer caption + the video's transcript
  as textual context. This keeps the entire pipeline zero-shot
  through a single frozen MLLM (BLIP-2 FlanT5-XL), which matches the
  spirit of the "probing-only half" requirement from the user.

Video adaptation (1 frame):
  Pro-Cap is image-only; we sample 1 middle frame per video as the
  canonical representative and run the 8 probes + 1 classification
  probe on that frame.

Total BLIP-2 calls per video: 9 (baselines are exempt from our
method's 2-call cap; user directive is "faithful reproduction").

Output schema matches eval_one's `{video_id, pred}` so
`eval_generative_predictions.eval_one()` consumes the jsonl directly.
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
sys.path.insert(0, os.path.join(_HERE, "..", "naive_baseline"))

from data_utils import (  # noqa: E402
    DATASET_ROOTS,
    SKIP_VIDEOS,
    get_media_path,
    load_annotations,
)

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
DEFAULT_MODEL = "Salesforce/blip2-flan-t5-xl"

# 8 VQA probing questions from Pro-Cap-Generation.ipynb
# (Cao et al., Pro-Cap ACM MM 2023), in paper order.
PROCAP_QUESTIONS = [
    ("race", "what is the race of the person in the image?"),
    ("gender", "what is the gender of the person in the image?"),
    ("country", "which country does the person in the image come from?"),
    ("religion", "what is the religion of the person in the image?"),
    ("person", "is there a person in the image?"),
    ("animal", "is there an animal in the image?"),
    ("what_animal", "what animal is in the image?"),
    ("disabled", "are there disabled people in the image?"),
]

# BLIP-2 VQA prompt template from Pro-Cap-Generation.ipynb cell 13:
#   prompt = "Question: %s Answer:" % ques
VQA_PROMPT_TEMPLATE = "Question: {q} Answer:"

# Label-free 9th probe: classification using concatenated captions +
# transcript as textual context.
CLASSIFICATION_PROMPT_TEMPLATE = (
    "Question: The image has been described as follows: {caption}. "
    "The accompanying transcript says: \"{transcript}\". "
    "Is this content promoting hateful, discriminatory, or "
    "dehumanizing stereotypes against any identity group? "
    "Answer yes or no. Answer:"
)


def load_middle_frame(media_path, media_type):
    """Return a PIL.Image RGB of the middle frame of the video.

    For pre-extracted frames: pick the middle jpg.
    For mp4: decode with decord and grab the middle frame.
    """
    from PIL import Image
    if media_type == "frames":
        import glob as globmod
        jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")))
        if not jpgs:
            return None
        return Image.open(jpgs[len(jpgs) // 2]).convert("RGB")
    # mp4 path
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(media_path)
    n = len(vr)
    if n == 0:
        return None
    mid = n // 2
    frame = vr[mid].asnumpy()
    return Image.fromarray(frame).convert("RGB")


def parse_yes_no(text):
    t = (text or "").strip().lower().lstrip("\"'*`#- ")
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return -1


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
    """Scan the output jsonl for malformed lines and missing fields."""
    if not os.path.exists(out_path):
        return 0, 0
    n_total = 0
    n_ok = 0
    with open(out_path) as f:
        for line in f:
            n_total += 1
            try:
                r = json.loads(line)
                if r.get("video_id") and "pred" in r:
                    n_ok += 1
            except Exception:
                pass
    return n_ok, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--transcript-limit", type=int, default=300)
    parser.add_argument("--length-penalty", type=float, default=3.0,
                        help="BLIP-2 generation length_penalty; paper uses 3.0")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="Paper uses BLIP-2 default (~30); raise if truncation observed")
    args = parser.parse_args()
    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    datasets = ALL_DATASETS if args.all else [args.dataset]

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info(f"Pro-Cap repro: model={args.model} datasets={datasets}")

    import torch
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()
    logging.info("BLIP-2 loaded")

    def blip2_ask(image, prompt):
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(
            "cuda:0", torch.float16
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                length_penalty=args.length_penalty,
                num_beams=1,
                do_sample=False,
            )
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        return text.strip()

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

        out_dir = os.path.join(PROJECT_ROOT, "results", "procap_blip2_flan_t5_xl", ds)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{args.split}_procap.jsonl")
        done = resume_done_ids(out_path)
        if done:
            logging.info(f"[{ds}] resume: {len(done)} already done")
        remaining = [v for v in vids if v not in done]
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
                frame = load_middle_frame(media_path, media_type)
            except Exception as e:
                logging.warning(f"  {vid}: frame load failed: {e}")
                frame = None
            if frame is None:
                rec = {
                    "video_id": vid,
                    "pred": -1,
                    "error": "no_frame",
                }
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                n_skipped += 1
                continue

            # 8 probing calls
            probe_answers = {}
            for key, q in PROCAP_QUESTIONS:
                try:
                    ans = blip2_ask(frame, VQA_PROMPT_TEMPLATE.format(q=q))
                except Exception as e:
                    logging.warning(f"  {vid}/{key}: probe failed: {e}")
                    ans = ""
                probe_answers[key] = ans

            # Build concatenated caption (paper-style)
            caption_parts = []
            for key, _ in PROCAP_QUESTIONS:
                ans = probe_answers.get(key, "").strip()
                if ans:
                    caption_parts.append(f"{key}: {ans}")
            caption = "; ".join(caption_parts) if caption_parts else "(no caption)"

            # 9th probe: label-free classification head
            transcript = (ann.get("transcript", "") or "")[: args.transcript_limit]
            class_prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
                caption=caption, transcript=transcript.replace('"', "'")
            )
            try:
                verdict_text = blip2_ask(frame, class_prompt)
            except Exception as e:
                logging.error(f"  {vid}: classification probe failed: {e}")
                verdict_text = ""
            pred = parse_yes_no(verdict_text)

            rec = {
                "video_id": vid,
                "pred": pred,
                "verdict_text": verdict_text,
                "caption": caption,
                "probe_answers": probe_answers,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            n_processed += 1

            # Periodic integrity audit every 50 videos
            if n_processed % 50 == 0:
                n_ok, n_total = verify_output_integrity(out_path)
                rate = n_processed / (time.time() - t0) if (time.time() - t0) > 0 else 0
                logging.info(
                    f"  [{ds}] [{n_processed}/{len(remaining)}] "
                    f"{rate:.2f} vid/s  integrity {n_ok}/{n_total}"
                )

        logging.info(f"[{ds}] done. {n_processed} processed, {n_skipped} skipped.")

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
