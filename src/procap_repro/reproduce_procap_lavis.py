"""Pro-Cap faithful reproduction (LAVIS version).

Upstream: Cao et al., ACM MM 2023, arxiv 2308.08088
Repo: https://github.com/Social-AI-Studio/Pro-Cap
Source of truth: external_repos/procap/codes/Pro-Cap-Generation.ipynb

This script is the paper-faithful rewrite that replaces the earlier
HuggingFace `Blip2ForConditionalGeneration` port. Upstream's notebook
uses LAVIS `load_model_and_preprocess` with `name="blip2_t5"` and
`model_type="caption_coco_flant5xl"`, with a `.float()` upcast and a
`length_penalty=3.0` generation kwarg. Those are reproduced here
verbatim.

Label-free 9th probe: we substitute upstream's supervised RoBERTa
classifier with a single BLIP-2 yes/no call that takes the 8 VQA
answers plus the video transcript as textual context (see README).
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

# 8 VQA probing questions from Pro-Cap-Generation.ipynb (Cao et al.),
# in upstream order. Strings copied verbatim from the notebook cells.
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

# Label-free 9th probe: classification with concatenated captions + transcript.
CLASSIFICATION_PROMPT_TEMPLATE = (
    "The image has been described as follows: {captions}. "
    "The accompanying transcript says: \"{transcript}\". "
    "Is this content promoting hateful, discriminatory, or "
    "dehumanizing stereotypes against any identity group? "
    "Answer yes or no."
)


def load_middle_frame(media_path, media_type):
    from PIL import Image
    if media_type == "frames":
        import glob as globmod
        jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")))
        if not jpgs:
            return None
        return Image.open(jpgs[len(jpgs) // 2]).convert("RGB")
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
                    and "pred" in r
                    and "verdict_text" in r
                    and "probe_answers" in r
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
    parser.add_argument("--transcript-limit", type=int, default=1000)
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
    logging.info(f"Pro-Cap LAVIS repro: datasets={datasets}")

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

        out_dir = os.path.join(
            PROJECT_ROOT, "results", "procap_lavis_blip2_flan_t5_xl", ds
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{args.split}_procap.jsonl")
        done = resume_done_ids(out_path)
        if done:
            logging.info(f"[{ds}] resume: {len(done)} already done")
        remaining = [v for v in vids if v not in done]
        if args.limit and args.limit > 0:
            remaining = remaining[: args.limit]
            logging.info(f"[{ds}] --limit {args.limit} (smoke test)")
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
                rec = {"video_id": vid, "pred": -1, "error": "no_frame"}
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                n_skipped += 1
                continue

            probe_answers = {}
            for key, q in PROCAP_QUESTIONS:
                try:
                    ans = generate_prompt_result(frame, q)
                except Exception as e:
                    logging.warning(f"  {vid}/{key}: probe failed: {e}")
                    ans = ""
                probe_answers[key] = ans
                logging.info(f"  {vid} probe[{key}]: {ans}")

            captions_concat = "; ".join(
                probe_answers.get(k, "") for k, _ in PROCAP_QUESTIONS
            )
            transcript = (ann.get("transcript", "") or "")[: args.transcript_limit]
            class_question = CLASSIFICATION_PROMPT_TEMPLATE.format(
                captions=captions_concat,
                transcript=transcript.replace('"', "'"),
            )
            try:
                verdict_text = generate_prompt_result(frame, class_question)
            except Exception as e:
                logging.error(f"  {vid}: classification probe failed: {e}")
                verdict_text = ""
            pred = parse_yes_no(verdict_text)
            logging.info(f"  {vid} verdict: {verdict_text!r} -> pred={pred}")

            rec = {
                "video_id": vid,
                "pred": pred,
                "verdict_text": verdict_text,
                "probe_answers": probe_answers,
                "caption": captions_concat,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            n_processed += 1

            if n_processed % 50 == 0:
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
