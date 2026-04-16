"""
Naive 2B text-output baseline.

A minimal zero-shot prompt asking "Is this video hateful?" with title and
transcript, producing a plain-text Yes/No response (NOT logprobs). Saves the
decoded string and the parsed prediction to jsonl.

Prompt (single-turn, no system message, no policy, no role):

    Is this video hateful?
    Title: {title}
    Transcript: {transcript}

    Only respond Yes or No.

Usage:
  python src/naive_baseline/score_naive_2b.py --dataset MHClip_EN
  python src/naive_baseline/score_naive_2b.py --all
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

# Import data_utils from sibling src/our_method/
_OUR_METHOD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "our_method")
sys.path.insert(0, _OUR_METHOD)
from data_utils import DATASET_ROOTS, get_media_path, load_annotations  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]


def output_root_for_model(model_id, override_tag=None):
    """Pick the output parent directory based on the model ID.

    Defaults to ``results/naive_2b`` for Qwen3-VL-2B (back-compat) and
    auto-tags other families so results land in
    ``results/naive_<tag>/<ds>/<split>_naive.jsonl``.
    """
    if override_tag:
        return os.path.join(PROJECT_ROOT, "results", f"naive_{override_tag}")
    name = model_id.lower()
    if "qwen3-vl-2b" in name:
        return os.path.join(PROJECT_ROOT, "results", "naive_2b")
    if "qwen3-vl-8b" in name:
        return os.path.join(PROJECT_ROOT, "results", "naive_8b")
    if "llava-onevision" in name or "llava_onevision" in name:
        sz = "7b" if "7b" in name else "0.5b" if "0.5b" in name else "unk"
        return os.path.join(PROJECT_ROOT, "results", f"naive_llava_ov_{sz}")
    if "internvl3" in name:
        for sz in ("8b", "14b", "78b", "38b", "2b", "1b"):
            if sz in name:
                return os.path.join(PROJECT_ROOT, "results", f"naive_internvl3_{sz}")
        return os.path.join(PROJECT_ROOT, "results", "naive_internvl3_unk")
    if "gemma" in name:
        for sz in ("4b", "12b", "27b"):
            if sz in name:
                return os.path.join(PROJECT_ROOT, "results", f"naive_gemma3_vl_{sz}")
        return os.path.join(PROJECT_ROOT, "results", "naive_gemma3_vl_unk")
    if "qwen2.5-vl" in name or "qwen2_5-vl" in name:
        for sz in ("7b", "32b", "72b", "3b"):
            if sz in name:
                return os.path.join(PROJECT_ROOT, "results", f"naive_qwen25_vl_{sz}")
        return os.path.join(PROJECT_ROOT, "results", "naive_qwen25_vl_unk")
    safe = name.replace("/", "_").replace("-", "_")
    return os.path.join(PROJECT_ROOT, "results", f"naive_{safe}")

NAIVE_PROMPT = """Is this video hateful?
Title: {title}
Transcript: {transcript}

Only respond Yes or No."""


def build_media_content(media_path, media_type):
    if media_type == "video":
        return [{"type": "video_url", "video_url": {"url": f"file://{media_path}"}}]
    import glob as globmod
    jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")))
    if len(jpgs) > 8:
        indices = np.linspace(0, len(jpgs) - 1, 8, dtype=int)
        jpgs = [jpgs[i] for i in indices]
    return [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in jpgs]


def parse_yes_no(text):
    """Return 1 if response starts with 'yes', 0 if 'no', -1 otherwise."""
    t = (text or "").strip().lower()
    t = t.lstrip("\"'*`#- ")
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return -1


def load_split_ids(dataset, split):
    root = DATASET_ROOTS[dataset]
    split_path = os.path.join(root, "splits", f"{split}_clean.csv")
    if not os.path.isfile(split_path):
        from data_utils import generate_clean_splits
        generate_clean_splits(dataset)
    with open(split_path) as f:
        return [line.strip() for line in f if line.strip()]


def resume_done_ids(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("video_id"):
                        done.add(r["video_id"])
                except json.JSONDecodeError:
                    pass
    return done


def score_dataset(dataset, split, llm, sampling_params, batch_size, transcript_limit, out_root):
    annotations = load_annotations(dataset)
    split_ids = load_split_ids(dataset, split)
    logging.info(f"[{dataset}] {len(split_ids)} videos in {split}_clean")

    out_dir = os.path.join(out_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}_naive.jsonl")

    done_ids = resume_done_ids(out_path)
    if done_ids:
        logging.info(f"[{dataset}] Resume: {len(done_ids)} already scored")
    remaining = [v for v in split_ids if v not in done_ids]
    if not remaining:
        logging.info(f"[{dataset}] all done")
        return out_path

    t0 = time.time()
    n_processed = 0
    n_skipped = 0

    for batch_start in range(0, len(remaining), batch_size):
        batch_ids = remaining[batch_start:batch_start + batch_size]
        batch_messages = []
        batch_vid_ids = []
        for vid_id in batch_ids:
            ann = annotations.get(vid_id)
            if ann is None:
                logging.warning(f"  {vid_id}: not in annotations, skipping")
                n_skipped += 1
                continue
            media = get_media_path(vid_id, dataset)
            if media is None:
                logging.warning(f"  {vid_id}: no media, skipping")
                n_skipped += 1
                continue
            media_path, media_type = media
            title = ann.get("title", "") or ""
            transcript = (ann.get("transcript", "") or "")[:transcript_limit]
            prompt_text = NAIVE_PROMPT.format(title=title, transcript=transcript)
            media_content = build_media_content(media_path, media_type)
            content = media_content + [{"type": "text", "text": prompt_text}]
            messages = [{"role": "user", "content": content}]
            batch_messages.append(messages)
            batch_vid_ids.append(vid_id)

        if not batch_messages:
            continue

        def _emit(vid_id, response_text, unparsable_reason=None):
            pred = parse_yes_no(response_text)
            rec = {
                "video_id": vid_id,
                "response_text": response_text,
                "pred": pred,
            }
            if unparsable_reason:
                rec["error"] = unparsable_reason
            return rec

        try:
            outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params)
        except Exception as e:
            err_msg = str(e)
            logging.error(f"  Batch failed: {err_msg}, falling back to single")
            with open(out_path, "a") as f:
                for i, msgs in enumerate(batch_messages):
                    try:
                        out_single = llm.chat(messages=[msgs], sampling_params=sampling_params)
                        text = out_single[0].outputs[0].text
                        rec = _emit(batch_vid_ids[i], text)
                    except Exception as e2:
                        err2_msg = str(e2)
                        if "longer than the maximum model length" in err2_msg or "max_model_len" in err2_msg:
                            logging.warning(f"  {batch_vid_ids[i]}: SKIPPED (prompt too long): {err2_msg[:120]}")
                        else:
                            logging.error(f"  {batch_vid_ids[i]}: single failed: {err2_msg}")
                        rec = _emit(batch_vid_ids[i], "", unparsable_reason=err2_msg[:200])
                        n_skipped += 1
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    n_processed += 1
            continue

        with open(out_path, "a") as f:
            for i, output in enumerate(outputs):
                text = output.outputs[0].text
                rec = _emit(batch_vid_ids[i], text)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        n_processed += len(batch_vid_ids)
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        logging.info(f"  [{dataset}] [{len(done_ids)+n_processed}/{len(split_ids)}] {rate:.1f} vid/s")

    logging.info(f"[{dataset}] done. {n_processed} scored, {n_skipped} skipped.")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Naive 2B text-output baseline")
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true", help="Run every dataset in one vLLM session")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--transcript-limit", type=int, default=300)
    parser.add_argument("--out-tag", default=None,
                        help="Override output directory tag: "
                             "results/naive_<tag>/<ds>/<split>_naive.jsonl")
    parser.add_argument("--max-model-len", type=int, default=32768)
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Provide --dataset or --all")

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "naive_2b.log")),
            logging.StreamHandler(),
        ],
    )

    datasets = ALL_DATASETS if args.all else [args.dataset]
    logging.info(f"Naive 2B baseline: datasets={datasets} split={args.split} model={args.model}")

    from vllm import LLM, SamplingParams

    # mm_processor_kwargs is model-family specific; pass only what the
    # HF processor accepts to avoid TypeError at init.
    name_l = args.model.lower()
    if "qwen" in name_l and ("vl" in name_l or "omni" in name_l):
        mm_kwargs = {"max_pixels": 100352}
    elif "internvl" in name_l:
        mm_kwargs = {"max_dynamic_patch": 4}
    else:
        mm_kwargs = {}

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"video": 1, "image": 8},
        allowed_local_media_path="/data/jehc223",
        mm_processor_kwargs=mm_kwargs,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=8)

    out_root = output_root_for_model(args.model, args.out_tag)
    logging.info(f"Output root: {out_root}")

    for ds in datasets:
        score_dataset(ds, args.split, llm, sampling_params, args.batch_size,
                      args.transcript_limit, out_root)

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
