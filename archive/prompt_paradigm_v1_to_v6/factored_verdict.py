"""
prompt_paradigm v2 — Factored Verdict scorer.

Two distinct-role MLLM calls per video with Qwen3-VL-2B-Instruct (bf16, vLLM).
Both calls read the RAW media (frames + title + transcript). No text flows
between the two calls — this is parallel factorization, not sequential
decomposition.

  Call 1 — Target-Detector: perception/existence query.
           "Does this video identify/depict/mention a protected-identity group?"
           P_T = P(Yes)/(P(Yes)+P(No)) from single-token logprobs.

  Call 2 — Stance-Judge: affective valence query.
           "Does this video take a derogatory/mocking/hostile stance toward its
            subject?" (ignores whether the subject is a protected group.)
           P_S = P(Yes)/(P(Yes)+P(No)) from single-token logprobs.

Final score = P_T * P_S (AND-semantics factorization).

Output JSONL: {"video_id": str, "p_target": float, "p_stance": float, "score": float}
matching the contract read by src/quick_eval_all.py::load_scores_file (uses "score").

Usage (always via sbatch):
  sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh \
    && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 \
    && python src/prompt_paradigm/factored_verdict.py --dataset MHClip_EN --split test"
"""

import argparse
import glob as globmod
import json
import logging
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_utils import DATASET_ROOTS, get_media_path, load_annotations

PROJECT_ROOT = "/data/jehc223/EMNLP2"


# --- Call 1: Target-Detector (protected-identity group present?) ---
TARGET_SYSTEM = (
    "You analyze video content. You identify whether a video concerns a "
    "protected-identity group. You answer strictly Yes or No based on "
    "observable evidence in the video, title, and transcript."
)

TARGET_USER = """Video title: {title}
Transcript: {transcript}

Does this video identify, depict, mention, target, or focus on a protected-identity group?

Protected-identity groups include people grouped by: race, ethnicity, nationality, religion, gender, sexual orientation, gender identity, disability, age, or any group defined by these attributes.

- Yes: the video is about, mentions, targets, depicts, or centrally involves such a group.
- No: the video is about something else (e.g., products, pets, animals, entertainment without identity targeting, personal stories unrelated to protected groups, generic opinions, abstract topics).

Answer "Yes" or "No"."""


# --- Call 2: Stance-Judge (derogatory stance toward the subject?) ---
STANCE_SYSTEM = (
    "You analyze video content. You identify the affective stance a video "
    "takes toward its subject, on a derogatory versus non-derogatory axis. "
    "You answer strictly Yes or No based on observable evidence in the video, "
    "title, and transcript."
)

STANCE_USER = """Video title: {title}
Transcript: {transcript}

Does this video take a derogatory, mocking, dehumanizing, disparaging, or hostile stance toward any person or group it depicts or discusses?

Consider only the stance/attitude of the video, not whether its subject is a protected group:
- Yes: the video uses slurs, dehumanization, hostile mockery, contempt, insults, humiliation, or derogatory framing toward any person or group.
- No: the video is neutral, factual, admiring, celebratory, affectionate, descriptive, or otherwise non-derogatory — even if it discusses a sensitive topic.

Answer "Yes" or "No"."""


def build_binary_token_ids(tokenizer):
    def first_tok(s):
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if ids else None

    mapping = {}
    for label in ["Yes", "No"]:
        tids = set()
        for variant in [label, f" {label}", label.lower(), f" {label.lower()}",
                        label.upper(), f" {label.upper()}"]:
            tid = first_tok(variant)
            if tid is not None:
                tids.add(tid)
        mapping[label] = list(tids)
        decoded = [tokenizer.decode([t]) for t in mapping[label]]
        logging.info(f"  Binary label '{label}' -> token IDs {mapping[label]} ({decoded})")
    return mapping


def extract_binary_score(output, label_token_ids):
    if not output or not output.outputs:
        return None
    gen = output.outputs[0]
    if not gen.logprobs or len(gen.logprobs) == 0:
        return None

    pos0 = gen.logprobs[0]
    FALLBACK = -30.0

    yes_exps = []
    for tid in label_token_ids["Yes"]:
        lp = pos0[tid].logprob if tid in pos0 else FALLBACK
        yes_exps.append(math.exp(lp))

    no_exps = []
    for tid in label_token_ids["No"]:
        lp = pos0[tid].logprob if tid in pos0 else FALLBACK
        no_exps.append(math.exp(lp))

    p_yes = sum(yes_exps)
    p_no = sum(no_exps)
    total = p_yes + p_no
    if total <= 0:
        return None
    return p_yes / total


def build_media_content(media_path, media_type):
    if media_type == "video":
        return [{"type": "video_url", "video_url": {"url": f"file://{media_path}"}}]
    jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")))
    if len(jpgs) > 8:
        indices = np.linspace(0, len(jpgs) - 1, 8, dtype=int)
        jpgs = [jpgs[i] for i in indices]
    return [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in jpgs]


def build_messages(vid_id, ann, media_path, media_type, transcript_limit,
                   system_prompt, user_template):
    title = ann.get("title", "") or ""
    transcript = (ann.get("transcript", "") or "")[:transcript_limit]
    user_text = user_template.format(title=title, transcript=transcript)
    media_content = build_media_content(media_path, media_type)
    content = media_content + [{"type": "text", "text": user_text}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["MHClip_EN", "MHClip_ZH"])
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--transcript-limit", type=int, default=300)
    args = parser.parse_args()

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(
                log_dir, f"prompt_paradigm_factored_{args.dataset}_{args.split}.log")),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Config: dataset={args.dataset} split={args.split} model={args.model}")

    annotations = load_annotations(args.dataset)
    root = DATASET_ROOTS[args.dataset]
    split_path = os.path.join(root, "splits", f"{args.split}_clean.csv")
    if not os.path.isfile(split_path):
        from data_utils import generate_clean_splits
        generate_clean_splits(args.dataset)
    with open(split_path) as f:
        split_ids = [line.strip() for line in f if line.strip()]
    logging.info(f"Clean {args.split} split: {len(split_ids)} videos")

    out_dir = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}_factored.jsonl")

    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    if r.get("video_id"):
                        done_ids.add(r["video_id"])
                except json.JSONDecodeError:
                    pass
        logging.info(f"Resuming: {len(done_ids)} already done")

    remaining = [v for v in split_ids if v not in done_ids]
    if not remaining:
        logging.info("All scored.")
        return

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_model_len=32768,
        limit_mm_per_prompt={"video": 1, "image": 8},
        allowed_local_media_path="/data/jehc223",
        mm_processor_kwargs={"max_pixels": 100352},
    )
    tokenizer = llm.get_tokenizer()

    label_token_ids = build_binary_token_ids(tokenizer)
    all_yesno_ids = set()
    for tids in label_token_ids.values():
        all_yesno_ids.update(tids)

    binary_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=list(all_yesno_ids),
    )

    t0 = time.time()
    n_processed = 0
    n_skipped = 0

    for batch_start in range(0, len(remaining), args.batch_size):
        batch_ids = remaining[batch_start:batch_start + args.batch_size]

        target_messages = []
        stance_messages = []
        batch_vid_ids = []

        for vid_id in batch_ids:
            ann = annotations.get(vid_id)
            if ann is None:
                logging.warning(f"  {vid_id}: not in annotations, skipping")
                n_skipped += 1
                continue
            media = get_media_path(vid_id, args.dataset)
            if media is None:
                logging.warning(f"  {vid_id}: no media, skipping")
                n_skipped += 1
                continue
            media_path, media_type = media

            t_msgs = build_messages(
                vid_id, ann, media_path, media_type, args.transcript_limit,
                TARGET_SYSTEM, TARGET_USER,
            )
            s_msgs = build_messages(
                vid_id, ann, media_path, media_type, args.transcript_limit,
                STANCE_SYSTEM, STANCE_USER,
            )

            target_messages.append(t_msgs)
            stance_messages.append(s_msgs)
            batch_vid_ids.append(vid_id)

        if not batch_vid_ids:
            continue

        # --- Call 1: Target-Detector ---
        p_targets = [None] * len(batch_vid_ids)
        try:
            t_outputs = llm.chat(messages=target_messages, sampling_params=binary_params)
            for i, out in enumerate(t_outputs):
                p_targets[i] = extract_binary_score(out, label_token_ids)
        except Exception as e:
            err_msg = str(e)
            logging.error(f"  Target batch failed: {err_msg}, falling back to single")
            for i, msgs in enumerate(target_messages):
                try:
                    out_single = llm.chat(messages=[msgs], sampling_params=binary_params)
                    p_targets[i] = extract_binary_score(out_single[0], label_token_ids)
                except Exception as e2:
                    err2 = str(e2)
                    if "longer than the maximum model length" in err2 or "max_model_len" in err2:
                        logging.warning(f"  {batch_vid_ids[i]}: target SKIPPED oversize")
                    else:
                        logging.error(f"  {batch_vid_ids[i]}: target single failed: {err2}")
                    p_targets[i] = None

        # --- Call 2: Stance-Judge ---
        p_stances = [None] * len(batch_vid_ids)
        try:
            s_outputs = llm.chat(messages=stance_messages, sampling_params=binary_params)
            for i, out in enumerate(s_outputs):
                p_stances[i] = extract_binary_score(out, label_token_ids)
        except Exception as e:
            err_msg = str(e)
            logging.error(f"  Stance batch failed: {err_msg}, falling back to single")
            for i, msgs in enumerate(stance_messages):
                try:
                    out_single = llm.chat(messages=[msgs], sampling_params=binary_params)
                    p_stances[i] = extract_binary_score(out_single[0], label_token_ids)
                except Exception as e2:
                    err2 = str(e2)
                    if "longer than the maximum model length" in err2 or "max_model_len" in err2:
                        logging.warning(f"  {batch_vid_ids[i]}: stance SKIPPED oversize")
                    else:
                        logging.error(f"  {batch_vid_ids[i]}: stance single failed: {err2}")
                    p_stances[i] = None

        # --- Persist ---
        with open(out_path, "a") as f:
            for i in range(len(batch_vid_ids)):
                p_t = p_targets[i]
                p_s = p_stances[i]
                if p_t is None or p_s is None:
                    rec = {
                        "video_id": batch_vid_ids[i],
                        "p_target": p_t,
                        "p_stance": p_s,
                        "score": None,
                        "skipped": True,
                    }
                else:
                    rec = {
                        "video_id": batch_vid_ids[i],
                        "p_target": p_t,
                        "p_stance": p_s,
                        "score": p_t * p_s,
                    }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        n_processed += len(batch_vid_ids)
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        logging.info(
            f"  [{len(done_ids)+n_processed}/{len(split_ids)}] rate={rate:.1f} vid/s"
        )

    logging.info(f"Done. scored={n_processed} skipped={n_skipped} out={out_path}")


if __name__ == "__main__":
    main()
