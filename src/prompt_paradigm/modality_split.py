"""
prompt_paradigm v4 — Modality-Split Evidence Probes scorer.

Two video-grounded MLLM calls per video with Qwen3-VL-2B-Instruct (bf16, vLLM).
Each call operates on a DISJOINT input support (visual vs text) and asks the
SAME "violates rules?" evidence question:

  Call 1 — Visual-Evidence-Probe: frames only, no title, no transcript.
      p_vis = P(Yes)/(P(Yes)+P(No)) from position-0 logprobs.

  Call 2 — Text-Evidence-Probe: no frames, title + transcript only.
      p_txt = P(Yes)/(P(Yes)+P(No)) from position-0 logprobs.

Both calls share the same rule list (copied verbatim from frozen
src/score_holistic_2b.py YOUTUBE_RULES / BILIBILI_RULES) and the same
"Does this contain content that violates any of the above rules?" question
(copied verbatim from frozen BINARY_PROMPT). Only the observation line is
adapted per specialist since each call sees a different input subset, and
misstating what the model is observing would be factually incorrect in the
prompt. The rules and question text are unchanged.

Fusion is NOT performed here; the scorer persists raw (p_vis, p_txt) per
video. eval_modality.py computes rank-space noisy-OR against the train-split
reference distribution and all the ablations (A/B/C/D/E).

Output JSONL (contract with src/quick_eval_all.py::load_scores_file uses
"score" — here we also write a placeholder `score = 1 - (1 - p_vis)*(1 - p_txt)`
for the simple raw-prob noisy-OR, but the evaluator computes its own
rank-space score from the raw p_vis / p_txt columns):

    {"video_id": str, "p_visual": float, "p_text": float, "score": float}

Usage (always via sbatch, never direct on login node):
  sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh \
    && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 \
    && python src/prompt_paradigm/modality_split.py --dataset MHClip_EN --split test"
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


# --- Rules copied verbatim from src/score_holistic_2b.py:27-49 (frozen) ---
CONSTITUTION_MAP = {
    "MHClip_EN": "youtube",
    "MHClip_ZH": "bilibili",
}

YOUTUBE_RULES = """1. Encourage violence against groups based on protected status
2. Incite hatred against groups based on protected status
3. Dehumanize groups by comparing to non-human entities
4. Praise or glorify violence against protected groups
5. Use slurs and stereotypes that incite hatred (including through humor, irony, or coded language)
6. Claim groups are inferior based on protected status
7. Promote hateful supremacism
8. Make conspiratorial claims about groups based on protected status
9. Deny or minimize major violent events"""

BILIBILI_RULES = """1. Discriminatory content (violence, slurs, exclusion based on protected status)
2. Attacks, disparagement, or belittlement targeting groups by protected attribute
3. Verbal abuse and personal attacks based on identity
4. Displaying or promoting hateful ideologies/symbols
5. Mocking death, sickness, or disability
6. Making jokes about disasters or tragedies
7. Gender-based hate speech or discrimination
8. Inciting confrontation or conflict between groups"""


# --- Call 1: Visual-Evidence-Probe (frames-only) ---
# The rules and the "Does this contain content that violates any of the above
# rules?" question are copied verbatim from frozen BINARY_PROMPT. The
# observation line is adapted to reflect the frames-only input support.
VISUAL_SYSTEM = "You are a content moderation analyst."
VISUAL_USER = """Observe this video's visual frames.

Based on this platform's hate speech policy, the following types of content are prohibited:
{rules}

Does this video contain content that violates any of the above rules?

Answer "Yes" or "No"."""


# --- Call 2: Text-Evidence-Probe (title+transcript only) ---
# Same rules and same question, observation line reflects text-only input.
TEXT_SYSTEM = "You are a content moderation analyst."
TEXT_USER = """Observe the following video title and transcript.

Title: {title}
Transcript: {transcript}

Based on this platform's hate speech policy, the following types of content are prohibited:
{rules}

Does this video contain content that violates any of the above rules?

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


def build_visual_media_content(media_path, media_type):
    """Video-only content: frames or video URL."""
    if media_type == "video":
        return [{"type": "video_url", "video_url": {"url": f"file://{media_path}"}}]
    jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")))
    if len(jpgs) > 8:
        indices = np.linspace(0, len(jpgs) - 1, 8, dtype=int)
        jpgs = [jpgs[i] for i in indices]
    return [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in jpgs]


def build_visual_messages(media_path, media_type, rules_text):
    user_text = VISUAL_USER.format(rules=rules_text)
    media_content = build_visual_media_content(media_path, media_type)
    content = media_content + [{"type": "text", "text": user_text}]
    return [
        {"role": "system", "content": VISUAL_SYSTEM},
        {"role": "user", "content": content},
    ]


def build_text_messages(ann, transcript_limit, rules_text):
    title = ann.get("title", "") or ""
    transcript = (ann.get("transcript", "") or "")[:transcript_limit]
    user_text = TEXT_USER.format(title=title, transcript=transcript, rules=rules_text)
    return [
        {"role": "system", "content": TEXT_SYSTEM},
        {"role": "user", "content": user_text},
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
                log_dir, f"prompt_paradigm_modality_{args.dataset}_{args.split}.log")),
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

    platform = CONSTITUTION_MAP[args.dataset]
    rules_text = YOUTUBE_RULES if platform == "youtube" else BILIBILI_RULES

    out_dir = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}_modality.jsonl")

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

        visual_messages = []
        text_messages = []
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

            v_msgs = build_visual_messages(media_path, media_type, rules_text)
            t_msgs = build_text_messages(ann, args.transcript_limit, rules_text)

            visual_messages.append(v_msgs)
            text_messages.append(t_msgs)
            batch_vid_ids.append(vid_id)

        if not batch_vid_ids:
            continue

        # --- Call 1: Visual-Evidence-Probe ---
        p_visuals = [None] * len(batch_vid_ids)
        try:
            v_outputs = llm.chat(messages=visual_messages, sampling_params=binary_params)
            for i, out in enumerate(v_outputs):
                p_visuals[i] = extract_binary_score(out, label_token_ids)
        except Exception as e:
            logging.error(f"  Visual batch failed: {e}, falling back to single")
            for i, msgs in enumerate(visual_messages):
                try:
                    out_single = llm.chat(messages=[msgs], sampling_params=binary_params)
                    p_visuals[i] = extract_binary_score(out_single[0], label_token_ids)
                except Exception as e2:
                    err2 = str(e2)
                    if "longer than the maximum model length" in err2 or "max_model_len" in err2:
                        logging.warning(f"  {batch_vid_ids[i]}: visual SKIPPED oversize")
                    else:
                        logging.error(f"  {batch_vid_ids[i]}: visual single failed: {err2}")
                    p_visuals[i] = None

        # --- Call 2: Text-Evidence-Probe ---
        p_texts = [None] * len(batch_vid_ids)
        try:
            t_outputs = llm.chat(messages=text_messages, sampling_params=binary_params)
            for i, out in enumerate(t_outputs):
                p_texts[i] = extract_binary_score(out, label_token_ids)
        except Exception as e:
            logging.error(f"  Text batch failed: {e}, falling back to single")
            for i, msgs in enumerate(text_messages):
                try:
                    out_single = llm.chat(messages=[msgs], sampling_params=binary_params)
                    p_texts[i] = extract_binary_score(out_single[0], label_token_ids)
                except Exception as e2:
                    err2 = str(e2)
                    if "longer than the maximum model length" in err2 or "max_model_len" in err2:
                        logging.warning(f"  {batch_vid_ids[i]}: text SKIPPED oversize")
                    else:
                        logging.error(f"  {batch_vid_ids[i]}: text single failed: {err2}")
                    p_texts[i] = None

        # --- Persist ---
        with open(out_path, "a") as f:
            for i in range(len(batch_vid_ids)):
                p_v = p_visuals[i]
                p_t = p_texts[i]
                if p_v is None or p_t is None:
                    rec = {
                        "video_id": batch_vid_ids[i],
                        "p_visual": p_v,
                        "p_text": p_t,
                        "score": None,
                        "skipped": True,
                    }
                else:
                    raw_nor = 1.0 - (1.0 - p_v) * (1.0 - p_t)
                    rec = {
                        "video_id": batch_vid_ids[i],
                        "p_visual": p_v,
                        "p_text": p_t,
                        "score": raw_nor,
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
