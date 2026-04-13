"""
prompt_paradigm v3 — Polarity-Calibrated Probes scorer.

Two video-grounded MLLM calls per video with Qwen3-VL-2B-Instruct (bf16, vLLM).
Both calls read the RAW media (frames + title + transcript). No text flows
between the two calls.

  Call 1 — Evidence-Probe (affirmative framing):
    "Does this video contain content that violates any of the above rules?"
    p_E = P(Yes)/(P(Yes)+P(No)) from single-token logprobs.
    FN-biased (conservative under RLHF refusal bias on violates framing).

  Call 2 — Compliance-Probe (negated framing):
    "Is this video fully consistent with all of the above community guidelines?"
    p_C = P(Yes)/(P(Yes)+P(No)) from single-token logprobs.
    After negation (1 - p_C), FP-biased (RLHF refusal bias on consistent framing).

Fusion (logit-space bias cancellation, NOT probability-space average):

    L_E = logit(p_E); L_C = logit(p_C)
    L_fused = 0.5 * (L_E - L_C)      # negation of Call 2 enters as -L_C
    score   = sigmoid(L_fused)

Output JSONL (contract with src/quick_eval_all.py::load_scores_file uses "score"):
    {"video_id": str, "p_evidence": float, "p_compliance": float, "score": float}

Both prompt templates are COPIED VERBATIM from frozen src/score_holistic_2b.py
lines 52-62 (BINARY_PROMPT) and 64-74 (DEFLECTED_BINARY_PROMPT) to keep the
frozen file untouched per CLAUDE.md.

Usage (always via sbatch, never direct on login node):
  sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh \
    && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 \
    && python src/prompt_paradigm/polarity_calibration.py --dataset MHClip_EN --split test"
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


# --- Rules text copied from src/score_holistic_2b.py:32-49 (frozen) ---
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


# --- Call 1: Evidence-Probe (copied from src/score_holistic_2b.py:52-62 frozen) ---
EVIDENCE_SYSTEM = ("You are a content moderation analyst. "
                   "Answer based strictly on observable evidence.")
EVIDENCE_USER = """Observe this video's visual frames, title, and transcript.

Title: {title}
Transcript: {transcript}

Based on this platform's hate speech policy, the following types of content are prohibited:
{rules}

Does this video contain content that violates any of the above rules?

Answer "Yes" or "No"."""


# --- Call 2: Compliance-Probe (copied from src/score_holistic_2b.py:64-74 frozen) ---
COMPLIANCE_SYSTEM = ("You are a content moderation analyst. "
                     "Answer based strictly on observable evidence.")
COMPLIANCE_USER = """Observe this video's visual frames, title, and transcript.

Title: {title}
Transcript: {transcript}

This platform's community guidelines require that content must NOT:
{rules}

Is this video fully consistent with all of the above community guidelines?

Answer "Yes" or "No"."""


def build_binary_token_ids(tokenizer):
    """Map Yes/No labels to first-token IDs, accepting capitalization/space variants."""
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
    """Extract P(Yes)/(P(Yes)+P(No)) from position-0 logprobs. Returns None on failure."""
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
    """Build vLLM content list for video or frames (up to 8 sampled frames)."""
    if media_type == "video":
        return [{"type": "video_url", "video_url": {"url": f"file://{media_path}"}}]
    jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")))
    if len(jpgs) > 8:
        indices = np.linspace(0, len(jpgs) - 1, 8, dtype=int)
        jpgs = [jpgs[i] for i in indices]
    return [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in jpgs]


def build_messages(ann, media_path, media_type, transcript_limit,
                   system_prompt, user_template, rules_text):
    title = ann.get("title", "") or ""
    transcript = (ann.get("transcript", "") or "")[:transcript_limit]
    user_text = user_template.format(title=title, transcript=transcript, rules=rules_text)
    media_content = build_media_content(media_path, media_type)
    content = media_content + [{"type": "text", "text": user_text}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def fuse_polarity(p_evidence, p_compliance, eps=1e-6):
    """Logit-space bias-cancelling fusion.

        L_E      = logit(p_evidence)
        L_C      = logit(p_compliance)
        L_fused  = 0.5 * (L_E - L_C)       # negation of Call 2 enters as -L_C
        score    = sigmoid(L_fused)

    Returns None if either input is None."""
    if p_evidence is None or p_compliance is None:
        return None
    p_e = min(max(p_evidence, eps), 1.0 - eps)
    p_c = min(max(p_compliance, eps), 1.0 - eps)
    l_e = math.log(p_e / (1.0 - p_e))
    l_c = math.log(p_c / (1.0 - p_c))
    l_fused = 0.5 * (l_e - l_c)
    return 1.0 / (1.0 + math.exp(-l_fused))


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
                log_dir, f"prompt_paradigm_polarity_{args.dataset}_{args.split}.log")),
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
    out_path = os.path.join(out_dir, f"{args.split}_polarity.jsonl")

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

        evidence_messages = []
        compliance_messages = []
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

            e_msgs = build_messages(
                ann, media_path, media_type, args.transcript_limit,
                EVIDENCE_SYSTEM, EVIDENCE_USER, rules_text,
            )
            c_msgs = build_messages(
                ann, media_path, media_type, args.transcript_limit,
                COMPLIANCE_SYSTEM, COMPLIANCE_USER, rules_text,
            )

            evidence_messages.append(e_msgs)
            compliance_messages.append(c_msgs)
            batch_vid_ids.append(vid_id)

        if not batch_vid_ids:
            continue

        # --- Call 1: Evidence-Probe ---
        p_evidences = [None] * len(batch_vid_ids)
        try:
            e_outputs = llm.chat(messages=evidence_messages, sampling_params=binary_params)
            for i, out in enumerate(e_outputs):
                p_evidences[i] = extract_binary_score(out, label_token_ids)
        except Exception as e:
            logging.error(f"  Evidence batch failed: {e}, falling back to single")
            for i, msgs in enumerate(evidence_messages):
                try:
                    out_single = llm.chat(messages=[msgs], sampling_params=binary_params)
                    p_evidences[i] = extract_binary_score(out_single[0], label_token_ids)
                except Exception as e2:
                    err2 = str(e2)
                    if "longer than the maximum model length" in err2 or "max_model_len" in err2:
                        logging.warning(f"  {batch_vid_ids[i]}: evidence SKIPPED oversize")
                    else:
                        logging.error(f"  {batch_vid_ids[i]}: evidence single failed: {err2}")
                    p_evidences[i] = None

        # --- Call 2: Compliance-Probe ---
        p_compliances = [None] * len(batch_vid_ids)
        try:
            c_outputs = llm.chat(messages=compliance_messages, sampling_params=binary_params)
            for i, out in enumerate(c_outputs):
                p_compliances[i] = extract_binary_score(out, label_token_ids)
        except Exception as e:
            logging.error(f"  Compliance batch failed: {e}, falling back to single")
            for i, msgs in enumerate(compliance_messages):
                try:
                    out_single = llm.chat(messages=[msgs], sampling_params=binary_params)
                    p_compliances[i] = extract_binary_score(out_single[0], label_token_ids)
                except Exception as e2:
                    err2 = str(e2)
                    if "longer than the maximum model length" in err2 or "max_model_len" in err2:
                        logging.warning(f"  {batch_vid_ids[i]}: compliance SKIPPED oversize")
                    else:
                        logging.error(f"  {batch_vid_ids[i]}: compliance single failed: {err2}")
                    p_compliances[i] = None

        # --- Persist ---
        with open(out_path, "a") as f:
            for i in range(len(batch_vid_ids)):
                p_e = p_evidences[i]
                p_c = p_compliances[i]
                score = fuse_polarity(p_e, p_c)
                if score is None:
                    rec = {
                        "video_id": batch_vid_ids[i],
                        "p_evidence": p_e,
                        "p_compliance": p_c,
                        "score": None,
                        "skipped": True,
                    }
                else:
                    rec = {
                        "video_id": batch_vid_ids[i],
                        "p_evidence": p_e,
                        "p_compliance": p_c,
                        "score": score,
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
