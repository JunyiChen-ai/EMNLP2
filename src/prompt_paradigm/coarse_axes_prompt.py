"""
prompt_paradigm v6 — Coarse Axes Prompt (CAP) scorer.

Single call to Qwen3-VL-2B-Instruct (bf16, vLLM). Input-side manipulation:
replace the verbatim 9-rule (YouTube) / 8-rule (Bilibili) rule block inside
the frozen BINARY_PROMPT with a platform-derived 2-axis policy statement
("target x hostility"). Everything else identical to the frozen 2B baseline:
same media construction, same question wording ("Does this video contain
content that violates any of the above rules?"), same Yes/No extraction
from position-0 logprobs, same threshold family downstream.

Two conditions selectable via CLI:
  --condition axes     (default): use YOUTUBE_COARSE_AXES / BILIBILI_COARSE_AXES
  --condition control : use the length-matched NON-taxonomic safety-language
                        control (clause 5 AP2 defense). Same pipeline, only
                        the rules-slot text differs. Approximately
                        character-length-matched to the axes string.

Output JSONL (contract with src/quick_eval_all.py::load_scores_file):
    {"video_id": str, "score": float}

Usage (always via sbatch):
  sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh \\
    && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 \\
    && python src/prompt_paradigm/coarse_axes_prompt.py \\
         --dataset MHClip_EN --split test --condition axes"

The axes strings and the length-matched control strings below are FROZEN at
proposal time (docs/proposals/prompt_paradigm_v6.md section 3.2 and 4.2).
Do not tune on results.
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


# =============================================================================
# FROZEN STRINGS — do not edit after Gate 1 approval
# =============================================================================

# --- Coarse-axes (v6 treatment) ---
# Derived from the frozen YOUTUBE_RULES in src/score_holistic_2b.py lines 32-40.
# Axis 1 is the shared subject of every YouTube rule ("groups based on protected
# status"). Axis 2 enumerates the action vocabulary of rules 1-9 (violence,
# incite, dehumanize, inferiority, supremacism, conspiracy, deny, slurs).
YOUTUBE_COARSE_AXES = """1. The content identifies a target that is a group defined by a protected attribute (race, ethnicity, national origin, religion, sexual orientation, gender identity, disability, or similar).
2. The content expresses hostility toward that target through violence, incitement, dehumanization, inferiority claims, supremacist framing, conspiratorial attacks, denial of past harms, or slurs and stereotypes."""

# Derived from the frozen BILIBILI_RULES in src/score_holistic_2b.py lines 42-49.
# Axis 1 combines the subject scope of rules 1, 2, 4, 7 ("protected groups"),
# rule 3 ("personal identity"), and rules 5, 6 ("death/sickness/disability/
# disaster circumstances"). Axis 2 is the action vocabulary of rules 1-8
# (discrimination, attack, abuse, ideology, mocking, incitement).
BILIBILI_COARSE_AXES = """1. The content identifies a target that is a group defined by a protected attribute, an identifiable person, or a circumstance involving death, sickness, disability, or disaster.
2. The content expresses hostility toward that target through discrimination, personal attacks, abuse, hateful ideologies or symbols, mocking, or incitement to confrontation."""


# --- Length-matched non-taxonomic control (v6 clause-5 control) ---
# Character-length-matched (~within 20%) to the coarse-axes strings above,
# but the content is platform-flavored safety boilerplate that does NOT encode
# the target x hostility schema. If v6 gains come from prompt length alone,
# running with these strings should produce a comparable gain. If v6 gains are
# content-specific, running with these strings should NOT beat baseline.
# FROZEN at Gate 1 submission, authored by prompt-paradigm before any v6 score
# is computed.
YOUTUBE_LENGTH_CONTROL = """1. This platform's policies exist to protect user wellbeing and to promote respectful discourse across the diverse communities that use the service every day.
2. Users are encouraged to consider the impact of their contributions on other viewers before publishing, and to follow the community guidelines published by the platform."""

BILIBILI_LENGTH_CONTROL = """1. This platform's policies exist to protect user wellbeing and to promote respectful discourse across the diverse communities that use the service every day.
2. Users are encouraged to consider the impact of their contributions on other viewers before publishing, and to follow the community guidelines published by the platform."""


def _rules_text_for(dataset: str, condition: str) -> str:
    if condition == "axes":
        return YOUTUBE_COARSE_AXES if dataset == "MHClip_EN" else BILIBILI_COARSE_AXES
    if condition == "control":
        return YOUTUBE_LENGTH_CONTROL if dataset == "MHClip_EN" else BILIBILI_LENGTH_CONTROL
    raise ValueError(f"unknown condition: {condition}")


# =============================================================================
# Prompt template — verbatim copy of frozen BINARY_PROMPT from
# src/score_holistic_2b.py lines 52-62. The ONLY thing v6 changes is which
# string fills the {rules} slot.
# =============================================================================
SYSTEM_PROMPT = "You are a content moderation analyst."
USER_TEMPLATE = """Observe this video's visual frames, title, and transcript.

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


def build_media_content(media_path, media_type):
    if media_type == "video":
        return [{"type": "video_url", "video_url": {"url": f"file://{media_path}"}}]
    jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")))
    if len(jpgs) > 8:
        indices = np.linspace(0, len(jpgs) - 1, 8, dtype=int)
        jpgs = [jpgs[i] for i in indices]
    return [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in jpgs]


def build_messages(media_path, media_type, ann, rules_text, transcript_limit):
    title = ann.get("title", "") or ""
    transcript = (ann.get("transcript", "") or "")[:transcript_limit]
    user_text = USER_TEMPLATE.format(title=title, transcript=transcript, rules=rules_text)
    media_content = build_media_content(media_path, media_type)
    content = media_content + [{"type": "text", "text": user_text}]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["MHClip_EN", "MHClip_ZH"])
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--condition", required=True, choices=["axes", "control"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--transcript-limit", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N videos (pilot); 0 = all")
    parser.add_argument("--out-suffix", default="",
                        help="extra suffix on output filename, e.g. _pilot10")
    args = parser.parse_args()

    rules_text = _rules_text_for(args.dataset, args.condition)

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(
                log_dir,
                f"prompt_paradigm_v6_{args.dataset}_{args.split}_{args.condition}.log")),
            logging.StreamHandler(),
        ],
    )
    logging.info(
        f"Config: dataset={args.dataset} split={args.split} "
        f"condition={args.condition} model={args.model}"
    )
    logging.info(f"Rules block ({len(rules_text)} chars):\n{rules_text}")

    annotations = load_annotations(args.dataset)
    root = DATASET_ROOTS[args.dataset]
    split_path = os.path.join(root, "splits", f"{args.split}_clean.csv")
    if not os.path.isfile(split_path):
        from data_utils import generate_clean_splits
        generate_clean_splits(args.dataset)
    with open(split_path) as f:
        split_ids = [line.strip() for line in f if line.strip()]
    logging.info(f"Clean {args.split} split: {len(split_ids)} videos")

    if args.limit > 0:
        split_ids = split_ids[:args.limit]
        logging.info(f"Limit applied: scoring {len(split_ids)} videos")

    out_dir = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"{args.split}_coarse_axes_{args.condition}{args.out_suffix}.jsonl"
    )

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
    n_null = 0

    for batch_start in range(0, len(remaining), args.batch_size):
        batch_ids = remaining[batch_start:batch_start + args.batch_size]

        batch_messages = []
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

            msgs = build_messages(media_path, media_type, ann, rules_text,
                                  args.transcript_limit)
            batch_messages.append(msgs)
            batch_vid_ids.append(vid_id)

        if not batch_vid_ids:
            continue

        scores = [None] * len(batch_vid_ids)
        try:
            outputs = llm.chat(messages=batch_messages, sampling_params=binary_params)
            for i, out in enumerate(outputs):
                scores[i] = extract_binary_score(out, label_token_ids)
        except Exception as e:
            logging.error(f"  Batch failed: {e}, falling back to single")
            for i, msgs in enumerate(batch_messages):
                try:
                    out_single = llm.chat(messages=[msgs], sampling_params=binary_params)
                    scores[i] = extract_binary_score(out_single[0], label_token_ids)
                except Exception as e2:
                    err2 = str(e2)
                    if "longer than the maximum model length" in err2 or "max_model_len" in err2:
                        logging.warning(f"  {batch_vid_ids[i]}: SKIPPED oversize")
                    else:
                        logging.error(f"  {batch_vid_ids[i]}: single failed: {err2}")
                    scores[i] = None

        with open(out_path, "a") as f:
            for i in range(len(batch_vid_ids)):
                s = scores[i]
                if s is None:
                    rec = {"video_id": batch_vid_ids[i], "score": None, "skipped": True}
                    n_null += 1
                else:
                    rec = {"video_id": batch_vid_ids[i], "score": s}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        n_processed += len(batch_vid_ids)
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        logging.info(
            f"  [{len(done_ids)+n_processed}/{len(split_ids)}] rate={rate:.1f} vid/s"
        )

    logging.info(
        f"Done. scored={n_processed} skipped={n_skipped} null={n_null} out={out_path}"
    )


if __name__ == "__main__":
    main()
