"""
prompt_paradigm v5 — Per-Rule Disjunction Readout scorer.

ONE MLLM forward pass per video with Qwen3-VL-2B-Instruct (bf16, vLLM).
Prompt asks the model to rate each of K constitution rules as '1' (violated)
or '0' (not) in a fixed order on a single line of output, separated by spaces.
Constrained decoding: allowed_token_ids = {'0', '1', ' '}. Max tokens = 2K-1.

At each EVEN decode position i ∈ {0, 2, ..., 2(K-1)}, extract:
    p_i = P('1' at position i) / (P('0') + P('1'))
yielding a K-vector p_rules ∈ [0,1]^K. Positions 1, 3, ... are space
separators (ignored).

Aggregations:
    score_max        = max_i p_i            (PRIMARY, pre-committed)
    score_mean       = mean_i p_i
    score_nor        = 1 - prod_i (1 - p_i)
    score_top2_mean  = mean of top 2 p_i
    score_weighted   = sum_i w_i * p_i (weights filled in at eval time from train positive rate)

Output JSONL contract (for src/quick_eval_all.py::load_scores_file which reads
`score`, we write the primary `score = max(p_rules)`):

    {"video_id": str, "p_rules": [p1,...,pK], "score": float, "raw_tokens": [...],
     "raw_token_texts": [...], "format_clean": bool}

raw_tokens and raw_token_texts are kept for Ablation E (structured-output
compliance rate). format_clean is True iff the decoder emitted exactly K
binary chars separated by spaces in 2K-1 positions.

Rule counts:
    MHClip_EN (youtube): K=9
    MHClip_ZH (bilibili): K=8

Usage (always via sbatch, never direct on login node):
  sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh \
    && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 \
    && python src/prompt_paradigm/per_rule_readout.py --dataset MHClip_EN --split test"
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

RULE_COUNT = {"MHClip_EN": 9, "MHClip_ZH": 8}


# --- v5 prompt (new text; rules are verbatim from frozen) ---
PER_RULE_SYSTEM = "You are a content moderation analyst. Answer based strictly on observable evidence."

PER_RULE_USER_TEMPLATE = """Observe this video's visual frames, title, and transcript.

Title: {title}
Transcript: {transcript}

Below are the {K} community guideline rules for this platform. For each rule, decide whether this video violates that specific rule.

{rules}

For each rule in the order listed, reply with a single character: '1' if the video violates that rule, '0' if it does not. Separate the characters with single spaces on one line. Do not explain. Output exactly {K} characters separated by spaces.

Answer:"""


def build_01_space_token_ids(tokenizer):
    """Collect token IDs for '0', '1', and ' '. Multiple variants may tokenize to
    different IDs; we keep them all and normalize at read time."""
    def first_tok(s):
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if ids else None

    zeros = set()
    ones = set()
    spaces = set()
    for variant in ["0", " 0"]:
        tid = first_tok(variant)
        if tid is not None:
            zeros.add(tid)
    for variant in ["1", " 1"]:
        tid = first_tok(variant)
        if tid is not None:
            ones.add(tid)
    for variant in [" ", "  "]:
        tid = first_tok(variant)
        if tid is not None:
            spaces.add(tid)

    logging.info(f"  '0' token ids: {sorted(zeros)} decoded={[tokenizer.decode([t]) for t in zeros]}")
    logging.info(f"  '1' token ids: {sorted(ones)} decoded={[tokenizer.decode([t]) for t in ones]}")
    logging.info(f"  ' ' token ids: {sorted(spaces)} decoded={[repr(tokenizer.decode([t])) for t in spaces]}")
    return {"0": list(zeros), "1": list(ones), " ": list(spaces)}


def extract_per_rule_scores(output, id_map, K):
    """Extract K per-rule probabilities from a single generation.

    Strategy: walk the decoded tokens; at every position that decodes as '0'
    or '1' (or starts with one), compute p_1 = P('1')/(P('0')+P('1')) from the
    position's logprobs. Stop after collecting K values.

    Returns (p_rules, raw_tokens, raw_texts, format_clean):
      - p_rules: list of K floats (None for positions we could not read)
      - raw_tokens: list of token IDs the model emitted
      - raw_texts: list of decoded per-token strings
      - format_clean: True iff the decoder emitted exactly K digit positions
        (and any remaining positions are spaces).
    """
    if not output or not output.outputs:
        return [None] * K, [], [], False
    gen = output.outputs[0]
    if not gen.logprobs or len(gen.logprobs) == 0:
        return [None] * K, [], [], False

    FALLBACK = -30.0
    zero_ids = set(id_map["0"])
    one_ids = set(id_map["1"])
    space_ids = set(id_map[" "])

    p_rules = []
    raw_tokens = list(gen.token_ids) if gen.token_ids is not None else []
    raw_texts = []

    # For each decoded position, read the distribution over {0,1}.
    # Positions are (roughly) interleaved digit/space but we don't enforce
    # order — instead we collect the first K digit positions.
    digit_positions_seen = 0
    non_space_non_digit_positions = 0
    for pos_idx in range(len(gen.logprobs)):
        pos_lp = gen.logprobs[pos_idx]
        if pos_lp is None:
            continue
        # Identify the actual decoded token at this position
        emitted_tid = raw_tokens[pos_idx] if pos_idx < len(raw_tokens) else None
        if emitted_tid is None:
            continue
        raw_texts.append(_safe_decode(emitted_tid, pos_lp))

        if emitted_tid in space_ids:
            continue  # separator, skip
        if emitted_tid not in zero_ids and emitted_tid not in one_ids:
            non_space_non_digit_positions += 1
            # Still try to read P('1')/P('0') from this position's dist
            # (maybe the model started the line with '10' as a single token or similar)

        # Read the Bernoulli probability at this position
        zero_exp = 0.0
        for tid in zero_ids:
            lp = pos_lp[tid].logprob if tid in pos_lp else FALLBACK
            zero_exp += math.exp(lp)
        one_exp = 0.0
        for tid in one_ids:
            lp = pos_lp[tid].logprob if tid in pos_lp else FALLBACK
            one_exp += math.exp(lp)
        total = zero_exp + one_exp
        if total <= 0:
            p = None
        else:
            p = one_exp / total

        p_rules.append(p)
        digit_positions_seen += 1
        if digit_positions_seen >= K:
            break

    # Pad to K
    while len(p_rules) < K:
        p_rules.append(None)

    format_clean = (digit_positions_seen == K and non_space_non_digit_positions == 0)
    return p_rules, raw_tokens, raw_texts, format_clean


def _safe_decode(tid, pos_lp):
    """Best-effort decode of the emitted token for logging."""
    try:
        if tid in pos_lp and hasattr(pos_lp[tid], "decoded_token"):
            return pos_lp[tid].decoded_token
    except Exception:
        pass
    return str(tid)


def build_media_content(media_path, media_type):
    if media_type == "video":
        return [{"type": "video_url", "video_url": {"url": f"file://{media_path}"}}]
    jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")))
    if len(jpgs) > 8:
        indices = np.linspace(0, len(jpgs) - 1, 8, dtype=int)
        jpgs = [jpgs[i] for i in indices]
    return [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in jpgs]


def build_messages(media_path, media_type, title, transcript, rules_text, K, transcript_limit):
    user_text = PER_RULE_USER_TEMPLATE.format(
        title=title,
        transcript=transcript[:transcript_limit],
        rules=rules_text,
        K=K,
    )
    media_content = build_media_content(media_path, media_type)
    content = media_content + [{"type": "text", "text": user_text}]
    return [
        {"role": "system", "content": PER_RULE_SYSTEM},
        {"role": "user", "content": content},
    ]


def main():
    parser = argparse.ArgumentParser(description="prompt_paradigm v5 per-rule readout scorer")
    parser.add_argument("--dataset", required=True, choices=["MHClip_EN", "MHClip_ZH"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--transcript-limit", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of videos (for 10-video pilot)")
    parser.add_argument("--out-suffix", default="",
                        help="Optional suffix on output filename (for pilot runs)")
    parser.add_argument("--unconstrained", action="store_true",
                        help="Disable allowed_token_ids constraint (fallback mode)")
    args = parser.parse_args()

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                f"v5_per_rule_{args.dataset}_{args.split}{args.out_suffix}.log")),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"[v5 per-rule] dataset={args.dataset} split={args.split} "
                 f"limit={args.limit} unconstrained={args.unconstrained}")

    K = RULE_COUNT[args.dataset]
    platform = CONSTITUTION_MAP[args.dataset]
    rules_text = YOUTUBE_RULES if platform == "youtube" else BILIBILI_RULES

    annotations = load_annotations(args.dataset)
    root = DATASET_ROOTS[args.dataset]
    split_path = os.path.join(root, "splits", f"{args.split}_clean.csv")
    if not os.path.isfile(split_path):
        logging.info(f"Clean split not found at {split_path}, generating...")
        from data_utils import generate_clean_splits
        generate_clean_splits(args.dataset)

    with open(split_path) as f:
        split_ids = [line.strip() for line in f if line.strip()]
    if args.limit is not None:
        split_ids = split_ids[:args.limit]
    logging.info(f"Clean {args.split} split: {len(split_ids)} videos, K={K} rules")

    out_dir = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}_per_rule{args.out_suffix}.jsonl")

    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
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
    id_map = build_01_space_token_ids(tokenizer)

    all_constrained_ids = set()
    for tids in id_map.values():
        all_constrained_ids.update(tids)

    max_tokens = 2 * K - 1  # K digits + K-1 spaces, no trailing space

    sp_kwargs = dict(
        temperature=0,
        max_tokens=max_tokens,
        logprobs=20,
    )
    if not args.unconstrained:
        sp_kwargs["allowed_token_ids"] = list(all_constrained_ids)
    sampling_params = SamplingParams(**sp_kwargs)

    t0 = time.time()
    n_processed = 0
    n_skipped = 0
    n_clean = 0

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
            title = ann.get("title", "") or ""
            transcript = ann.get("transcript", "") or ""

            msgs = build_messages(media_path, media_type, title, transcript,
                                  rules_text, K, args.transcript_limit)
            batch_messages.append(msgs)
            batch_vid_ids.append(vid_id)

        if not batch_messages:
            continue

        def _build_record(vid_id, output):
            p_rules, raw_tokens, raw_texts, clean = extract_per_rule_scores(output, id_map, K)
            valid = [p for p in p_rules if p is not None]
            score = max(valid) if valid else None
            return {
                "video_id": vid_id,
                "p_rules": p_rules,
                "score": score,
                "raw_token_ids": raw_tokens,
                "raw_token_texts": raw_texts,
                "format_clean": clean,
            }

        try:
            outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params)
        except Exception as e:
            err_msg = str(e)
            logging.error(f"  Batch failed: {err_msg[:200]}, falling back to single")
            for i, msgs in enumerate(batch_messages):
                try:
                    out_single = llm.chat(messages=[msgs], sampling_params=sampling_params)
                    rec = _build_record(batch_vid_ids[i], out_single[0])
                except Exception as e2:
                    err2_msg = str(e2)
                    logging.error(f"  {batch_vid_ids[i]}: single failed: {err2_msg[:200]}")
                    rec = {
                        "video_id": batch_vid_ids[i],
                        "p_rules": [None] * K,
                        "score": None,
                        "raw_token_ids": [],
                        "raw_token_texts": [],
                        "format_clean": False,
                    }
                    n_skipped += 1
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                n_processed += 1
                if rec.get("format_clean"):
                    n_clean += 1
            continue

        with open(out_path, "a") as f:
            for i, output in enumerate(outputs):
                rec = _build_record(batch_vid_ids[i], output)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec.get("format_clean"):
                    n_clean += 1
            f.flush()
            os.fsync(f.fileno())

        n_processed += len(batch_vid_ids)
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        logging.info(f"  [{len(done_ids)+n_processed}/{len(split_ids)}] "
                     f"{rate:.1f} vid/s, format_clean={n_clean}/{n_processed}")

    logging.info(f"\nScoring done. processed={n_processed} skipped={n_skipped} "
                 f"format_clean={n_clean}/{n_processed} "
                 f"clean_rate={(n_clean / max(n_processed, 1)):.3f}")


if __name__ == "__main__":
    main()
