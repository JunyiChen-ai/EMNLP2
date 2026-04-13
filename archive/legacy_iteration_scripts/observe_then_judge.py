"""
Observe-then-Judge pipeline for hateful video detection.

Step 1 (observe): Multimodal call extracts free-text observations from video.
Step 2 (judge):   Text-only call scores observations against constitution rules.

Usage:
  python src/observe_then_judge.py --dataset MHClip_EN --step observe
  python src/observe_then_judge.py --dataset MHClip_EN --step judge
"""

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import DATASET_ROOTS, get_media_path, load_annotations

PROJECT_ROOT = "/data/jehc223/EMNLP2"

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

# ---------- OBSERVE PROMPT ----------
# Designed to elicit hate-relevant observations without priming for a verdict.
# The model describes what it sees/hears; judgment is deferred to step 2.
OBSERVE_PROMPT = """You are a content analyst. Watch this video and read the title and transcript below.

Title: {title}
Transcript: {transcript}

Describe the observable content that would be relevant to evaluating this video against community guidelines on hate speech. Specifically note:
- What groups, individuals, or identities are depicted or referenced
- Any slurs, stereotypes, dehumanizing comparisons, or coded language
- Tone and apparent intent (sincere, ironic, humorous, aggressive)
- Whether visual content reinforces or contradicts the spoken/text message

Be factual and specific. Do not make a judgment about whether the video violates rules — just describe what you observe."""

# ---------- JUDGE PROMPT ----------
# Text-only: given observations + rules, produce Yes/No verdict.
JUDGE_PROMPT = """You are a content moderation analyst. Based on the following analyst observations about a video, determine whether the video violates community guidelines.

Analyst observations:
{observation}

The platform prohibits the following:
{rules}

Based on the observations above, does this video contain content that violates any of the above rules?

Answer "Yes" or "No"."""


def build_media_content(media_path, media_type):
    """Build vLLM content list for video or frames."""
    if media_type == "video":
        return [{"type": "video_url", "video_url": {"url": f"file://{media_path}"}}]
    else:
        import glob as globmod
        jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")))
        if len(jpgs) > 8:
            indices = np.linspace(0, len(jpgs) - 1, 8, dtype=int)
            jpgs = [jpgs[i] for i in indices]
        return [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in jpgs]


def build_binary_token_ids(tokenizer):
    """Map Yes/No labels to token IDs."""
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
    """Extract P(Yes)/(P(Yes)+P(No)) from logprobs."""
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


def evaluate_scores(out_path, dataset):
    """Load scores and ground truth, sweep thresholds, print best metrics."""
    annotations = load_annotations(dataset)

    scores = []
    labels = []
    with open(out_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vid = r["video_id"]
            s = r["score"]
            if s is None or vid not in annotations:
                continue
            gt_label = annotations[vid]["label"]
            gt = 1 if gt_label in ("Hateful", "Offensive") else 0
            scores.append(s)
            labels.append(gt)

    scores = np.array(scores)
    labels = np.array(labels)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    logging.info(f"\nEvaluation: {len(labels)} scored videos, {n_pos} positive, {n_neg} negative")

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    logging.info(f"  Positive class scores: mean={pos_scores.mean():.4f}, std={pos_scores.std():.4f}, "
                 f"min={pos_scores.min():.4f}, max={pos_scores.max():.4f}")
    logging.info(f"  Negative class scores: mean={neg_scores.mean():.4f}, std={neg_scores.std():.4f}, "
                 f"min={neg_scores.min():.4f}, max={neg_scores.max():.4f}")

    best_acc = 0
    best_thresh_acc = 0
    best_f1 = 0
    best_thresh_f1 = 0
    best_metrics_at_f1 = {}

    logging.info(f"\nThreshold sweep (0.1 increments):")
    logging.info(f"  Thresh     ACC      F1    Prec  Recall")

    for t_int in range(1, 100):
        thresh = t_int / 100.0
        preds = (scores >= thresh).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        acc = (tp + tn) / len(labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if t_int % 10 == 0:
            logging.info(f"     {thresh:.1f}  {acc:.4f}  {f1:.4f}  {precision:.4f}  {recall:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_thresh_acc = thresh
        if f1 > best_f1:
            best_f1 = f1
            best_thresh_f1 = thresh
            best_metrics_at_f1 = {"acc": acc, "f1": f1, "precision": precision, "recall": recall,
                                   "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    logging.info(f"\nBest ACC: {best_acc:.4f} (thresh={best_thresh_acc:.2f})")
    logging.info(f"Best F1:  {best_f1:.4f} (thresh={best_thresh_f1:.2f})")
    if best_metrics_at_f1:
        m = best_metrics_at_f1
        logging.info(f"  ACC={m['acc']:.4f}  Prec={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}")
        logging.info(f"\nConfusion matrix at best F1 threshold ({best_thresh_f1:.2f}):")
        logging.info(f"                Pred=Hateful  Pred=Normal")
        logging.info(f"  GT=Hateful    TP={m['tp']:>4d}        FN={m['fn']:>4d}")
        logging.info(f"  GT=Normal     FP={m['fp']:>4d}        TN={m['tn']:>4d}")

    return {
        "n_scored": len(labels),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "best_acc": float(best_acc),
        "best_acc_thresh": float(best_thresh_acc),
        "best_f1": float(best_f1),
        "best_f1_thresh": float(best_thresh_f1),
        "best_f1_precision": float(best_metrics_at_f1.get("precision", 0)),
        "best_f1_recall": float(best_metrics_at_f1.get("recall", 0)),
    }


# ========================= OBSERVE STEP =========================

def run_observe(args):
    """Step 1: Extract free-text observations from videos (multimodal call)."""
    annotations = load_annotations(args.dataset)
    root = DATASET_ROOTS[args.dataset]
    split_path = os.path.join(root, "splits", f"{args.split}_clean.csv")

    if not os.path.isfile(split_path):
        logging.info(f"Clean split not found at {split_path}, generating...")
        from data_utils import generate_clean_splits
        generate_clean_splits(args.dataset)

    with open(split_path) as f:
        split_ids = [line.strip() for line in f if line.strip()]
    logging.info(f"Clean {args.split} split: {len(split_ids)} videos")

    out_dir = os.path.join(PROJECT_ROOT, "results", "otj_8b", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}_observations.jsonl")

    # Resume support
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
        logging.info("All observations extracted.")
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

    # Free-text generation: allow up to 256 tokens, temperature=0 for deterministic
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
    )

    t0 = time.time()
    n_processed = 0
    n_skipped = 0

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

            prompt_text = OBSERVE_PROMPT.format(
                title=title,
                transcript=transcript[:300],
            )

            media_content = build_media_content(media_path, media_type)
            content = media_content + [{"type": "text", "text": prompt_text}]

            messages = [
                {"role": "system", "content": "You are a content analyst. Describe observable facts without making judgments."},
                {"role": "user", "content": content},
            ]
            batch_messages.append(messages)
            batch_vid_ids.append(vid_id)

        if not batch_messages:
            continue

        try:
            outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params)
        except Exception as e:
            err_msg = str(e)
            logging.error(f"  Batch failed: {err_msg}, falling back to single")
            for i, msgs in enumerate(batch_messages):
                try:
                    out_single = llm.chat(messages=[msgs], sampling_params=sampling_params)
                    obs_text = out_single[0].outputs[0].text.strip() if out_single[0].outputs else ""
                except Exception as e2:
                    err2_msg = str(e2)
                    if "longer than the maximum model length" in err2_msg or "max_model_len" in err2_msg:
                        logging.warning(f"  {batch_vid_ids[i]}: SKIPPED (prompt exceeds max_model_len)")
                        obs_text = None
                        n_skipped += 1
                    else:
                        logging.error(f"  {batch_vid_ids[i]}: single failed: {err2_msg}")
                        obs_text = None
                rec = {"video_id": batch_vid_ids[i], "observation": obs_text}
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                n_processed += 1
            continue

        with open(out_path, "a") as f:
            for i, output in enumerate(outputs):
                obs_text = output.outputs[0].text.strip() if output.outputs else ""
                rec = {"video_id": batch_vid_ids[i], "observation": obs_text}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        n_processed += len(batch_vid_ids)
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        logging.info(f"  [{len(done_ids)+n_processed}/{len(split_ids)}] {rate:.1f} vid/s")

    logging.info(f"\nObserve done. {n_processed} extracted, {n_skipped} skipped.")
    logging.info(f"Output: {out_path}")


# ========================= JUDGE STEP =========================

def run_judge(args):
    """Step 2: Text-only judgment on observations (no video needed)."""
    platform = CONSTITUTION_MAP[args.dataset]
    rules_text = YOUTUBE_RULES if platform == "youtube" else BILIBILI_RULES

    obs_dir = os.path.join(PROJECT_ROOT, "results", "otj_8b", args.dataset)
    obs_path = os.path.join(obs_dir, f"{args.split}_observations.jsonl")

    if not os.path.isfile(obs_path):
        logging.error(f"Observations not found at {obs_path}. Run --step observe first.")
        sys.exit(1)

    # Load observations
    observations = {}
    with open(obs_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                vid = r["video_id"]
                obs = r.get("observation")
                if obs:
                    observations[vid] = obs
    logging.info(f"Loaded {len(observations)} observations from {obs_path}")

    out_path = os.path.join(obs_dir, f"{args.split}_judge_scores.jsonl")

    # Resume support
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

    remaining = [v for v in observations if v not in done_ids]
    if not remaining:
        logging.info("All judged. Running evaluation...")
        metrics = evaluate_scores(out_path, args.dataset)
        logging.info(f"Final metrics: {json.dumps(metrics, indent=2)}")
        return

    from vllm import LLM, SamplingParams

    # Text-only: no video processing needed, much faster
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_model_len=32768,
    )

    tokenizer = llm.get_tokenizer()
    label_token_ids = build_binary_token_ids(tokenizer)
    all_constrained_ids = set()
    for tids in label_token_ids.values():
        all_constrained_ids.update(tids)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=list(all_constrained_ids),
    )

    t0 = time.time()
    n_processed = 0

    # Larger batch size for text-only (no video memory overhead)
    judge_batch_size = args.batch_size * 4

    for batch_start in range(0, len(remaining), judge_batch_size):
        batch_ids = remaining[batch_start:batch_start + judge_batch_size]
        batch_messages = []
        batch_vid_ids = []

        for vid_id in batch_ids:
            obs = observations[vid_id]
            prompt_text = JUDGE_PROMPT.format(
                observation=obs,
                rules=rules_text,
            )
            messages = [
                {"role": "system", "content": "You are a content moderation analyst. Answer based strictly on the observations provided."},
                {"role": "user", "content": prompt_text},
            ]
            batch_messages.append(messages)
            batch_vid_ids.append(vid_id)

        if not batch_messages:
            continue

        try:
            outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params)
        except Exception as e:
            logging.error(f"  Batch failed: {e}, falling back to single")
            for i, msgs in enumerate(batch_messages):
                try:
                    out_single = llm.chat(messages=[msgs], sampling_params=sampling_params)
                    score = extract_binary_score(out_single[0], label_token_ids)
                except Exception:
                    score = None
                rec = {"video_id": batch_vid_ids[i], "score": score}
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                n_processed += 1
            continue

        with open(out_path, "a") as f:
            for i, output in enumerate(outputs):
                score = extract_binary_score(output, label_token_ids)
                rec = {"video_id": batch_vid_ids[i], "score": score}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        n_processed += len(batch_vid_ids)
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        logging.info(f"  [{len(done_ids)+n_processed}/{len(observations)}] {rate:.1f} vid/s")

    logging.info(f"\nJudge done. {n_processed} scored.")

    # Evaluate
    metrics = evaluate_scores(out_path, args.dataset)
    logging.info(f"Final metrics: {json.dumps(metrics, indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="Observe-then-Judge pipeline")
    parser.add_argument("--dataset", required=True, choices=["MHClip_EN", "MHClip_ZH"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--step", required=True, choices=["observe", "judge"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for observe (judge uses 4x this)")
    args = parser.parse_args()

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-6s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                f"otj_{args.dataset}_{args.step}.log")),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Config: dataset={args.dataset} step={args.step} model={args.model} split={args.split}")

    if args.step == "observe":
        run_observe(args)
    else:
        run_judge(args)


if __name__ == "__main__":
    main()
