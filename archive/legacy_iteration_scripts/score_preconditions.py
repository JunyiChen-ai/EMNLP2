"""
Steps 4-5: MLLM precondition scoring.

Reads CLIP filter output (relevant rules + selected frame indices per video),
then scores each precondition via debiased token probability judgment.
Falls back to cascaded CoT reasoning for uncertain cases.

vLLM only — no CLIP loaded in this script.

Usage:
  python src/score_preconditions.py --dataset HateMM --split test \
      --model Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np

PROJECT_ROOT = "/data/jehc223/EMNLP2"

DATASET_CONSTITUTION = {
    "HateMM": "youtube",
    "MHClip_EN": "youtube",
    "MHClip_ZH": "bilibili",
}

DATASET_CONFIGS = {
    "HateMM": {
        "annotation": "datasets/HateMM/annotation(new).json",
        "splits_dir": "datasets/HateMM/splits",
        "frames_dir": "datasets/HateMM/frames",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
        "label_field": "Label",
        "label_map": {"Hate": 1, "Non Hate": 0},
    },
    "MHClip_EN": {
        "annotation": "datasets/MHClip_EN/annotation(new).json",
        "splits_dir": "datasets/MHClip_EN/splits",
        "frames_dir": "datasets/MHClip_EN/frames",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
        "label_field": "Label",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    },
    "MHClip_ZH": {
        "annotation": "datasets/MHClip_ZH/annotation(new).json",
        "splits_dir": "datasets/MHClip_ZH/splits",
        "frames_dir": "datasets/MHClip_ZH/frames",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
        "label_field": "Label",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    },
}

SYSTEM_PROMPT = "You are a content moderation analyst. Answer each question based strictly on observable evidence. Do not speculate beyond what is shown."

PRECONDITION_PROMPT = """{preamble}

Determine if the following content is present in this video based on what you can observe. No external context or intent of the video should be considered when determining presence.

Content: {precondition_text}

Answer "Yes" or "No"."""

REASONING_PROMPT_STEP1 = """{preamble}

Determine if the following content is present in this video based on what you can observe. No external context or intent of the video should be considered when determining a visible state. Think step by step.

Content: {precondition_text}"""

REASONING_PROMPT_STEP2 = """Based on the answer, summarize the results. Steps:
1. Select a "rating". This should be "Yes" or "No".
2. Provide a "rationale". Explain the reason for your decision.
Respond with ONLY a JSON object in this exact format (no markdown, no code fences):
{"rating": "Yes", "rationale": "your reason here"}"""


# ── MLLM helpers ───────────────────────────────────────────────────

def build_preamble_full(frame_paths, title, transcript):
    """Build multimodal user content with frames + text."""
    content = []
    for fp in frame_paths:
        content.append({"type": "image_url", "image_url": {"url": f"file://{fp}"}})
    text = f"You are given {len(frame_paths)} frames sampled from a video, along with its title and transcript.\n\nTitle: {title}\nTranscript: {transcript}"
    content.append({"type": "text", "text": text})
    return content


def build_preamble_text_only(title, transcript):
    return f"You are given the title and transcript of a video (no visual content available).\n\nTitle: {title}\nTranscript: {transcript}"


def build_preamble_prior():
    return "You are given a video. No title, transcript, or visual content is available."


def extract_yes_no_score(output, label_token_ids):
    """Extract P(Yes) / (P(Yes) + P(No)) from position-0 logprobs."""
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


def build_label_token_ids(tokenizer):
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
        logging.info(f"  Label '{label}' -> token IDs {mapping[label]} ({decoded})")
    return mapping


def score_token_prob(llm, sampling_params, messages, label_token_ids):
    """Score via MLLM token probability. Returns P(Yes) or None."""
    try:
        outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
        return extract_yes_no_score(outputs[0], label_token_ids)
    except Exception as e:
        logging.warning(f"  MLLM call failed: {e}")
        return None


def score_reasoning(llm, sampling_params, messages_step1, step2_text):
    """Score via cascaded CoT reasoning. Returns (satisfied: bool, rationale: str)."""
    import re as _re
    try:
        outputs = llm.chat(messages=[messages_step1], sampling_params=sampling_params)
        cot_response = outputs[0].outputs[0].text.strip()

        messages_step2 = messages_step1 + [
            {"role": "assistant", "content": cot_response},
            {"role": "user", "content": step2_text},
        ]
        outputs2 = llm.chat(messages=[messages_step2], sampling_params=sampling_params)
        summary = outputs2[0].outputs[0].text.strip()

        json_match = _re.search(r'\{[^}]+\}', summary)
        if json_match:
            parsed = json.loads(json_match.group())
            rating = parsed.get("rating", "").strip().lower()
            rationale = parsed.get("rationale", "")
            return rating == "yes", rationale

        if "yes" in summary.lower()[:20]:
            return True, summary
        return False, summary

    except Exception as e:
        logging.warning(f"  Reasoning fallback failed: {e}")
        return False, str(e)


# ── Frame path helpers ─────────────────────────────────────────────

def get_all_frame_paths(frames_dir, vid_id, max_frames=32):
    """Get sorted frame paths, subsampled to max_frames."""
    d = os.path.join(frames_dir, vid_id)
    if not os.path.isdir(d):
        return []
    files = sorted(
        (f for f in os.listdir(d) if f.endswith((".jpg", ".png", ".jpeg"))),
        key=lambda f: int(''.join(c for c in os.path.splitext(f)[0] if c.isdigit()) or '0'),
    )
    if not files:
        return []
    if len(files) > max_frames:
        indices = np.linspace(0, len(files) - 1, max_frames).round().astype(int)
        seen = set()
        unique = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                unique.append(i)
        files = [files[i] for i in unique]
    return [os.path.join(d, f) for f in files]


def select_frames_by_indices(all_frame_paths, indices):
    """Select frames by index list from CLIP filter output."""
    return [all_frame_paths[i] for i in indices if i < len(all_frame_paths)]


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score preconditions via MLLM")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num-logprobs", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--constitution-suffix", default="",
                        help="Suffix for constitution file, e.g. '_merged' loads preconditions_{platform}_merged.json")
    parser.add_argument("--alpha2-override", type=float, default=None,
                        help="Override alpha2 threshold (default: 0.8*(1-prior)). Use 0.3 when prior≈0.")
    parser.add_argument("--disable-reasoning", action="store_true",
                        help="Disable reasoning fallback — use token prob only with threshold")
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    root = args.project_root
    constitution_name = DATASET_CONSTITUTION[args.dataset]

    # Determine output suffix for separate result files
    out_suffix = args.constitution_suffix  # e.g. "_merged" or ""

    # Logging
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"score_{args.dataset}_{args.split}{out_suffix}.log")),
            logging.StreamHandler(),
        ],
    )

    # Load preconditions
    precond_path = os.path.join(root, "constitution", f"preconditions_{constitution_name}{args.constitution_suffix}.json")
    if not os.path.exists(precond_path):
        logging.error(f"Preconditions not found: {precond_path}")
        sys.exit(1)
    with open(precond_path) as f:
        rules = json.load(f)
    total_pc = sum(len(r["preconditions"]) for r in rules)
    logging.info(f"Loaded {len(rules)} rules with {total_pc} preconditions from {constitution_name}")

    # Load CLIP filter output
    clip_filter_path = os.path.join(root, "results", "clip_filter", args.dataset, f"{args.split}{out_suffix}.json")
    if not os.path.exists(clip_filter_path):
        logging.error(f"CLIP filter output not found: {clip_filter_path}. Run clip_filter.py first.")
        sys.exit(1)
    with open(clip_filter_path) as f:
        clip_filter = json.load(f)
    logging.info(f"Loaded CLIP filter for {len(clip_filter)} videos")

    # Load annotations + split
    with open(os.path.join(root, cfg["annotation"])) as f:
        annotations = json.load(f)
    id2sample = {s[cfg["id_field"]]: s for s in annotations}

    split_path = os.path.join(root, cfg["splits_dir"], f"{args.split}.csv")
    with open(split_path) as f:
        split_ids = [line.strip() for line in f if line.strip()]
    logging.info(f"Dataset={args.dataset} split={args.split} n={len(split_ids)}")

    # Output + resume
    out_dir = os.path.join(root, "results", "scores", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}{out_suffix}.jsonl")

    done_ids = set()
    if os.path.exists(out_path):
        valid_bytes = 0
        with open(out_path, "rb") as f:
            for raw_line in f:
                try:
                    text_line = raw_line.decode("utf-8")
                    rec = json.loads(text_line)
                    if rec.get("_metadata"):
                        valid_bytes += len(raw_line)
                        continue
                    done_ids.add(rec["video_id"])
                    valid_bytes += len(raw_line)
                except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                    break
        with open(out_path, "r+b") as f:
            f.truncate(valid_bytes)

    run_metadata = {
        "_metadata": True,
        "model": args.model,
        "constitution": constitution_name,
        "num_logprobs": args.num_logprobs,
    }
    if not done_ids:
        with open(out_path, "w") as f:
            f.write(json.dumps(run_metadata) + "\n")
    else:
        with open(out_path) as f:
            first_line = f.readline().strip()
            if first_line:
                try:
                    saved_meta = json.loads(first_line)
                    if saved_meta.get("_metadata"):
                        for key in ["model", "constitution", "num_logprobs"]:
                            if saved_meta.get(key) != run_metadata.get(key):
                                logging.error(
                                    f"Config mismatch on resume! {key}: "
                                    f"saved={saved_meta.get(key)}, current={run_metadata.get(key)}. "
                                    f"Delete {out_path} to start fresh."
                                )
                                sys.exit(1)
                except json.JSONDecodeError:
                    pass

    todo_ids = [v for v in split_ids if v in id2sample and v in clip_filter and v not in done_ids]
    if not todo_ids:
        logging.info("All done.")
        return
    logging.info(f"Resume: {len(done_ids)} done, {len(todo_ids)} remaining")

    # Load vLLM (no CLIP — GPU is fully available)
    logging.info("Loading vLLM...")
    from vllm import LLM, SamplingParams

    frames_dir = os.path.join(root, cfg["frames_dir"])
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        allowed_local_media_path="/data/jehc223",
        limit_mm_per_prompt={"image": args.max_frames},
        max_logprobs=max(20, args.num_logprobs),
        mm_processor_kwargs={"max_pixels": 1003520},
    )
    tokenizer = llm.get_tokenizer()
    label_token_ids = build_label_token_ids(tokenizer)

    all_label_tids = label_token_ids["Yes"] + label_token_ids["No"]
    sampling_token = SamplingParams(
        temperature=0, max_tokens=1, logprobs=args.num_logprobs,
        allowed_token_ids=all_label_tids,
    )
    sampling_reasoning = SamplingParams(temperature=0, max_tokens=256)

    # Prior cache
    prior_cache_path = os.path.join(out_dir, f"prior_cache_{constitution_name}{out_suffix}.json")
    if os.path.exists(prior_cache_path):
        with open(prior_cache_path) as f:
            prior_cache = json.load(f)
        logging.info(f"Loaded prior cache from {prior_cache_path}")
    else:
        prior_cache = {}

    # Compute missing priors (handles merged constitution with new rules)
    missing_keys = []
    for rule in rules:
        for precond in rule["preconditions"]:
            key = f"{rule['rule_id']}::{precond}"
            if key not in prior_cache:
                missing_keys.append((rule, precond, key))

    if missing_keys:
        logging.info(f"Computing {len(missing_keys)} prior scores (no input)...")
        preamble_prior = build_preamble_prior()
        for rule, precond, key in missing_keys:
            prompt_text = PRECONDITION_PROMPT.format(
                preamble=preamble_prior, precondition_text=precond,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
            score = score_token_prob(llm, sampling_token, messages, label_token_ids)
            prior_cache[key] = score if score is not None else 0.5
            logging.info(f"  Prior {rule['rule_id']}: {precond[:50]}... = {prior_cache[key]:.3f}")

        with open(prior_cache_path, "w") as f:
            json.dump(prior_cache, f, indent=2)
        logging.info(f"Saved prior cache to {prior_cache_path}")
    else:
        logging.info(f"All priors cached ({len(prior_cache)} entries)")

    # Process videos
    t0 = time.time()
    n_done = 0

    for vid_id in todo_ids:
        sample = id2sample[vid_id]
        title = sample.get(cfg["title_field"], "") or ""
        transcript = sample.get(cfg["transcript_field"], "") or ""
        if len(transcript) > 1500:
            transcript = transcript[:1500] + "..."
        gt_label = cfg["label_map"].get(sample.get(cfg["label_field"], ""), -1)

        # Get all frame paths (same subsampling as clip_filter.py)
        all_frame_paths = get_all_frame_paths(frames_dir, vid_id, args.max_frames)

        # Read CLIP filter results for this video
        vid_clip = clip_filter[vid_id]

        video_result = {
            "video_id": vid_id,
            "gt_label": gt_label,
            "rules": {},
        }

        for rule in rules:
            rule_id = rule["rule_id"]
            rule_clip = vid_clip.get(rule_id, {})

            if not rule_clip.get("relevant", False):
                video_result["rules"][rule_id] = {
                    "relevant": False,
                    "relevance_score": rule_clip.get("relevance_score", 0.0),
                    "preconditions": {},
                }
                continue

            # Get selected frames from CLIP filter
            if rule_clip.get("text_only", False) or not all_frame_paths:
                selected_frame_paths = []
            else:
                indices = rule_clip.get("selected_frame_indices", [])
                selected_frame_paths = select_frames_by_indices(all_frame_paths, indices)

            # Score each precondition
            precond_results = {}
            all_satisfied = True

            for precond_idx, precond in enumerate(rule["preconditions"]):
                precond_key = f"{rule_id}::{precond}"
                score_prior = prior_cache.get(precond_key, 0.5)

                # Full mode (selected frames + text)
                if selected_frame_paths:
                    user_content = build_preamble_full(selected_frame_paths, title, transcript)
                    prompt_text = PRECONDITION_PROMPT.format(
                        preamble="", precondition_text=precond,
                    )
                    user_content.append({"type": "text", "text": prompt_text.lstrip()})
                    messages_full = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ]
                else:
                    preamble = build_preamble_text_only(title, transcript)
                    prompt_text = PRECONDITION_PROMPT.format(
                        preamble=preamble, precondition_text=precond,
                    )
                    messages_full = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_text},
                    ]

                score_full = score_token_prob(llm, sampling_token, messages_full, label_token_ids)
                if score_full is None:
                    score_full = 0.5

                # Text-only mode (for debiasing)
                preamble_text = build_preamble_text_only(title, transcript)
                prompt_text_only = PRECONDITION_PROMPT.format(
                    preamble=preamble_text, precondition_text=precond,
                )
                messages_text = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text_only},
                ]
                score_text = score_token_prob(llm, sampling_token, messages_text, label_token_ids)
                if score_text is None:
                    score_text = 0.5

                # Debiasing
                alpha1 = -0.3 * score_prior
                alpha2 = args.alpha2_override if args.alpha2_override is not None else 0.8 * (1.0 - score_prior)
                beta = 0.6
                diff_prior = score_full - score_prior

                if diff_prior < alpha1:
                    satisfied = False
                    method = "token_prob_negative"
                elif diff_prior > alpha2:
                    satisfied = True
                    method = "token_prob_positive"
                elif (score_full - score_text) > beta:
                    satisfied = True
                    method = "visual_contribution"
                elif args.disable_reasoning:
                    # No reasoning fallback — decide by threshold
                    satisfied = score_full > 0.5
                    method = "token_prob_threshold"
                else:
                    # Cascaded reasoning fallback
                    if selected_frame_paths:
                        user_content_r = build_preamble_full(selected_frame_paths, title, transcript)
                        r_prompt = REASONING_PROMPT_STEP1.format(
                            preamble="", precondition_text=precond,
                        )
                        user_content_r.append({"type": "text", "text": r_prompt.lstrip()})
                        messages_reasoning = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content_r},
                        ]
                    else:
                        preamble_r = build_preamble_text_only(title, transcript)
                        r_prompt = REASONING_PROMPT_STEP1.format(
                            preamble=preamble_r, precondition_text=precond,
                        )
                        messages_reasoning = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": r_prompt},
                        ]

                    satisfied, rationale = score_reasoning(
                        llm, sampling_reasoning, messages_reasoning, REASONING_PROMPT_STEP2,
                    )
                    method = "reasoning_fallback"

                precond_results[precond] = {
                    "score_full": score_full,
                    "score_text": score_text,
                    "score_prior": score_prior,
                    "satisfied": satisfied,
                    "method": method,
                }

                if not satisfied:
                    all_satisfied = False
                    remaining = rule["preconditions"][precond_idx + 1:]
                    for rem in remaining:
                        precond_results[rem] = {
                            "score_full": None,
                            "score_text": None,
                            "score_prior": prior_cache.get(f"{rule_id}::{rem}", 0.5),
                            "satisfied": None,
                            "method": "skipped_early_termination",
                        }
                    break

            video_result["rules"][rule_id] = {
                "relevant": True,
                "relevance_score": rule_clip.get("relevance_score", 0.0),
                "all_satisfied": all_satisfied,
                "preconditions": precond_results,
            }

        # Write result
        with open(out_path, "a") as f:
            f.write(json.dumps(video_result, ensure_ascii=False) + "\n")

        n_done += 1
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (len(todo_ids) - n_done) / rate if rate > 0 else 0
        logging.info(f"[{n_done}/{len(todo_ids)}] {vid_id} | {rate:.2f} vid/s | ETA {eta/60:.0f}m")

    logging.info(f"Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
