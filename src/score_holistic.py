"""
Holistic rule-level scoring: single binary P(Yes)/P(No) per video.

Instead of scoring individual preconditions, asks the MLLM a single
constitution-informed question: "Does this video violate any of these rules?"
and extracts P(Yes)/(P(Yes)+P(No)) as a continuous score.

Usage:
  python src/score_holistic.py --dataset MHClip_EN --split test
"""

import argparse
import json
import logging
import math
import os
import sys
import time

PROJECT_ROOT = "/data/jehc223/EMNLP2"

DATASET_CONSTITUTION = {
    "MHClip_EN": "youtube",
    "MHClip_ZH": "bilibili",
}

DATASET_CONFIGS = {
    "MHClip_EN": {
        "annotation": "datasets/MHClip_EN/annotation(new).json",
        "splits_dir": "datasets/MHClip_EN/splits",
        "video_dir": "datasets/MHClip_EN/video_mp4",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
    },
    "MHClip_ZH": {
        "annotation": "datasets/MHClip_ZH/annotation(new).json",
        "splits_dir": "datasets/MHClip_ZH/splits",
        "video_dir": "datasets/MHClip_ZH/video",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
    },
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

PROMPT_VARIANTS = {
    "clean": """You are a content moderation analyst. Observe this video's visual frames, title, and transcript.

Title: {title}
Transcript: {transcript}

Based on this platform's hate speech policy, the following types of content are prohibited:
{rules}

Does this video contain content that violates any of the above rules?

Answer "Yes" or "No".""",

    "mechanism": """You are a content moderation analyst. Observe this video's visual frames, title, and transcript.

Title: {title}
Transcript: {transcript}

Based on this platform's hate speech policy, the following types of content are prohibited:
{rules}

Consider both explicit and implicit forms of hate speech, including irony, coded language, humor-based stereotyping, and contextual dog-whistles.

Does this video contain content that violates any of the above rules?

Answer "Yes" or "No".""",
}


def find_video_path(video_dir, vid_id):
    """Find video file for a given ID."""
    for ext in (".mp4", ".webm", ".mkv"):
        p = os.path.join(video_dir, vid_id + ext)
        if os.path.isfile(p):
            return p
    return None


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


def extract_yes_no_score(output, label_token_ids):
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


def main():
    parser = argparse.ArgumentParser(description="Holistic rule-level scoring")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-variant", default="clean",
                        choices=["clean", "mechanism"])
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    root = args.project_root
    platform = DATASET_CONSTITUTION[args.dataset]

    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"holistic_{args.dataset}_{args.split}_{args.prompt_variant}.log")),
            logging.StreamHandler(),
        ],
    )

    # Load annotations + split
    with open(os.path.join(root, cfg["annotation"])) as f:
        annotations = json.load(f)
    id2sample = {s[cfg["id_field"]]: s for s in annotations}

    split_path = os.path.join(root, cfg["splits_dir"], f"{args.split}.csv")
    with open(split_path) as f:
        split_ids = [line.strip() for line in f if line.strip()]
    split_ids = [v for v in split_ids if v in id2sample]
    logging.info(f"Dataset={args.dataset} split={args.split} n={len(split_ids)}")

    rules_text = YOUTUBE_RULES if platform == "youtube" else BILIBILI_RULES
    prompt_template = PROMPT_VARIANTS[args.prompt_variant]

    # Output
    out_dir = os.path.join(root, "results", "holistic", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}_{args.prompt_variant}.jsonl")

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
        logging.info(f"Resuming: {len(done_ids)} done")

    remaining = [v for v in split_ids if v not in done_ids]
    if not remaining:
        logging.info("All done.")
        return

    # Load vLLM
    from vllm import LLM, SamplingParams

    all_yes_no_ids = set()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"video": 1},
        allowed_local_media_path="/data/jehc223",
    )

    tokenizer = llm.get_tokenizer()
    label_token_ids = build_label_token_ids(tokenizer)
    for tids in label_token_ids.values():
        all_yes_no_ids.update(tids)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=list(all_yes_no_ids),
    )

    video_dir = os.path.join(root, cfg["video_dir"])
    t0 = time.time()
    n_processed = 0

    for batch_start in range(0, len(remaining), args.batch_size):
        batch_ids = remaining[batch_start:batch_start + args.batch_size]
        batch_messages = []
        batch_vid_ids = []

        for vid_id in batch_ids:
            sample = id2sample[vid_id]
            title = sample.get(cfg["title_field"], "") or ""
            transcript = sample.get(cfg["transcript_field"], "") or ""
            video_path = find_video_path(video_dir, vid_id)

            if not video_path:
                logging.warning(f"  {vid_id}: no video file, skipping")
                continue

            prompt_text = prompt_template.format(
                title=title,
                transcript=transcript[:300],
                rules=rules_text,
            )

            content = [
                {"type": "video_url", "video_url": {"url": f"file://{video_path}"}},
                {"type": "text", "text": prompt_text},
            ]

            messages = [
                {"role": "system", "content": "You are a content moderation analyst. Answer based strictly on observable evidence."},
                {"role": "user", "content": content},
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
                    score = extract_yes_no_score(out_single[0], label_token_ids)
                except Exception as e2:
                    logging.error(f"  {batch_vid_ids[i]}: single failed: {e2}")
                    score = None
                with open(out_path, "a") as f:
                    f.write(json.dumps({"video_id": batch_vid_ids[i], "score": score}, ensure_ascii=False) + "\n")
                n_processed += 1
            elapsed = time.time() - t0
            rate = n_processed / elapsed if elapsed > 0 else 0
            logging.info(f"  [{len(done_ids)+n_processed}/{len(split_ids)}] {rate:.1f} vid/s (fallback)")
            continue

        with open(out_path, "a") as f:
            for i, output in enumerate(outputs):
                score = extract_yes_no_score(output, label_token_ids)
                record = {
                    "video_id": batch_vid_ids[i],
                    "score": score,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        n_processed += len(batch_vid_ids)
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        logging.info(
            f"  [{len(done_ids)+n_processed}/{len(split_ids)}] {rate:.1f} vid/s"
        )

    logging.info(f"\nDone. {n_processed} scored. Output: {out_path}")


if __name__ == "__main__":
    main()
