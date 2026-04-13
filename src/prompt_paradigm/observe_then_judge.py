"""
prompt_paradigm v1 — Observe-then-Judge scorer.

Two distinct-role MLLM calls per video with Qwen3-VL-2B-Instruct (bf16, vLLM):
  Call 1 — Observer: video frames + title + transcript -> one-sentence free-text
           description of who/what group is depicted and the stance taken.
           Forbidden to judge policy / hate / offensive / normal.
  Call 2 — Judge: text-only policy rules + title + observer description ->
           single-token Yes/No, score = P(Yes)/(P(Yes)+P(No)) from logprobs.

Output JSONL with {"video_id": str, "score": float, "observation": str} —
matches the contract read by src/quick_eval_all.py::load_scores_file.

Usage (always via sbatch, never on login node):
  sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh \
    && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 \
    && python src/prompt_paradigm/observe_then_judge.py --dataset MHClip_EN --split test"
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

# --- Observer (Call 1): perceptual description, forbidden to judge policy. ---
OBSERVER_SYSTEM = (
    "You are a careful observer describing what you see. "
    "You never judge whether content is hateful, offensive, normal, violating, appropriate, or any similar label. "
    "You only describe the target and the stance."
)

OBSERVER_USER = """Video title: {title}
Transcript excerpt: {transcript}

In ONE short sentence (max 30 words), describe:
(a) which identifiable person, group, identity, or entity this video is about, and
(b) the stance or attitude the video takes toward them (e.g., celebratory, mocking, neutral reporting, insulting, admiring, critical, praising, disparaging, joking).

Do NOT say whether the video is hateful, offensive, normal, appropriate, violating, or use any policy/moderation vocabulary. Only describe the target and the stance."""

# --- Judge (Call 2): text-only policy adjudicator, Yes/No logprob. ---
JUDGE_SYSTEM = (
    "You are a content moderation analyst. You apply the given policy literally "
    "based only on the textual evidence provided."
)

JUDGE_USER = """Platform policy — the following types of content are prohibited:
{rules}

Video title: {title}
Description of the video's target and stance: {observation}

Based only on the description above, does the described content violate any of the policy rules?

Answer "Yes" or "No"."""


def build_binary_token_ids(tokenizer):
    """Map Yes/No labels to token IDs (copy of baseline helper)."""
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
    """Extract P(Yes)/(P(Yes)+P(No)) from logprobs at position 0."""
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
    """Build vLLM content list for video or frames (mirror of baseline)."""
    if media_type == "video":
        return [{"type": "video_url", "video_url": {"url": f"file://{media_path}"}}]
    jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")))
    if len(jpgs) > 8:
        indices = np.linspace(0, len(jpgs) - 1, 8, dtype=int)
        jpgs = [jpgs[i] for i in indices]
    return [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in jpgs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["MHClip_EN", "MHClip_ZH"])
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--transcript-limit", type=int, default=300)
    parser.add_argument("--observer-max-tokens", type=int, default=60)
    args = parser.parse_args()

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(
                log_dir, f"prompt_paradigm_obsjudge_{args.dataset}_{args.split}.log")),
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
    out_path = os.path.join(out_dir, f"{args.split}_obsjudge.jsonl")

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

    observer_params = SamplingParams(
        temperature=0,
        max_tokens=args.observer_max_tokens,
    )
    judge_params = SamplingParams(
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
        obs_messages = []
        obs_vid_ids = []
        obs_titles = []

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

            obs_text = OBSERVER_USER.format(
                title=title,
                transcript=transcript[:args.transcript_limit],
            )
            media_content = build_media_content(media_path, media_type)
            content = media_content + [{"type": "text", "text": obs_text}]
            msgs = [
                {"role": "system", "content": OBSERVER_SYSTEM},
                {"role": "user", "content": content},
            ]
            obs_messages.append(msgs)
            obs_vid_ids.append(vid_id)
            obs_titles.append(title)

        if not obs_messages:
            continue

        # --- Call 1: Observer ---
        observations = [None] * len(obs_messages)
        try:
            obs_outputs = llm.chat(messages=obs_messages, sampling_params=observer_params)
            for i, out in enumerate(obs_outputs):
                if out and out.outputs:
                    observations[i] = (out.outputs[0].text or "").strip()
        except Exception as e:
            err_msg = str(e)
            logging.error(f"  Observer batch failed: {err_msg}, falling back to single")
            for i, msgs in enumerate(obs_messages):
                try:
                    out_single = llm.chat(messages=[msgs], sampling_params=observer_params)
                    if out_single and out_single[0].outputs:
                        observations[i] = (out_single[0].outputs[0].text or "").strip()
                except Exception as e2:
                    logging.warning(f"  {obs_vid_ids[i]}: observer single failed: {e2}")
                    observations[i] = None

        # --- Call 2: Judge (text-only) ---
        judge_messages = []
        judge_index = []  # index into obs_vid_ids
        for i, obs_text in enumerate(observations):
            if obs_text is None or obs_text == "":
                continue
            jtext = JUDGE_USER.format(
                rules=rules_text,
                title=obs_titles[i],
                observation=obs_text,
            )
            jmsgs = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": jtext}]},
            ]
            judge_messages.append(jmsgs)
            judge_index.append(i)

        scores = [None] * len(obs_messages)
        if judge_messages:
            try:
                j_outputs = llm.chat(messages=judge_messages, sampling_params=judge_params)
                for k, out in enumerate(j_outputs):
                    s = extract_binary_score(out, label_token_ids)
                    scores[judge_index[k]] = s
            except Exception as e:
                err_msg = str(e)
                logging.error(f"  Judge batch failed: {err_msg}, falling back to single")
                for k, jmsgs in enumerate(judge_messages):
                    try:
                        j_single = llm.chat(messages=[jmsgs], sampling_params=judge_params)
                        scores[judge_index[k]] = extract_binary_score(j_single[0], label_token_ids)
                    except Exception as e2:
                        logging.warning(f"  {obs_vid_ids[judge_index[k]]}: judge single failed: {e2}")
                        scores[judge_index[k]] = None

        # --- Persist ---
        with open(out_path, "a") as f:
            for i in range(len(obs_vid_ids)):
                rec = {
                    "video_id": obs_vid_ids[i],
                    "score": scores[i],
                    "observation": observations[i],
                }
                if scores[i] is None:
                    rec["skipped"] = True
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        n_processed += len(obs_vid_ids)
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        logging.info(
            f"  [{len(done_ids)+n_processed}/{len(split_ids)}] rate={rate:.1f} vid/s"
        )

    logging.info(f"Done. scored={n_processed} skipped={n_skipped} out={out_path}")


if __name__ == "__main__":
    main()
