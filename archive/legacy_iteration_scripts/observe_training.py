"""
TMBD Step A: Structured multimodal observation on training split.

For each training video, makes 1 MLLM multimodal call with 8 uniformly-sampled
frames + title + transcript. Collects structured JSON observations:
  verdict, covered_rule_ids, target_group, protected_attribute,
  mechanism, evidence_modality, observable_cues.

vLLM only — no CLIP loaded in this script.

Usage:
  python src/observe_training.py --dataset MHClip_EN
  python src/observe_training.py --dataset MHClip_ZH
"""

import argparse
import json
import logging
import os
import re
import sys
import time

import numpy as np

PROJECT_ROOT = "/data/jehc223/EMNLP2"

DATASET_CONSTITUTION = {
    "MHClip_EN": "youtube",
    "MHClip_ZH": "bilibili",
}

DATASET_CONFIGS = {
    "MHClip_EN": {
        "annotation": "datasets/MHClip_EN/annotation(new).json",
        "splits_dir": "datasets/MHClip_EN/splits",
        "frames_dir": "datasets/MHClip_EN/frames",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
    },
    "MHClip_ZH": {
        "annotation": "datasets/MHClip_ZH/annotation(new).json",
        "splits_dir": "datasets/MHClip_ZH/splits",
        "frames_dir": "datasets/MHClip_ZH/frames",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
    },
}

# Platform-specific observation prompts
# NOTE: "audio" cues limited to transcript-observable content (not prosody/tone)

YOUTUBE_RULES_TEXT = """R1: Encourage violence against groups based on protected status
R2: Incite hatred against groups based on protected status
R3: Dehumanize groups by comparing to non-human entities
R4: Praise or glorify violence against protected groups
R5: Use slurs and stereotypes that incite hatred
R6: Claim groups are inferior based on protected status
R7: Promote hateful supremacism
R8: Make conspiratorial claims about groups based on protected status
R9: Deny or minimize major violent events"""

YOUTUBE_MECHANISMS = (
    "incitement_violence|incitement_hatred|dehumanization|glorification_violence|"
    "explicit_slurs|stereotyping_as_fact|claims_inferiority|supremacism|conspiracy|"
    "denial_violence|ironic_mockery|coded_language|sarcastic_ridicule|"
    "contextual_dog_whistle|humor_based_stereotyping|visual_juxtaposition|OTHER:describe"
)

BILIBILI_RULES_TEXT = """R1: Discriminatory content (violence, slurs, exclusion based on protected status)
R2: Attacks, disparagement, or belittlement targeting groups by protected attribute
R3: Verbal abuse and personal attacks based on identity
R4: Displaying or promoting hateful ideologies/symbols
R5: Mocking death, sickness, or disability
R6: Making jokes about disasters or tragedies
R7: Gender-based hate speech or discrimination
R8: Inciting confrontation or conflict between groups"""

BILIBILI_MECHANISMS = (
    "discrimination_violence|attacks_disparagement|verbal_abuse|hateful_ideologies|"
    "mocking_death_disability|disaster_jokes|gender_based_hate|inciting_confrontation|"
    "ironic_mockery|coded_language|sarcastic_ridicule|contextual_dog_whistle|"
    "regional_slang_hate|implicit_gender_ridicule|cultural_stereotype_humor|"
    "visual_juxtaposition|OTHER:describe"
)

OBSERVATION_PROMPT_TEMPLATE = """You are a content moderation analyst. Observe this video's visual frames, title, and transcript carefully. "Audio" refers only to what is written in the transcript.

Title: {title}
Transcript: {transcript}

This platform's existing hate speech rules are:
{rules_text}

Task:
1. Determine if this video contains hateful or offensive content toward any group.
2. If hateful/offensive: identify which existing rules (if any) it violates, and what mechanism conveys the hate.

Respond with ONLY a JSON object in this exact format (no markdown, no code fences):
{{"verdict": "HATEFUL_OR_OFFENSIVE or NORMAL", "covered_rule_ids": ["R1", "R5"] or [], "target_group": "specific group targeted or none", "protected_attribute": "race|religion|gender|sexual_orientation|disability|nationality|ethnicity|age|none", "mechanism": "{mechanisms}", "evidence_modality": "visual|text|audio_transcript|visual+text|visual+audio|text+audio|all", "observable_cues": "1-2 sentence description of specific cues observed"}}"""


def get_uniform_frames(frames_dir, vid_id, n_frames=8):
    """Get n uniformly-sampled frames (temporal uniform, NOT CLIP-selected)."""
    d = os.path.join(frames_dir, vid_id)
    if not os.path.isdir(d):
        return []
    files = sorted(
        (f for f in os.listdir(d) if f.endswith((".jpg", ".png", ".jpeg"))),
        key=lambda f: int(''.join(c for c in os.path.splitext(f)[0] if c.isdigit()) or '0'),
    )
    if not files:
        return []
    if len(files) <= n_frames:
        return [os.path.join(d, f) for f in files]
    indices = np.linspace(0, len(files) - 1, n_frames).round().astype(int)
    seen = set()
    selected = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            selected.append(files[i])
    return [os.path.join(d, f) for f in selected]


def build_messages(frame_paths, title, transcript, platform):
    """Build chat messages for observation prompt."""
    if platform == "youtube":
        rules_text = YOUTUBE_RULES_TEXT
        mechanisms = YOUTUBE_MECHANISMS
    else:
        rules_text = BILIBILI_RULES_TEXT
        mechanisms = BILIBILI_MECHANISMS

    prompt_text = OBSERVATION_PROMPT_TEMPLATE.format(
        title=title or "(no title)",
        transcript=(transcript or "(no transcript)")[:300],
        rules_text=rules_text,
        mechanisms=mechanisms,
    )

    content = []
    for fp in frame_paths:
        content.append({"type": "image_url", "image_url": {"url": f"file://{fp}"}})
    content.append({"type": "text", "text": prompt_text})

    return [
        {"role": "system", "content": "You are a content moderation analyst. Answer based strictly on observable evidence in the frames, title, and transcript. Do not speculate."},
        {"role": "user", "content": content},
    ]


def parse_observation(text):
    """Parse JSON observation from MLLM response."""
    # Try to find JSON object in response
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            # Normalize verdict
            v = obj.get("verdict", "").upper()
            if "HATEFUL" in v or "OFFENSIVE" in v:
                obj["verdict"] = "HATEFUL_OR_OFFENSIVE"
            else:
                obj["verdict"] = "NORMAL"
            # Ensure all fields exist
            obj.setdefault("covered_rule_ids", [])
            obj.setdefault("target_group", "none")
            obj.setdefault("protected_attribute", "none")
            obj.setdefault("mechanism", "none")
            obj.setdefault("evidence_modality", "unknown")
            obj.setdefault("observable_cues", "")
            return obj
        except json.JSONDecodeError:
            pass

    # Fallback: emit PARSE_ERROR — exclude from discovery to avoid poisoning
    return {
        "verdict": "PARSE_ERROR",
        "covered_rule_ids": [],
        "target_group": "none",
        "protected_attribute": "none",
        "mechanism": "none",
        "evidence_modality": "none",
        "observable_cues": text[:200],
        "_parse_fallback": True,
    }


def main():
    parser = argparse.ArgumentParser(description="TMBD Step A: Structured observation")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--n-frames", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    root = args.project_root
    platform = DATASET_CONSTITUTION[args.dataset]

    # Logging
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"observe_{args.dataset}_{args.split}.log")),
            logging.StreamHandler(),
        ],
    )

    # Load annotations + train split
    with open(os.path.join(root, cfg["annotation"])) as f:
        annotations = json.load(f)
    id2sample = {s[cfg["id_field"]]: s for s in annotations}

    split_path = os.path.join(root, cfg["splits_dir"], f"{args.split}.csv")
    with open(split_path) as f:
        train_ids = [line.strip() for line in f if line.strip()]
    train_ids = [v for v in train_ids if v in id2sample]
    logging.info(f"Dataset={args.dataset} platform={platform} split={args.split} n={len(train_ids)}")

    # Output file — supports resume
    out_dir = os.path.join(root, "results", "observations", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}.jsonl")

    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("video_id"):
                            done_ids.add(rec["video_id"])
                    except json.JSONDecodeError:
                        pass
        logging.info(f"Resuming: {len(done_ids)} already done")

    remaining = [v for v in train_ids if v not in done_ids]
    if not remaining:
        logging.info("All videos already processed.")
        return

    # Load vLLM
    logging.info(f"Loading vLLM model: {args.model}")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_model_len=16384,
        limit_mm_per_prompt={"image": args.n_frames},
        allowed_local_media_path="/data/jehc223",
        mm_processor_kwargs={"max_pixels": 1003520},
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
    )

    frames_dir = os.path.join(root, cfg["frames_dir"])
    t0 = time.time()
    n_hateful = 0
    n_processed = 0

    # Process in batches
    for batch_start in range(0, len(remaining), args.batch_size):
        batch_ids = remaining[batch_start:batch_start + args.batch_size]
        batch_messages = []
        batch_meta = []

        for vid_id in batch_ids:
            sample = id2sample[vid_id]
            title = sample.get(cfg["title_field"], "") or ""
            transcript = sample.get(cfg["transcript_field"], "") or ""
            frame_paths = get_uniform_frames(frames_dir, vid_id, args.n_frames)

            if not frame_paths:
                # Skip videos with no frames — C2 requires multimodal input
                logging.warning(f"  {vid_id}: no frames, skipping (C2: must be multimodal)")
                continue

            msgs = build_messages(frame_paths, title, transcript, platform)
            batch_messages.append(msgs)
            batch_meta.append({"vid_id": vid_id, "n_frames": len(frame_paths)})

        if not batch_messages:
            continue

        # Batch inference
        try:
            outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params)
        except Exception as e:
            logging.error(f"  Batch failed: {e}")
            # Fall back to single-item processing
            for i, msgs in enumerate(batch_messages):
                try:
                    outputs_single = llm.chat(messages=[msgs], sampling_params=sampling_params)
                    response_text = outputs_single[0].outputs[0].text.strip()
                    obs = parse_observation(response_text)
                    obs["video_id"] = batch_meta[i]["vid_id"]
                    obs["n_frames"] = batch_meta[i]["n_frames"]
                    obs["_raw"] = response_text[:500]

                    with open(out_path, "a") as f:
                        f.write(json.dumps(obs, ensure_ascii=False) + "\n")
                    n_processed += 1
                    if obs["verdict"] == "HATEFUL_OR_OFFENSIVE":
                        n_hateful += 1
                except Exception as e2:
                    logging.error(f"  {batch_meta[i]['vid_id']}: single failed: {e2}")
            continue

        # Process batch outputs
        with open(out_path, "a") as f:
            for i, output in enumerate(outputs):
                response_text = output.outputs[0].text.strip()
                obs = parse_observation(response_text)
                obs["video_id"] = batch_meta[i]["vid_id"]
                obs["n_frames"] = batch_meta[i]["n_frames"]
                obs["_raw"] = response_text[:500]

                f.write(json.dumps(obs, ensure_ascii=False) + "\n")
                n_processed += 1
                if obs["verdict"] == "HATEFUL_OR_OFFENSIVE":
                    n_hateful += 1

        elapsed = time.time() - t0
        total_done = len(done_ids) + n_processed
        rate = n_processed / elapsed if elapsed > 0 else 0
        logging.info(
            f"  [{total_done}/{len(train_ids)}] {rate:.1f} vid/s | "
            f"hateful_so_far={n_hateful}/{n_processed}"
        )

    logging.info(f"\nDone. Processed {n_processed} videos. Hateful: {n_hateful}")
    logging.info(f"Output: {out_path}")


if __name__ == "__main__":
    main()
