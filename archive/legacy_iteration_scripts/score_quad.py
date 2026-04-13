"""
Holistic scoring using quad images (2x2 grid per image, 8 images per video = 32 frames).

Each quad image contains 4 video frames in a 2x2 grid:
  top-left → top-right → bottom-left → bottom-right (chronological order).

Usage:
  python src/score_quad.py --dataset MHClip_EN --split test
  python src/score_quad.py --dataset MHClip_ZH --split test
"""

import argparse
import json
import logging
import os
import re
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
        "quad_dir": "datasets/MHClip_EN/quad",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
    },
    "MHClip_ZH": {
        "annotation": "datasets/MHClip_ZH/annotation(new).json",
        "splits_dir": "datasets/MHClip_ZH/splits",
        "quad_dir": "datasets/MHClip_ZH/quad",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
    },
}

YOUTUBE_RULES_TEXT = """R1: Encourage violence against groups based on protected status
R2: Incite hatred against groups based on protected status
R3: Dehumanize groups by comparing to non-human entities
R4: Praise or glorify violence against protected groups
R5: Use slurs and stereotypes that incite hatred (including through humor, irony, or coded language)
R6: Claim groups are inferior based on protected status
R7: Promote hateful supremacism
R8: Make conspiratorial claims about groups based on protected status
R9: Deny or minimize major violent events"""

BILIBILI_RULES_TEXT = """R1: Discriminatory content (violence, slurs, exclusion based on protected status)
R2: Attacks, disparagement, or belittlement targeting groups by protected attribute
R3: Verbal abuse and personal attacks based on identity
R4: Displaying or promoting hateful ideologies/symbols
R5: Mocking death, sickness, or disability
R6: Making jokes about disasters or tragedies
R7: Gender-based hate speech or discrimination
R8: Inciting confrontation or conflict between groups"""

OBSERVATION_PROMPT = """You are a content moderation analyst. You are given a series of images from a video. Each image is a 2x2 grid containing 4 consecutive video frames — read them left-to-right, top-to-bottom (top-left is earliest, bottom-right is latest). Together these images cover the full video timeline.

Title: {title}
Transcript: {transcript}

This platform's existing hate speech rules are:
{rules_text}

Task:
1. Determine if this video contains hateful or offensive content toward any group.
2. If hateful/offensive: identify which existing rules (if any) it violates, and what mechanism conveys the hate.

Respond with ONLY a JSON object in this exact format (no markdown, no code fences):
{{"verdict": "HATEFUL_OR_OFFENSIVE" or "NORMAL", "covered_rule_ids": ["R1", "R5"] or [], "target_group": "specific group targeted or none", "protected_attribute": "race|religion|gender|sexual_orientation|disability|nationality|ethnicity|age|none", "mechanism": "explicit_slurs|stereotyping_as_fact|humor_based_stereotyping|dehumanization|incitement_violence|incitement_hatred|glorification_violence|claims_inferiority|supremacism|conspiracy|denial_violence|ironic_mockery|coded_language|sarcastic_ridicule|visual_juxtaposition|OTHER:describe", "evidence_modality": "visual|text|audio_transcript|visual+text|visual+audio|text+audio|all", "observable_cues": "1-2 sentence description of specific cues observed"}}"""


RESIZE_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "quad_resized")


def get_quad_paths(quad_dir, vid_id, max_pixels=360000):
    """Get sorted quad image paths, resizing large images to fit token budget."""
    from PIL import Image

    d = os.path.join(quad_dir, vid_id)
    if not os.path.isdir(d):
        return []
    files = sorted(
        f for f in os.listdir(d) if f.endswith((".jpg", ".png", ".jpeg"))
    )
    paths = []
    for f in files:
        src = os.path.join(d, f)
        img = Image.open(src)
        w, h = img.size
        if w * h > max_pixels:
            # Resize and cache
            cache_dir = os.path.join(RESIZE_CACHE_DIR, vid_id)
            os.makedirs(cache_dir, exist_ok=True)
            dst = os.path.join(cache_dir, f)
            if not os.path.exists(dst):
                scale = (max_pixels / (w * h)) ** 0.5
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                img.save(dst, quality=85)
            paths.append(dst)
        else:
            paths.append(src)
        img.close()
    return paths


def parse_observation(text):
    """Parse JSON observation from MLLM response."""
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            v = obj.get("verdict", "").upper()
            if "HATEFUL" in v or "OFFENSIVE" in v:
                obj["verdict"] = "HATEFUL_OR_OFFENSIVE"
            else:
                obj["verdict"] = "NORMAL"
            obj.setdefault("covered_rule_ids", [])
            obj.setdefault("target_group", "none")
            obj.setdefault("protected_attribute", "none")
            obj.setdefault("mechanism", "none")
            obj.setdefault("evidence_modality", "unknown")
            obj.setdefault("observable_cues", "")
            return obj
        except json.JSONDecodeError:
            pass
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


def build_messages(quad_paths, title, transcript, platform):
    """Build chat messages with quad images."""
    rules_text = YOUTUBE_RULES_TEXT if platform == "youtube" else BILIBILI_RULES_TEXT

    prompt_text = OBSERVATION_PROMPT.format(
        title=title or "(no title)",
        transcript=(transcript or "(no transcript)")[:500],
        rules_text=rules_text,
    )

    content = []
    for fp in quad_paths:
        content.append({"type": "image_url", "image_url": {"url": f"file://{fp}"}})
    content.append({"type": "text", "text": prompt_text})

    return [
        {"role": "system", "content": "You are a content moderation analyst. Answer based strictly on observable evidence in the video frames, title, and transcript."},
        {"role": "user", "content": content},
    ]


def main():
    parser = argparse.ArgumentParser(description="Holistic scoring with quad images")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
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
            logging.FileHandler(os.path.join(log_dir, f"quad_{args.dataset}_{args.split}.log")),
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

    quad_dir = os.path.join(root, cfg["quad_dir"])

    # Output — supports resume
    out_dir = os.path.join(root, "results", "quad", args.dataset)
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
        logging.info(f"Resuming: {len(done_ids)} done")

    remaining = [v for v in split_ids if v not in done_ids]
    if not remaining:
        logging.info("All done.")
        return

    # Load vLLM
    logging.info(f"Loading vLLM: {args.model}")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 8},
        allowed_local_media_path="/data/jehc223",
        mm_processor_kwargs={"max_pixels": 360000},
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    t0 = time.time()
    n_processed = 0
    n_hateful = 0

    for batch_start in range(0, len(remaining), args.batch_size):
        batch_ids = remaining[batch_start:batch_start + args.batch_size]
        batch_messages = []
        batch_meta = []

        for vid_id in batch_ids:
            sample = id2sample[vid_id]
            title = sample.get(cfg["title_field"], "") or ""
            transcript = sample.get(cfg["transcript_field"], "") or ""
            quad_paths = get_quad_paths(quad_dir, vid_id)

            if not quad_paths:
                logging.warning(f"  {vid_id}: no quad images, skipping")
                continue

            msgs = build_messages(quad_paths, title, transcript, platform)
            batch_messages.append(msgs)
            batch_meta.append({"vid_id": vid_id, "n_quads": len(quad_paths)})

        if not batch_messages:
            continue

        try:
            outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params)
        except Exception as e:
            logging.error(f"  Batch failed: {e}")
            # Fallback to single
            for i, msgs in enumerate(batch_messages):
                try:
                    out_single = llm.chat(messages=[msgs], sampling_params=sampling_params)
                    resp = out_single[0].outputs[0].text.strip()
                    obs = parse_observation(resp)
                    obs["video_id"] = batch_meta[i]["vid_id"]
                    obs["n_quads"] = batch_meta[i]["n_quads"]
                    with open(out_path, "a") as f:
                        f.write(json.dumps(obs, ensure_ascii=False) + "\n")
                    n_processed += 1
                    if obs["verdict"] == "HATEFUL_OR_OFFENSIVE":
                        n_hateful += 1
                except Exception as e2:
                    logging.error(f"  {batch_meta[i]['vid_id']}: single failed: {e2}")
            continue

        with open(out_path, "a") as f:
            for i, output in enumerate(outputs):
                resp = output.outputs[0].text.strip()
                obs = parse_observation(resp)
                obs["video_id"] = batch_meta[i]["vid_id"]
                obs["n_quads"] = batch_meta[i]["n_quads"]
                f.write(json.dumps(obs, ensure_ascii=False) + "\n")
                n_processed += 1
                if obs["verdict"] == "HATEFUL_OR_OFFENSIVE":
                    n_hateful += 1

        elapsed = time.time() - t0
        total_done = len(done_ids) + n_processed
        rate = n_processed / elapsed if elapsed > 0 else 0
        logging.info(
            f"  [{total_done}/{len(split_ids)}] {rate:.1f} vid/s | "
            f"hateful={n_hateful}/{n_processed}"
        )

    logging.info(f"\nDone. {n_processed} processed. Hateful: {n_hateful}")
    logging.info(f"Output: {out_path}")


if __name__ == "__main__":
    main()
