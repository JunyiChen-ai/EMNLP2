"""
Step 3: CLIP relevance scanning + frame selection.

For each video × each rule:
  3a. Compute CLIP cosine similarity (rule text vs video frames / title+transcript)
  3b. If relevant, select top-K frames for the MLLM scoring phase

CLIP runs on GPU alone — vLLM is NOT loaded in this script.

Usage:
  python src/clip_filter.py --dataset HateMM --split test --K 8
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

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
    },
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


# ── CLIP helpers ────────────────────────────────────────────────────

def load_clip(device="cuda"):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor, device


def clip_encode_text(model, processor, text, device):
    inputs = processor(text=[text[:300]], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    return emb[0].cpu().numpy()


def clip_encode_frames(model, processor, frame_paths, device):
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embs = model.get_image_features(**inputs)
    return embs.cpu().numpy()


def cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_n, b_n))


def get_frame_paths(frames_dir, vid_id, max_frames=32):
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


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLIP relevance scanning + frame selection")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    parser.add_argument("--K", type=int, default=8, help="Top-K frames per rule")
    parser.add_argument("--relevance-threshold", type=float, default=0.22)
    parser.add_argument("--frame-threshold", type=float, default=0.1)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--constitution-suffix", default="",
                        help="Suffix for constitution file, e.g. '_merged'")
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    root = args.project_root
    constitution_name = DATASET_CONSTITUTION[args.dataset]
    out_suffix = args.constitution_suffix

    # Logging
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"clip_filter_{args.dataset}_{args.split}{out_suffix}.log")),
            logging.StreamHandler(),
        ],
    )

    # Load preconditions (for objectified rule texts)
    precond_path = os.path.join(root, "constitution", f"preconditions_{constitution_name}{args.constitution_suffix}.json")
    if not os.path.exists(precond_path):
        logging.error(f"Preconditions not found: {precond_path}")
        sys.exit(1)
    with open(precond_path) as f:
        rules = json.load(f)
    logging.info(f"Loaded {len(rules)} rules from {constitution_name}")

    # Load annotations + split
    with open(os.path.join(root, cfg["annotation"])) as f:
        annotations = json.load(f)
    id2sample = {s[cfg["id_field"]]: s for s in annotations}

    split_path = os.path.join(root, cfg["splits_dir"], f"{args.split}.csv")
    with open(split_path) as f:
        split_ids = [line.strip() for line in f if line.strip()]
    split_ids = [v for v in split_ids if v in id2sample]
    logging.info(f"Dataset={args.dataset} split={args.split} n={len(split_ids)}")

    # Load CLIP on GPU
    logging.info("Loading CLIP on GPU...")
    clip_model, clip_processor, device = load_clip("cuda")

    # Pre-encode rule texts
    rule_embs = {}
    for rule in rules:
        rule_embs[rule["rule_id"]] = clip_encode_text(
            clip_model, clip_processor, rule["objectified_rule"], device
        )
    logging.info(f"Encoded {len(rule_embs)} rule texts")

    # Process each video
    frames_dir = os.path.join(root, cfg["frames_dir"])
    results = {}
    t0 = time.time()

    for idx, vid_id in enumerate(split_ids):
        sample = id2sample[vid_id]
        title = sample.get(cfg["title_field"], "") or ""
        transcript = sample.get(cfg["transcript_field"], "") or ""

        # Get frame paths
        all_frame_paths = get_frame_paths(frames_dir, vid_id, args.max_frames)

        # Encode frames
        frame_embs = None
        if all_frame_paths:
            try:
                frame_embs = clip_encode_frames(clip_model, clip_processor, all_frame_paths, device)
            except Exception as e:
                logging.warning(f"  {vid_id}: CLIP frame encoding failed: {e}")

        # Encode text
        text_for_clip = (title + " " + transcript)[:300]
        text_emb = clip_encode_text(clip_model, clip_processor, text_for_clip, device)

        video_result = {}

        for rule in rules:
            rule_id = rule["rule_id"]
            rule_emb = rule_embs[rule_id]

            # 3a: Rule-level relevance
            video_emb = frame_embs.mean(axis=0) if frame_embs is not None else np.zeros(768)
            rel_visual = cosine_sim(video_emb, rule_emb) if frame_embs is not None else 0.0
            rel_text = cosine_sim(text_emb, rule_emb)
            relevance = max(rel_visual, rel_text)

            if relevance < args.relevance_threshold:
                video_result[rule_id] = {
                    "relevant": False,
                    "relevance_score": round(relevance, 4),
                }
                continue

            # 3b: Frame selection (top-K)
            selected_indices = []
            text_only = False

            if frame_embs is not None and len(all_frame_paths) > 0:
                frame_scores = [cosine_sim(frame_embs[i], rule_emb) for i in range(len(all_frame_paths))]
                max_frame_score = max(frame_scores) if frame_scores else 0.0

                if max_frame_score < args.frame_threshold:
                    text_only = True
                else:
                    top_k = sorted(range(len(frame_scores)), key=lambda i: -frame_scores[i])[:args.K]
                    selected_indices = sorted(top_k)  # restore temporal order
            else:
                text_only = True

            video_result[rule_id] = {
                "relevant": True,
                "relevance_score": round(relevance, 4),
                "selected_frame_indices": selected_indices,
                "text_only": text_only,
            }

        results[vid_id] = video_result

        if (idx + 1) % 50 == 0 or idx == len(split_ids) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logging.info(f"  [{idx+1}/{len(split_ids)}] {rate:.1f} vid/s")

    # Save output
    out_dir = os.path.join(root, "results", "clip_filter", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}{out_suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved CLIP filter results for {len(results)} videos to {out_path}")

    # Summary stats
    total_relevant = 0
    total_text_only = 0
    total_pairs = 0
    for vid_data in results.values():
        for rule_data in vid_data.values():
            total_pairs += 1
            if rule_data.get("relevant"):
                total_relevant += 1
                if rule_data.get("text_only"):
                    total_text_only += 1
    logging.info(f"Total video-rule pairs: {total_pairs}")
    logging.info(f"Relevant: {total_relevant} ({100*total_relevant/max(total_pairs,1):.1f}%)")
    logging.info(f"Text-only (no relevant frames): {total_text_only}")


if __name__ == "__main__":
    main()
