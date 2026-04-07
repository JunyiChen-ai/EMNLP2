"""
Extract frame features (ViT/CLIP) and audio features (Wav2Vec2/CLAP) for downstream fusion.
Also encode rationale text via sentence-transformers.

Usage:
  python src/extract_features.py --dataset HateMM --feature-type frames
  python src/extract_features.py --dataset HateMM --feature-type audio
  python src/extract_features.py --dataset HateMM --feature-type rationale --prompt-family diagnostic
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path

DATASET_CONFIGS = {
    "HateMM": {
        "annotation": "datasets/HateMM/annotation(new).json",
        "frame_dir": "datasets/HateMM/frames",
        "audio_dir": "datasets/HateMM/audios",
        "id_field": "Video_ID",
        "label_field": "Label",
        "label_map": {"Hate": 1, "Non Hate": 0},
    },
    "MHClip_EN": {
        "annotation": "datasets/MHClip_EN/annotation(new).json",
        "frame_dir": "datasets/MHClip_EN/frames",
        "audio_dir": "datasets/MHClip_EN/audios",
        "id_field": "Video_ID",
        "label_field": "Label",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    },
    "MHClip_ZH": {
        "annotation": "datasets/MHClip_ZH/annotation(new).json",
        "frame_dir": "datasets/MHClip_ZH/frames",
        "audio_dir": "datasets/MHClip_ZH/audios",
        "id_field": "Video_ID",
        "label_field": "Label",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    },
}


def extract_frame_features(dataset_name, project_root, output_dir):
    """Extract CLIP ViT-L/14 features from sampled frames."""
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image

    cfg = DATASET_CONFIGS[dataset_name]
    ann_path = os.path.join(project_root, cfg["annotation"])
    with open(ann_path) as f:
        annotations = json.load(f)

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().cuda()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "frame_features.npz")
    if os.path.exists(out_path):
        print(f"Frame features already exist: {out_path}")
        return

    features = {}
    for i, sample in enumerate(annotations):
        vid_id = sample[cfg["id_field"]]
        frame_dir = os.path.join(project_root, cfg["frame_dir"], vid_id)
        if not os.path.isdir(frame_dir):
            print(f"  Skip {vid_id}: no frames")
            continue

        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
        if not frame_files:
            continue

        # Sample up to 16 frames uniformly
        if len(frame_files) > 16:
            indices = np.linspace(0, len(frame_files)-1, 16, dtype=int)
            frame_files = [frame_files[j] for j in indices]

        images = []
        for ff in frame_files:
            img = Image.open(os.path.join(frame_dir, ff)).convert("RGB")
            images.append(img)

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt", padding=True).to("cuda")
            outputs = model.get_image_features(**inputs)  # [N, 768]
            feat = outputs.cpu().numpy().mean(axis=0)  # mean pool → [768]

        features[vid_id] = feat
        if (i + 1) % 100 == 0:
            print(f"  Frames: {i+1}/{len(annotations)}")

    np.savez_compressed(out_path, **features)
    print(f"Saved {len(features)} frame features to {out_path}")


def extract_audio_features(dataset_name, project_root, output_dir):
    """Extract Wav2Vec2 features from audio files."""
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import librosa

    cfg = DATASET_CONFIGS[dataset_name]
    ann_path = os.path.join(project_root, cfg["annotation"])
    with open(ann_path) as f:
        annotations = json.load(f)

    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval().cuda()
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "audio_features.npz")
    if os.path.exists(out_path):
        print(f"Audio features already exist: {out_path}")
        return

    features = {}
    for i, sample in enumerate(annotations):
        vid_id = sample[cfg["id_field"]]
        audio_path = os.path.join(project_root, cfg["audio_dir"], vid_id + ".wav")
        if not os.path.exists(audio_path):
            continue

        try:
            waveform, sr = librosa.load(audio_path, sr=16000, duration=30)
            with torch.no_grad():
                inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True).to("cuda")
                outputs = model(**inputs)
                feat = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # [768]
            features[vid_id] = feat
        except Exception as e:
            print(f"  Audio error {vid_id}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  Audio: {i+1}/{len(annotations)}")

    np.savez_compressed(out_path, **features)
    print(f"Saved {len(features)} audio features to {out_path}")


def extract_rationale_features(dataset_name, project_root, output_dir, prompt_family):
    """Encode rationale text using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    rationale_path = os.path.join(project_root, "rationales", dataset_name, prompt_family, "rationales.jsonl")
    if not os.path.exists(rationale_path):
        print(f"Rationale file not found: {rationale_path}")
        sys.exit(1)

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

    records = []
    with open(rationale_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"rationale_features_{prompt_family}.npz")
    if os.path.exists(out_path):
        print(f"Rationale features already exist: {out_path}")
        return

    texts = [r["rationale_raw"] for r in records]
    vid_ids = [r["video_id"] for r in records]

    # Encode in batches
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)  # [N, 768]

    features = {}
    for vid_id, emb in zip(vid_ids, embeddings):
        features[vid_id] = emb

    np.savez_compressed(out_path, **features)
    print(f"Saved {len(features)} rationale features to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--feature-type", required=True, choices=["frames", "audio", "rationale"])
    parser.add_argument("--prompt-family", default="diagnostic", choices=["hvguard", "mars", "diagnostic"])
    parser.add_argument("--project-root", default="/data/jehc223/EMNLP2")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.project_root, "embeddings", args.dataset)

    if args.feature_type == "frames":
        extract_frame_features(args.dataset, args.project_root, args.output_dir)
    elif args.feature_type == "audio":
        extract_audio_features(args.dataset, args.project_root, args.output_dir)
    elif args.feature_type == "rationale":
        extract_rationale_features(args.dataset, args.project_root, args.output_dir, args.prompt_family)


if __name__ == "__main__":
    main()
