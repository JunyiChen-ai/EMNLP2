"""
Observation-Grounded Classification (OGC) Pipeline.

Core principle: Separate MLLM rationale into grounded observations vs.
inferential interpretations. Use raw modalities (frame, audio) to verify
observations. Gate interpretation influence by grounding confidence.

Pipeline:
  1. Parse diagnostic rationale → observation_text + interpretation_text
  2. Encode each section separately (sentence-transformers)
  3. Extract frame features (CLIP ViT-L/14) and audio features (Wav2Vec2)
  4. Train OGC fusion classifier

Usage:
  # Full pipeline: extract features + train + evaluate
  python src/ogc_pipeline.py --dataset HateMM --phase all --seeds 10

  # Feature extraction only
  python src/ogc_pipeline.py --dataset HateMM --phase extract

  # Training only (features must exist)
  python src/ogc_pipeline.py --dataset HateMM --phase train --seeds 10
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path

# ── Dataset configs ──────────────────────────────────────────────────
DATASET_CONFIGS = {
    "HateMM": {
        "annotation": "datasets/HateMM/annotation(new).json",
        "splits_dir": "datasets/HateMM/splits",
        "frame_dir": "datasets/HateMM/frames",
        "audio_dir": "datasets/HateMM/audios",
        "id_field": "Video_ID",
        "label_field": "Label",
        "label_map": {"Hate": 1, "Non Hate": 0},
        "num_classes": 2,
    },
    "MHClip_EN": {
        "annotation": "datasets/MHClip_EN/annotation(new).json",
        "splits_dir": "datasets/MHClip_EN/splits",
        "frame_dir": "datasets/MHClip_EN/frames",
        "audio_dir": "datasets/MHClip_EN/audios",
        "id_field": "Video_ID",
        "label_field": "Label",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
        "num_classes": 2,
    },
    "MHClip_ZH": {
        "annotation": "datasets/MHClip_ZH/annotation(new).json",
        "splits_dir": "datasets/MHClip_ZH/splits",
        "frame_dir": "datasets/MHClip_ZH/frames",
        "audio_dir": "datasets/MHClip_ZH/audios",
        "id_field": "Video_ID",
        "label_field": "Label",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
        "num_classes": 2,
    },
}


# ═════════════════════════════════════════════════════════════════════
# STEP 1: Parse rationales into observation / interpretation
# ═════════════════════════════════════════════════════════════════════

def parse_diagnostic_rationale(rationale_raw: str):
    """Parse a diagnostic rationale into observation and interpretation text.

    Handles BOTH the old 5-section format (archive) and the new 4-section format.

    Old format sections:
      1. TEMPORAL EVIDENCE
      2. GROUNDED OBSERVATIONS → observation
      3. DIAGNOSTIC INTERPRETATION → interpretation
      4. STRONGEST BENIGN INTERPRETATION → interpretation
      5. DIAGNOSTIC VERDICT → interpretation

    New format sections:
      1. OBSERVED EVIDENCE → observation
      2. SOCIAL FRAME → interpretation
      3. AMBIGUITY CHECK → interpretation
      4. DECISION → interpretation
    """
    text = rationale_raw.strip()

    # Try to split by section headers (### N. or **N.)
    # Match patterns like "### 1. TEMPORAL EVIDENCE", "**1. OBSERVED EVIDENCE**"
    section_pattern = r'(?:^|\n)\s*(?:#{1,4}\s*)?(?:\*{1,2})?\s*(\d+)\.\s*([A-Z][A-Z\s/]+?)(?:\*{1,2})?\s*\n'
    sections = {}
    matches = list(re.finditer(section_pattern, text))

    if matches:
        for i, m in enumerate(matches):
            sec_num = int(m.group(1))
            sec_name = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections[sec_name] = text[start:end].strip()

    # Determine format and extract observation / interpretation
    observation_parts = []
    interpretation_parts = []

    for name, content in sections.items():
        name_upper = name.upper()
        # Old format: GROUNDED OBSERVATIONS → observation
        # New format: OBSERVED EVIDENCE → observation
        # Also treat TEMPORAL EVIDENCE as observation (it's grounded)
        if any(k in name_upper for k in ["OBSERVED EVIDENCE", "GROUNDED OBSERVATION", "TEMPORAL EVIDENCE"]):
            observation_parts.append(content)
        else:
            # Everything else is interpretation
            interpretation_parts.append(content)

    # Fallback: if parsing fails, use first 40% as observation, rest as interpretation
    if not observation_parts:
        lines = text.split("\n")
        split_point = max(1, len(lines) * 2 // 5)
        observation_parts = ["\n".join(lines[:split_point])]
        interpretation_parts = ["\n".join(lines[split_point:])]

    obs_text = "\n".join(observation_parts).strip()
    int_text = "\n".join(interpretation_parts).strip()

    return obs_text, int_text


# ═════════════════════════════════════════════════════════════════════
# STEP 2: Feature extraction
# ═════════════════════════════════════════════════════════════════════

def extract_text_embeddings(rationale_path, output_dir, project_root):
    """Encode observation and interpretation text separately using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    obs_path = os.path.join(output_dir, "obs_embeddings.npz")
    int_path = os.path.join(output_dir, "int_embeddings.npz")
    full_path = os.path.join(output_dir, "full_embeddings.npz")

    if os.path.exists(obs_path) and os.path.exists(int_path) and os.path.exists(full_path):
        print(f"Text embeddings already exist in {output_dir}")
        return

    # Load rationales
    records = []
    with open(rationale_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} rationales from {rationale_path}")

    # Parse into observation / interpretation
    vid_ids = []
    obs_texts = []
    int_texts = []
    full_texts = []
    for rec in records:
        vid_ids.append(rec["video_id"])
        raw = rec.get("rationale_raw", "")
        if raw.startswith("ERROR:"):
            obs_texts.append("")
            int_texts.append("")
            full_texts.append("")
        else:
            obs, interp = parse_diagnostic_rationale(raw)
            obs_texts.append(obs)
            int_texts.append(interp)
            full_texts.append(raw)

    # Encode with sentence-transformers
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding observation texts...")
    obs_embs = model.encode(obs_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print("Encoding interpretation texts...")
    int_embs = model.encode(int_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print("Encoding full rationale texts...")
    full_embs = model.encode(full_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    # Save as npz (vid_id → embedding)
    os.makedirs(output_dir, exist_ok=True)
    obs_dict = {vid: emb for vid, emb in zip(vid_ids, obs_embs)}
    int_dict = {vid: emb for vid, emb in zip(vid_ids, int_embs)}
    full_dict = {vid: emb for vid, emb in zip(vid_ids, full_embs)}

    np.savez(obs_path, **obs_dict)
    np.savez(int_path, **int_dict)
    np.savez(full_path, **full_dict)
    print(f"Saved text embeddings: {len(vid_ids)} samples, dim={obs_embs.shape[1]}")


def extract_frame_features(dataset_name, output_dir, project_root):
    """Extract CLIP ViT-L/14 features from sampled frames."""
    out_path = os.path.join(output_dir, "frame_features.npz")
    if os.path.exists(out_path):
        print(f"Frame features already exist: {out_path}")
        return

    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image

    cfg = DATASET_CONFIGS[dataset_name]
    ann_path = os.path.join(project_root, cfg["annotation"])
    with open(ann_path) as f:
        annotations = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    os.makedirs(output_dir, exist_ok=True)
    features = {}
    for i, sample in enumerate(annotations):
        vid_id = sample[cfg["id_field"]]
        frame_dir = os.path.join(project_root, cfg["frame_dir"], vid_id)
        if not os.path.isdir(frame_dir):
            continue

        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
        if not frame_files:
            continue

        # Sample up to 16 frames uniformly
        if len(frame_files) > 16:
            indices = np.linspace(0, len(frame_files) - 1, 16, dtype=int)
            frame_files = [frame_files[j] for j in indices]

        images = []
        for ff in frame_files:
            try:
                img = Image.open(os.path.join(frame_dir, ff)).convert("RGB")
                images.append(img)
            except Exception:
                continue

        if not images:
            continue

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            embs = model.get_image_features(**inputs)  # [N, 768]
            embs = embs / embs.norm(dim=-1, keepdim=True)
            feat = embs.mean(dim=0).cpu().numpy()  # average pool

        features[vid_id] = feat

        if (i + 1) % 100 == 0:
            print(f"  Frame features: {i + 1}/{len(annotations)}")

    np.savez(out_path, **features)
    print(f"Saved frame features: {len(features)} videos, dim={list(features.values())[0].shape[0] if features else '?'}")


def extract_audio_features(dataset_name, output_dir, project_root):
    """Extract Wav2Vec2 features from audio files."""
    out_path = os.path.join(output_dir, "audio_features.npz")
    if os.path.exists(out_path):
        print(f"Audio features already exist: {out_path}")
        return

    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import soundfile as sf

    cfg = DATASET_CONFIGS[dataset_name]
    ann_path = os.path.join(project_root, cfg["annotation"])
    with open(ann_path) as f:
        annotations = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval().to(device)

    os.makedirs(output_dir, exist_ok=True)
    features = {}
    for i, sample in enumerate(annotations):
        vid_id = sample[cfg["id_field"]]
        audio_path = os.path.join(project_root, cfg["audio_dir"], vid_id + ".wav")
        if not os.path.exists(audio_path):
            continue

        try:
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # stereo → mono
            # Truncate to 30 seconds max
            max_samples = sr * 30
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            with torch.no_grad():
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                feat = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # [768]
        except Exception as e:
            print(f"  Audio error for {vid_id}: {e}")
            continue

        features[vid_id] = feat

        if (i + 1) % 100 == 0:
            print(f"  Audio features: {i + 1}/{len(annotations)}")

    np.savez(out_path, **features)
    print(f"Saved audio features: {len(features)} videos, dim={list(features.values())[0].shape[0] if features else '?'}")


# ═════════════════════════════════════════════════════════════════════
# STEP 3: OGC Model
# ═════════════════════════════════════════════════════════════════════

class OGCDataset(Dataset):
    def __init__(self, vid_ids, labels, obs_feats, int_feats, frame_feats, audio_feats,
                 text_dim=384, av_dim=768):
        self.vid_ids = vid_ids
        self.labels = labels
        self.obs_feats = obs_feats
        self.int_feats = int_feats
        self.frame_feats = frame_feats
        self.audio_feats = audio_feats
        self.text_dim = text_dim
        self.av_dim = av_dim

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        vid = self.vid_ids[idx]
        return {
            "obs": torch.tensor(self.obs_feats.get(vid, np.zeros(self.text_dim, dtype=np.float32)), dtype=torch.float32),
            "interp": torch.tensor(self.int_feats.get(vid, np.zeros(self.text_dim, dtype=np.float32)), dtype=torch.float32),
            "frame": torch.tensor(self.frame_feats.get(vid, np.zeros(self.av_dim, dtype=np.float32)), dtype=torch.float32),
            "audio": torch.tensor(self.audio_feats.get(vid, np.zeros(self.av_dim, dtype=np.float32)), dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class OGCClassifier(nn.Module):
    """Observation-Grounded Classification.

    Core mechanism:
      g = sigmoid(MLP([obs_proj; frame_proj; audio_proj]))   # grounding score
      fused = g * interp_proj + (1 - g) * modality_avg       # gated representation
      logits = classifier([obs_proj; fused])                  # final prediction
    """

    def __init__(self, text_dim=384, av_dim=768, hidden=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.obs_proj = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.int_proj = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.frame_proj = nn.Sequential(nn.Linear(av_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.audio_proj = nn.Sequential(nn.Linear(av_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())

        # Grounding gate: measures how well raw modalities corroborate observations
        self.ground_gate = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, obs, interp, frame, audio):
        obs_h = self.obs_proj(obs)
        int_h = self.int_proj(interp)
        frame_h = self.frame_proj(frame)
        audio_h = self.audio_proj(audio)

        # Grounding score: how well do raw modalities match observations?
        g = self.ground_gate(torch.cat([obs_h, frame_h, audio_h], dim=-1))

        # Gated fusion: trust interpretation if grounded, else fall back to modalities
        modality_avg = (frame_h + audio_h) / 2
        fused = g * int_h + (1 - g) * modality_avg

        # Classify from [observations; gated_fusion]
        combined = torch.cat([obs_h, fused], dim=-1)
        logits = self.classifier(combined)
        return logits, g


class BaselineMLP(nn.Module):
    """Simple MLP on full rationale text (baseline for comparison)."""

    def __init__(self, text_dim=384, hidden=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ═════════════════════════════════════════════════════════════════════
# STEP 4: Training and Evaluation
# ═════════════════════════════════════════════════════════════════════

def load_split(splits_dir, split_name):
    """Load video IDs from a split file (CSV with one ID per line, no header)."""
    path = os.path.join(splits_dir, f"{split_name}.csv")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_features(feat_dir, text_dim=384, av_dim=768):
    """Load all feature files from a directory."""
    obs = dict(np.load(os.path.join(feat_dir, "obs_embeddings.npz"), allow_pickle=True))
    interp = dict(np.load(os.path.join(feat_dir, "int_embeddings.npz"), allow_pickle=True))
    full = dict(np.load(os.path.join(feat_dir, "full_embeddings.npz"), allow_pickle=True))

    frame_path = os.path.join(feat_dir, "frame_features.npz")
    frame = dict(np.load(frame_path, allow_pickle=True)) if os.path.exists(frame_path) else {}

    audio_path = os.path.join(feat_dir, "audio_features.npz")
    audio = dict(np.load(audio_path, allow_pickle=True)) if os.path.exists(audio_path) else {}

    return obs, interp, full, frame, audio


def train_ogc(dataset_name, feat_dir, project_root, seed, device="cuda",
              epochs=50, lr=1e-3, batch_size=64, hidden=256, dropout=0.3, patience=8):
    """Train OGC classifier for one seed. Returns test metrics."""
    cfg = DATASET_CONFIGS[dataset_name]

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load annotations for labels
    ann_path = os.path.join(project_root, cfg["annotation"])
    with open(ann_path) as f:
        annotations = json.load(f)
    id_to_label = {s[cfg["id_field"]]: cfg["label_map"].get(s[cfg["label_field"]], -1) for s in annotations}

    # Load splits
    splits_dir = os.path.join(project_root, cfg["splits_dir"])
    train_ids = load_split(splits_dir, "train")
    valid_ids = load_split(splits_dir, "valid")
    test_ids = load_split(splits_dir, "test")

    # Load features
    obs, interp, full, frame, audio = load_features(feat_dir)

    # Determine dimensions
    text_dim = next(iter(obs.values())).shape[0] if obs else 384
    av_dim = next(iter(frame.values())).shape[0] if frame else 768

    # Build datasets
    def make_dataset(ids):
        valid_ids_list = [v for v in ids if v in id_to_label and id_to_label[v] >= 0]
        labels = [id_to_label[v] for v in valid_ids_list]
        return OGCDataset(valid_ids_list, labels, obs, interp, frame, audio, text_dim, av_dim)

    train_ds = make_dataset(train_ids)
    valid_ds = make_dataset(valid_ids)
    test_ds = make_dataset(test_ids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ── Train OGC model ──────────────────────────────────────────
    model = OGCClassifier(text_dim=text_dim, av_dim=av_dim, hidden=hidden,
                          num_classes=cfg["num_classes"], dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            obs_b = batch["obs"].to(device)
            int_b = batch["interp"].to(device)
            frame_b = batch["frame"].to(device)
            audio_b = batch["audio"].to(device)
            labels_b = batch["label"].to(device)

            logits, g = model(obs_b, int_b, frame_b, audio_b)
            loss = criterion(logits, labels_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                logits, _ = model(
                    batch["obs"].to(device), batch["interp"].to(device),
                    batch["frame"].to(device), batch["audio"].to(device),
                )
                val_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                val_labels.extend(batch["label"].tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # ── Test ──────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels, test_gates = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, g = model(
                batch["obs"].to(device), batch["interp"].to(device),
                batch["frame"].to(device), batch["audio"].to(device),
            )
            test_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            test_labels.extend(batch["label"].tolist())
            test_gates.extend(g.squeeze(-1).cpu().tolist())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    avg_gate = np.mean(test_gates)

    return {
        "seed": seed,
        "test_acc": round(test_acc * 100, 2),
        "test_f1": round(test_f1 * 100, 2),
        "val_acc": round(best_val_acc * 100, 2),
        "avg_grounding_gate": round(avg_gate, 4),
        "model_state": best_state,
    }


def train_baseline(dataset_name, feat_dir, project_root, seed, device="cuda",
                   epochs=50, lr=1e-3, batch_size=64, hidden=256, dropout=0.3, patience=8):
    """Train baseline MLP on full rationale text for comparison."""
    cfg = DATASET_CONFIGS[dataset_name]

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ann_path = os.path.join(project_root, cfg["annotation"])
    with open(ann_path) as f:
        annotations = json.load(f)
    id_to_label = {s[cfg["id_field"]]: cfg["label_map"].get(s[cfg["label_field"]], -1) for s in annotations}

    splits_dir = os.path.join(project_root, cfg["splits_dir"])
    train_ids = load_split(splits_dir, "train")
    valid_ids = load_split(splits_dir, "valid")
    test_ids = load_split(splits_dir, "test")

    _, _, full, _, _ = load_features(feat_dir)
    text_dim = next(iter(full.values())).shape[0] if full else 384

    def get_ids_labels(ids):
        valid = [(v, id_to_label[v]) for v in ids if v in id_to_label and id_to_label[v] >= 0]
        return [v for v, _ in valid], [l for _, l in valid]

    train_vids, train_labels = get_ids_labels(train_ids)
    valid_vids, valid_labels = get_ids_labels(valid_ids)
    test_vids, test_labels_list = get_ids_labels(test_ids)

    def make_tensors(vids, labels):
        X = torch.stack([torch.tensor(full.get(v, np.zeros(text_dim, dtype=np.float32)), dtype=torch.float32) for v in vids])
        y = torch.tensor(labels, dtype=torch.long)
        return X, y

    X_train, y_train = make_tensors(train_vids, train_labels)
    X_valid, y_valid = make_tensors(valid_vids, valid_labels)
    X_test, y_test = make_tensors(test_vids, test_labels_list)

    model = BaselineMLP(text_dim=text_dim, hidden=hidden, num_classes=cfg["num_classes"], dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_train[idx].to(device))
            loss = criterion(logits, y_train[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_valid.to(device))
            val_preds = val_logits.argmax(dim=-1).cpu()
            val_acc = accuracy_score(y_valid, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(device))
        test_preds = test_logits.argmax(dim=-1).cpu()

    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")

    return {
        "seed": seed,
        "test_acc": round(test_acc * 100, 2),
        "test_f1": round(test_f1 * 100, 2),
        "val_acc": round(best_val_acc * 100, 2),
    }


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OGC Pipeline")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--phase", default="all", choices=["extract", "train", "all"])
    parser.add_argument("--prompt-family", default="diagnostic")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--project-root", default="/data/jehc223/EMNLP2")
    args = parser.parse_args()

    project_root = args.project_root
    rationale_dir = os.path.join(project_root, "rationales", args.dataset, args.prompt_family)
    feat_dir = os.path.join(project_root, "embeddings", args.dataset, args.prompt_family)
    result_dir = os.path.join(project_root, "results", args.dataset, args.prompt_family, "ogc")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rationale_path = os.path.join(rationale_dir, "rationales.jsonl")

    # ── Phase: Extract ───────────────────────────────────────────
    if args.phase in ("extract", "all"):
        if not os.path.exists(rationale_path):
            print(f"ERROR: Rationale file not found: {rationale_path}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"EXTRACTING FEATURES: {args.dataset} / {args.prompt_family}")
        print(f"{'='*60}")

        extract_text_embeddings(rationale_path, feat_dir, project_root)
        extract_frame_features(args.dataset, feat_dir, project_root)
        extract_audio_features(args.dataset, feat_dir, project_root)
        print("Feature extraction complete.\n")

    # ── Phase: Train ─────────────────────────────────────────────
    if args.phase in ("train", "all"):
        print(f"\n{'='*60}")
        print(f"TRAINING: {args.dataset} / OGC / {args.seeds} seeds")
        print(f"{'='*60}")

        ogc_results = []
        baseline_results = []

        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")

            # OGC
            ogc_r = train_ogc(
                args.dataset, feat_dir, project_root, seed, device,
                args.epochs, args.lr, args.batch_size, args.hidden, args.dropout, args.patience,
            )
            ogc_results.append(ogc_r)
            print(f"  OGC:      ACC={ogc_r['test_acc']:.2f}  F1={ogc_r['test_f1']:.2f}  gate={ogc_r['avg_grounding_gate']:.4f}")

            # Baseline
            base_r = train_baseline(
                args.dataset, feat_dir, project_root, seed, device,
                args.epochs, args.lr, args.batch_size, args.hidden, args.dropout, args.patience,
            )
            baseline_results.append(base_r)
            print(f"  Baseline: ACC={base_r['test_acc']:.2f}  F1={base_r['test_f1']:.2f}")

        # ── Summary ──────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY: {args.dataset}")
        print(f"{'='*60}")

        ogc_accs = [r["test_acc"] for r in ogc_results]
        ogc_f1s = [r["test_f1"] for r in ogc_results]
        base_accs = [r["test_acc"] for r in baseline_results]
        base_f1s = [r["test_f1"] for r in baseline_results]

        print(f"\nOGC ({args.seeds} seeds):")
        print(f"  ACC: {np.mean(ogc_accs):.2f} +/- {np.std(ogc_accs):.2f}  (max={np.max(ogc_accs):.2f}, min={np.min(ogc_accs):.2f})")
        print(f"  F1:  {np.mean(ogc_f1s):.2f} +/- {np.std(ogc_f1s):.2f}  (max={np.max(ogc_f1s):.2f}, min={np.min(ogc_f1s):.2f})")
        print(f"  Avg grounding gate: {np.mean([r['avg_grounding_gate'] for r in ogc_results]):.4f}")

        print(f"\nBaseline MLP ({args.seeds} seeds):")
        print(f"  ACC: {np.mean(base_accs):.2f} +/- {np.std(base_accs):.2f}  (max={np.max(base_accs):.2f}, min={np.min(base_accs):.2f})")
        print(f"  F1:  {np.mean(base_f1s):.2f} +/- {np.std(base_f1s):.2f}  (max={np.max(base_f1s):.2f}, min={np.min(base_f1s):.2f})")

        delta_acc = np.mean(ogc_accs) - np.mean(base_accs)
        delta_f1 = np.mean(ogc_f1s) - np.mean(base_f1s)
        print(f"\nDelta (OGC - Baseline):")
        print(f"  ACC: {delta_acc:+.2f}")
        print(f"  F1:  {delta_f1:+.2f}")

        # Save detailed results
        result_file = os.path.join(result_dir, "metrics.json")
        results_payload = {
            "dataset": args.dataset,
            "prompt_family": args.prompt_family,
            "model": "OGC",
            "seeds": args.seeds,
            "ogc": {
                "per_seed": ogc_results,
                "mean_acc": round(np.mean(ogc_accs), 2),
                "std_acc": round(np.std(ogc_accs), 2),
                "max_acc": round(np.max(ogc_accs), 2),
                "mean_f1": round(np.mean(ogc_f1s), 2),
                "std_f1": round(np.std(ogc_f1s), 2),
                "max_f1": round(np.max(ogc_f1s), 2),
            },
            "baseline_mlp": {
                "per_seed": baseline_results,
                "mean_acc": round(np.mean(base_accs), 2),
                "std_acc": round(np.std(base_accs), 2),
                "max_acc": round(np.max(base_accs), 2),
                "mean_f1": round(np.mean(base_f1s), 2),
                "std_f1": round(np.std(base_f1s), 2),
                "max_f1": round(np.max(base_f1s), 2),
            },
            "delta": {
                "acc": round(delta_acc, 2),
                "f1": round(delta_f1, 2),
            },
            "config": {
                "hidden": args.hidden,
                "dropout": args.dropout,
                "lr": args.lr,
                "epochs": args.epochs,
                "patience": args.patience,
            },
        }
        # Remove model_state from saved results (not JSON serializable)
        for r in results_payload["ogc"]["per_seed"]:
            r.pop("model_state", None)

        with open(result_file, "w") as f:
            json.dump(results_payload, f, indent=2)
        print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()
