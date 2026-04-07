"""
Oracle analysis v2:
1. Small model = multi-head MLP routing on text + frame + audio (multimodal)
2. Large model = MLLM zero-shot
3. Error pattern analysis for both models
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, Wav2Vec2Processor, Wav2Vec2Model
from PIL import Image
from collections import Counter, defaultdict
import soundfile as sf
import os
import warnings
warnings.filterwarnings("ignore")

DATASET_PATHS = {
    "HateMM": {
        "base": "datasets/HateMM",
        "frame_dir": "datasets/HateMM/frames",
        "audio_dir": "datasets/HateMM/audios",
        "label_map": {"Hate": 1, "Non Hate": 0},
    },
    "MHClip_EN": {
        "base": "datasets/Multihateclip/English",
        "frame_dir": "datasets/Multihateclip/English/frames",
        "audio_dir": "datasets/Multihateclip/English/audios",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    },
    "MHClip_ZH": {
        "base": "datasets/Multihateclip/Chinese",
        "frame_dir": "datasets/Multihateclip/Chinese/frames",
        "audio_dir": "datasets/Multihateclip/Chinese/audios",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    },
}


def load_data(dataset_name):
    cfg = DATASET_PATHS[dataset_name]
    with open(f"{cfg['base']}/generic_data.json") as f:
        data = json.load(f)
    with open(f"{cfg['base']}/splits/train.csv") as f:
        train_ids = [l.strip() for l in f if l.strip()]
    with open(f"{cfg['base']}/splits/test.csv") as f:
        test_ids = [l.strip() for l in f if l.strip()]
    id2sample = {s["Video_ID"]: s for s in data}
    return id2sample, train_ids, test_ids, cfg["label_map"]


def get_text(sample):
    title = sample.get("Title", "") or ""
    transcript = sample.get("Transcript", "") or ""
    text = f"{title} {transcript}".strip()
    return text if text else "[empty]"


def get_mllm_pred(sample):
    resp = sample.get("generic_response", {})
    if isinstance(resp, dict):
        oj = resp.get("overall_judgment", "")
    else:
        oj = str(resp)
    oj_lower = oj.lower()
    if "not hateful" in oj_lower or "not hate" in oj_lower or "normal" in oj_lower[:30]:
        return 0
    elif "yes" in oj_lower[:10] or "hateful" in oj_lower[:30] or "hate" in oj_lower[:20]:
        return 1
    return -1


def extract_frame_features(vid_ids, id2sample, frame_dir, clip_model, clip_processor, device):
    """Extract CLIP features for a list of video IDs."""
    features = {}
    for vid in vid_ids:
        fdir = os.path.join(frame_dir, vid)
        if not os.path.isdir(fdir):
            features[vid] = np.zeros(768, dtype=np.float32)
            continue
        frame_files = sorted([f for f in os.listdir(fdir) if f.endswith(('.jpg', '.png'))])
        if not frame_files:
            features[vid] = np.zeros(768, dtype=np.float32)
            continue
        # Use all 32 frames (or up to 32)
        if len(frame_files) > 32:
            indices = np.linspace(0, len(frame_files) - 1, 32, dtype=int)
            frame_files = [frame_files[j] for j in indices]
        images = []
        for ff in frame_files:
            try:
                img = Image.open(os.path.join(fdir, ff)).convert("RGB")
                images.append(img)
            except Exception:
                continue
        if not images:
            features[vid] = np.zeros(768, dtype=np.float32)
            continue
        with torch.no_grad():
            inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
            embs = clip_model.get_image_features(**inputs)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            features[vid] = embs.mean(dim=0).cpu().numpy()
    return features


def extract_audio_features(vid_ids, id2sample, audio_dir, wav_processor, wav_model, device):
    """Extract Wav2Vec2 features."""
    features = {}
    for vid in vid_ids:
        audio_path = os.path.join(audio_dir, vid + ".wav")
        if not os.path.exists(audio_path):
            # try .mp3
            audio_path = os.path.join(audio_dir, vid + ".mp3")
        if not os.path.exists(audio_path):
            features[vid] = np.zeros(768, dtype=np.float32)
            continue
        try:
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if len(audio) > sr * 30:
                audio = audio[:sr * 30]
            with torch.no_grad():
                inputs = wav_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
                outputs = wav_model(**inputs)
                feat = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            features[vid] = feat
        except Exception:
            features[vid] = np.zeros(768, dtype=np.float32)
    return features


class MultimodalDataset(Dataset):
    def __init__(self, text_embs, frame_feats, audio_feats, labels, vid_ids):
        self.text = torch.tensor(np.array([text_embs[v] for v in vid_ids]), dtype=torch.float32)
        self.frame = torch.tensor(np.array([frame_feats.get(v, np.zeros(768)) for v in vid_ids]), dtype=torch.float32)
        self.audio = torch.tensor(np.array([audio_feats.get(v, np.zeros(768)) for v in vid_ids]), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text[idx], self.frame[idx], self.audio[idx], self.labels[idx]


class MultiHeadRouter(nn.Module):
    """Multi-head routing: each modality has its own head, plus a gating/routing mechanism."""
    def __init__(self, text_dim=384, av_dim=768, hidden=128, num_classes=2, dropout=0.3):
        super().__init__()
        # Per-modality heads
        self.text_head = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.frame_head = nn.Sequential(
            nn.Linear(av_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.audio_head = nn.Sequential(
            nn.Linear(av_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        # Router: learns which modality to trust
        self.router = nn.Sequential(
            nn.Linear(text_dim + av_dim + av_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),  # weights for 3 heads
        )

    def forward(self, text, frame, audio):
        # Per-head logits
        text_logits = self.text_head(text)
        frame_logits = self.frame_head(frame)
        audio_logits = self.audio_head(audio)

        # Router weights
        combined = torch.cat([text, frame, audio], dim=-1)
        weights = torch.softmax(self.router(combined), dim=-1)  # [B, 3]

        # Weighted combination
        stacked = torch.stack([text_logits, frame_logits, audio_logits], dim=1)  # [B, 3, C]
        logits = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, C]
        return logits, weights


def train_multimodal(train_text, train_frame, train_audio, train_labels, train_vids,
                     test_text, test_frame, test_audio, test_labels, test_vids,
                     n_seeds=20, text_dim=384, av_dim=768):
    all_preds = []
    all_weights = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = MultiHeadRouter(text_dim=text_dim, av_dim=av_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        train_ds = MultimodalDataset(train_text, train_frame, train_audio, train_labels, train_vids)
        loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(50):
            for t, f, a, y in loader:
                optimizer.zero_grad()
                logits, _ = model(t, f, a)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

        model.eval()
        test_ds = MultimodalDataset(test_text, test_frame, test_audio, test_labels, test_vids)
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
        with torch.no_grad():
            for t, f, a, y in test_loader:
                logits, weights = model(t, f, a)
                preds = logits.argmax(dim=1).numpy()
                w = weights.numpy()
        all_preds.append(preds)
        all_weights.append(w)

    return np.array(all_preds), np.array(all_weights)


def analyze_errors(gt, small_pred, mllm_pred, small_correct, mllm_correct, vids, id2sample, label_map):
    """Categorize error patterns."""
    inv_label = {v: k for k, v in label_map.items()}

    patterns = {
        "small_only_wrong": [],  # small wrong, MLLM right — MLLM strength
        "mllm_only_wrong": [],   # MLLM wrong, small right — small strength
        "both_wrong": [],        # both wrong — truly hard
        "both_correct": [],      # both correct — easy
    }

    for i, vid in enumerate(vids):
        s = id2sample.get(vid, {})
        info = {
            "vid": vid,
            "gt": int(gt[i]),
            "gt_label": s.get("Label", "?"),
            "small_pred": int(small_pred[i]),
            "mllm_pred": int(mllm_pred[i]),
            "title": (s.get("Title", "") or "")[:80],
            "transcript": (s.get("Transcript", "") or "")[:120],
        }
        resp = s.get("generic_response", {})
        if isinstance(resp, dict):
            info["mllm_judgment"] = resp.get("overall_judgment", "")[:150]
        else:
            info["mllm_judgment"] = ""

        if small_correct[i] and mllm_correct[i]:
            patterns["both_correct"].append(info)
        elif small_correct[i] and not mllm_correct[i]:
            patterns["mllm_only_wrong"].append(info)
        elif not small_correct[i] and mllm_correct[i]:
            patterns["small_only_wrong"].append(info)
        else:
            patterns["both_wrong"].append(info)

    return patterns


def print_error_analysis(patterns, dataset_name):
    """Print structured error pattern analysis."""
    print(f"\n{'='*60}")
    print(f"ERROR PATTERN ANALYSIS: {dataset_name}")
    print(f"{'='*60}")

    # --- Small model weaknesses (MLLM strength) ---
    small_errors = patterns["small_only_wrong"]
    print(f"\n--- Small Model Errors that MLLM Gets Right ({len(small_errors)}) ---")
    print("(= MLLM's strength / cases where MLLM should handle)")

    # Categorize by GT label
    fn_small = [e for e in small_errors if e["gt"] == 1 and e["small_pred"] == 0]
    fp_small = [e for e in small_errors if e["gt"] == 0 and e["small_pred"] == 1]
    print(f"  Small FN (missed hate, MLLM catches): {len(fn_small)}")
    for e in fn_small[:5]:
        print(f"    [{e['vid']}] {e['gt_label']} | transcript: {e['transcript'][:80]}")
    print(f"  Small FP (false alarm, MLLM correct): {len(fp_small)}")
    for e in fp_small[:5]:
        print(f"    [{e['vid']}] {e['gt_label']} | transcript: {e['transcript'][:80]}")

    # --- MLLM weaknesses (Small model strength) ---
    mllm_errors = patterns["mllm_only_wrong"]
    print(f"\n--- MLLM Errors that Small Model Gets Right ({len(mllm_errors)}) ---")
    print("(= Small model's strength / cases where small model is better)")

    fn_mllm = [e for e in mllm_errors if e["gt"] == 1 and e["mllm_pred"] == 0]
    fp_mllm = [e for e in mllm_errors if e["gt"] == 0 and e["mllm_pred"] == 1]
    print(f"  MLLM FN (missed hate, small catches): {len(fn_mllm)}")
    for e in fn_mllm[:5]:
        print(f"    [{e['vid']}] {e['gt_label']} | transcript: {e['transcript'][:80]}")
        print(f"      MLLM said: {e['mllm_judgment'][:100]}")
    print(f"  MLLM FP (false alarm, small correct): {len(fp_mllm)}")
    for e in fp_mllm[:5]:
        print(f"    [{e['vid']}] {e['gt_label']} | transcript: {e['transcript'][:80]}")
        print(f"      MLLM said: {e['mllm_judgment'][:100]}")

    # --- Both wrong ---
    both = patterns["both_wrong"]
    print(f"\n--- Both Wrong ({len(both)}) ---")
    fn_both = [e for e in both if e["gt"] == 1]
    fp_both = [e for e in both if e["gt"] == 0]
    print(f"  Both FN: {len(fn_both)}, Both FP: {len(fp_both)}")
    for e in both[:5]:
        print(f"    [{e['vid']}] GT={e['gt_label']} | {e['transcript'][:80]}")


def run_one_dataset(dataset_name):
    print(f"\n{'#'*60}")
    print(f"# {dataset_name}")
    print(f"{'#'*60}")

    id2sample, train_ids, test_ids, label_map = load_data(dataset_name)
    cfg = DATASET_PATHS[dataset_name]

    # Prepare samples
    train_vids, train_labels, train_texts = [], [], []
    for vid in train_ids:
        if vid not in id2sample:
            continue
        s = id2sample[vid]
        gt = label_map.get(s["Label"], -1)
        if gt == -1:
            continue
        train_vids.append(vid)
        train_labels.append(gt)
        train_texts.append(get_text(s))

    test_vids, test_labels, test_texts, test_mllm_preds = [], [], [], []
    for vid in test_ids:
        if vid not in id2sample:
            continue
        s = id2sample[vid]
        gt = label_map.get(s["Label"], -1)
        if gt == -1:
            continue
        test_vids.append(vid)
        test_labels.append(gt)
        test_texts.append(get_text(s))
        test_mllm_preds.append(get_mllm_pred(s))

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    test_mllm_preds = np.array(test_mllm_preds)

    print(f"Train: {len(train_vids)}, Test: {len(test_vids)}")

    # Encode text
    print("Encoding text...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    train_text_embs = {v: e for v, e in zip(train_vids, encoder.encode(train_texts, show_progress_bar=False))}
    test_text_embs = {v: e for v, e in zip(test_vids, encoder.encode(test_texts, show_progress_bar=False))}

    # Extract frame features
    print("Extracting frame features...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    all_vids = train_vids + test_vids
    frame_feats = extract_frame_features(all_vids, id2sample, cfg["frame_dir"], clip_model, clip_processor, device)
    del clip_model, clip_processor
    torch.cuda.empty_cache()

    # Extract audio features
    print("Extracting audio features...")
    wav_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval().to(device)
    audio_feats = extract_audio_features(all_vids, id2sample, cfg["audio_dir"], wav_processor, wav_model, device)
    del wav_model, wav_processor
    torch.cuda.empty_cache()

    # Train multimodal multi-head router
    print("Training multi-head router (20 seeds)...")
    all_preds, all_weights = train_multimodal(
        train_text_embs, frame_feats, audio_feats, train_labels, train_vids,
        test_text_embs, frame_feats, audio_feats, test_labels, test_vids,
        n_seeds=20,
    )

    # Majority vote
    majority_preds = np.round(all_preds.mean(axis=0)).astype(int)
    avg_weights = all_weights.mean(axis=0)  # [n_test, 3]
    pred_mean = all_preds.mean(axis=0)
    small_confidence = np.abs(pred_mean - 0.5) * 2

    # Filter valid MLLM predictions
    valid_mask = test_mllm_preds != -1
    print(f"MLLM valid: {valid_mask.sum()}/{len(valid_mask)}")

    gt = test_labels[valid_mask]
    small_pred = majority_preds[valid_mask]
    mllm_pred = test_mllm_preds[valid_mask]
    conf = small_confidence[valid_mask]
    vids = np.array(test_vids)[valid_mask]
    weights = avg_weights[valid_mask]

    small_correct = (small_pred == gt)
    mllm_correct = (mllm_pred == gt)

    small_acc = accuracy_score(gt, small_pred)
    small_f1 = f1_score(gt, small_pred, average="macro")
    mllm_acc = accuracy_score(gt, mllm_pred)
    mllm_f1 = f1_score(gt, mllm_pred, average="macro")

    print(f"\n{'='*60}")
    print(f"RESULTS: {dataset_name} (n={len(gt)})")
    print(f"{'='*60}")
    print(f"Small (multimodal multi-head): Acc={small_acc:.3f}, F1={small_f1:.3f}")
    print(f"MLLM zero-shot:               Acc={mllm_acc:.3f}, F1={mllm_f1:.3f}")

    # Router weight analysis
    print(f"\nAvg router weights: text={weights[:, 0].mean():.3f}, frame={weights[:, 1].mean():.3f}, audio={weights[:, 2].mean():.3f}")

    # Complementarity
    both_correct = (small_correct & mllm_correct).sum()
    small_only = (small_correct & ~mllm_correct).sum()
    mllm_only = (~small_correct & mllm_correct).sum()
    both_wrong = (~small_correct & ~mllm_correct).sum()

    print(f"\n--- Complementarity Matrix ---")
    print(f"                    MLLM correct  MLLM wrong")
    print(f"Small correct       {both_correct:>8}       {small_only:>8}")
    print(f"Small wrong         {mllm_only:>8}       {both_wrong:>8}")

    n_small_err = (~small_correct).sum()
    n_mllm_err = (~mllm_correct).sum()
    print(f"\nSmall errors: {n_small_err}")
    print(f"  → MLLM fixes: {mllm_only} ({mllm_only/max(n_small_err,1)*100:.1f}%)")
    print(f"MLLM errors: {n_mllm_err}")
    print(f"  → Small fixes: {small_only} ({small_only/max(n_mllm_err,1)*100:.1f}%)")

    oracle_acc = (small_correct | mllm_correct).sum() / len(gt)
    print(f"\nOracle: Acc={oracle_acc:.3f}")
    print(f"  vs small: +{(oracle_acc - small_acc)*100:.1f}pp")
    print(f"  vs MLLM:  +{(oracle_acc - mllm_acc)*100:.1f}pp")

    # Confidence analysis
    print(f"\n--- Confidence-based Routing ---")
    for threshold in [0.5, 0.7, 0.9]:
        high = conf >= threshold
        low = conf < threshold
        if high.sum() > 0 and low.sum() > 0:
            h_acc = (small_pred[high] == gt[high]).mean()
            l_acc_s = (small_pred[low] == gt[low]).mean()
            l_acc_m = (mllm_pred[low] == gt[low]).mean()
            # Simulated cascade: use small for high-conf, MLLM for low-conf
            cascade_pred = np.where(high, small_pred, mllm_pred)
            cascade_acc = accuracy_score(gt, cascade_pred)
            cascade_f1 = f1_score(gt, cascade_pred, average="macro")
            print(f"  conf>={threshold:.1f}: n={high.sum():>4} small_acc={h_acc:.3f} | conf<{threshold:.1f}: n={low.sum():>4} small={l_acc_s:.3f} mllm={l_acc_m:.3f}")
            print(f"    → Cascade (small if conf>={threshold:.1f}, else MLLM): Acc={cascade_acc:.3f} F1={cascade_f1:.3f} | MLLM calls={low.sum()} ({low.sum()/len(gt)*100:.0f}%)")

    # Error pattern analysis
    patterns = analyze_errors(gt, small_pred, mllm_pred, small_correct, mllm_correct, vids, id2sample, label_map)
    print_error_analysis(patterns, dataset_name)


def main():
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["HateMM", "MHClip_EN", "MHClip_ZH"]
    for ds in datasets:
        run_one_dataset(ds)


if __name__ == "__main__":
    main()
