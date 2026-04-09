#!/usr/bin/env python3
"""Stacking ensemble that combines multiple classifiers.

For each video, generates predictions from:
1. Multimodal MoE classifier (HVGuard features)
2. Fine-tuned BERT on MLLM analysis text
3. Unified BERT + multimodal model
4. MLLM structured scores

Then learns optimal combination weights on the validation set.
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression


HVGUARD_DIR = "/data/jehc223/EMNLP2/baseline/HVGuard"

DATASET_CONFIGS = {
    "HateMM": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/HateMM",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/HateMM/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/HateMM/mllm_scores.json",
        "label_file": f"{HVGUARD_DIR}/datasets/HateMM/data (base).json",
        "splits_dir": "/data/jehc223/HateMM/splits",
        "label_map": {"Hate": 1, "Non Hate": 0},
    },
    "MultiHateClip_CN": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/Multihateclip/Chinese",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_CN/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_CN/mllm_scores.json",
        "label_file": f"{HVGUARD_DIR}/datasets/Multihateclip/Chinese/data.json",
        "splits_dir": "/data/jehc223/Multihateclip/Chinese/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
    "MultiHateClip_EN": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/Multihateclip/English",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_EN/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_EN/mllm_scores.json",
        "label_file": f"{HVGUARD_DIR}/datasets/Multihateclip/English/data.json",
        "splits_dir": "/data/jehc223/Multihateclip/English/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
}

SCORE_KEYS = ["hate_speech_score", "visual_hate_score", "cross_modal_score",
              "implicit_hate_score", "overall_hate_score", "confidence"]


# ─── Component 1: MoE Classifier ─────────────────────────────

class MoEClassifier(nn.Module):
    def __init__(self, input_dim=3072, num_experts=8, expert_dim=256, hidden_dim=128, num_classes=2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, expert_dim), nn.ReLU(), nn.Linear(expert_dim, hidden_dim))
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(nn.Linear(input_dim, num_experts), nn.Softmax(dim=-1))
        self.gate_dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        gate_w = self.gate_dropout(self.gate(x))
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        fused = torch.sum(gate_w.unsqueeze(-1) * expert_out, dim=1)
        return self.classifier(fused)


def get_moe_probs(ds_name, splits, device, seed):
    """Train MoE and return probabilities for val/test."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg = DATASET_CONFIGS[ds_name]

    text = torch.load(cfg["hvguard_emb"] + "/text_features.pth", map_location="cpu", weights_only=True)
    audio = torch.load(cfg["hvguard_emb"] + "/audio_features.pth", map_location="cpu", weights_only=True)
    frame = torch.load(cfg["hvguard_emb"] + "/frame_features.pth", map_location="cpu", weights_only=True)
    mllm = torch.load(cfg["hvguard_emb"] + "/MLLM_rationale_features.pth", map_location="cpu", weights_only=True)

    with open(cfg["label_file"]) as f:
        labels = {d["Video_ID"]: d["Label"] for d in json.load(f)}

    zero = torch.zeros(768)
    def get_feats(vids):
        X, Y = [], []
        for vid in vids:
            x = torch.cat([text.get(vid, zero), audio.get(vid, zero),
                          frame.get(vid, zero), mllm.get(vid, zero)])
            X.append(x)
            Y.append(cfg["label_map"][labels[vid]])
        return torch.stack(X), torch.tensor(Y)

    X_train, y_train = get_feats(splits["train"])
    X_val, y_val = get_feats(splits["valid"])
    X_test, y_test = get_feats(splits["test"])

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = MoEClassifier().to(device)
    counts = Counter(y_train.numpy())
    total = sum(counts.values())
    weights = torch.tensor([total / (2 * counts[c]) for c in range(2)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val = 0
    best_state = None
    for epoch in range(30):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            val_acc = (val_out.argmax(1).cpu() == y_val).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        test_probs = torch.softmax(model(X_test.to(device)), dim=1)[:, 1].cpu().numpy()
    return val_probs, test_probs


# ─── Component 2: Text Classifier ────────────────────────────

def build_text_input(vid, mllm_results, mllm_scores, ann_data):
    parts = []
    if vid in mllm_results:
        a = mllm_results[vid].get("analysis", "")
        if a and not a.startswith("ERROR") and not a.startswith("[TEXT-ONLY"):
            parts.append("[ANALYSIS] " + a[:1500])
    ann = ann_data.get(vid, {})
    mix = ann.get("Mix_description", "").strip()
    if mix: parts.append("[HVGUARD] " + mix[:500])
    title = ann.get("Title", "").strip()
    transcript = ann.get("Transcript", "").strip()
    if title: parts.append("[TITLE] " + title)
    if transcript: parts.append("[TRANSCRIPT] " + transcript[:500])
    return " ".join(parts) if parts else "[NO CONTENT]"


def get_bert_probs(ds_name, splits, device, seed, texts, labels, label_map, model_name="bert-base-uncased"):
    """Train BERT text classifier and return probabilities."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Create datasets
    class TDS(Dataset):
        def __init__(self, vids):
            self.vids = vids
        def __len__(self): return len(self.vids)
        def __getitem__(self, i):
            v = self.vids[i]
            enc = tokenizer(texts[v], truncation=True, max_length=512, padding="max_length", return_tensors="pt")
            return enc["input_ids"].squeeze(), enc["attention_mask"].squeeze(), torch.tensor(label_map[labels[v]])

    train_loader = DataLoader(TDS(splits["train"]), batch_size=16, shuffle=True)
    val_loader = DataLoader(TDS(splits["valid"]), batch_size=32, shuffle=False)
    test_loader = DataLoader(TDS(splits["test"]), batch_size=32, shuffle=False)

    train_labels = [label_map[labels[v]] for v in splits["train"]]
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = torch.tensor([total / (2 * counts[c]) for c in range(2)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 10
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    best_val = 0; best_state = None; patience = 0
    for epoch in range(10):
        model.train()
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(ids, attention_mask=mask)
            criterion(out.logits, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

        model.eval()
        vpreds, vlabels = [], []
        with torch.no_grad():
            for ids, mask, y in val_loader:
                out = model(ids.to(device), attention_mask=mask.to(device))
                vpreds.extend(out.logits.argmax(1).cpu().numpy())
                vlabels.extend(y.numpy())
        val_acc = accuracy_score(vlabels, vpreds)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3: break

    model.load_state_dict(best_state); model.to(device).eval()

    def get_probs(loader):
        probs = []
        with torch.no_grad():
            for ids, mask, _ in loader:
                out = model(ids.to(device), attention_mask=mask.to(device))
                p = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
                probs.extend(p)
        return np.array(probs)

    return get_probs(val_loader), get_probs(test_loader)


# ─── Component 3: MLLM Score Features ────────────────────────

def get_score_features(ds_name, splits):
    """Extract MLLM score features for val/test."""
    cfg = DATASET_CONFIGS[ds_name]
    scores = {}
    if os.path.exists(cfg["mllm_scores"]):
        with open(cfg["mllm_scores"]) as f:
            scores = json.load(f)

    def extract(vids):
        feats = []
        for vid in vids:
            sc = scores.get(vid, {}).get("scores", {})
            f = [sc.get(k, 5) / 10.0 for k in SCORE_KEYS]
            f.append(1 if sc.get("classification", "").lower() in ["hateful", "offensive"] else 0)
            feats.append(f)
        return np.array(feats)

    return extract(splits["valid"]), extract(splits["test"])


# ─── Main Ensemble ────────────────────────────────────────────

def run_ensemble(ds_name, seed, args):
    print(f"\n  --- {ds_name} seed={seed} ---", flush=True)
    cfg = DATASET_CONFIGS[ds_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(cfg["label_file"]) as f:
        raw_data = json.load(f)
    ann_data = {d["Video_ID"]: d for d in raw_data}
    labels = {d["Video_ID"]: d["Label"] for d in raw_data}

    splits = {}
    for split in ["train", "valid", "test"]:
        with open(os.path.join(cfg["splits_dir"], split + ".csv")) as f:
            splits[split] = [l.strip() for l in f if l.strip() in labels and labels[l.strip()] in cfg["label_map"]]

    mllm_results = {}
    if os.path.exists(cfg["mllm_results"]):
        with open(cfg["mllm_results"]) as f:
            mllm_results = json.load(f)
    mllm_scores = {}
    if os.path.exists(cfg["mllm_scores"]):
        with open(cfg["mllm_scores"]) as f:
            mllm_scores = json.load(f)

    texts = {vid: build_text_input(vid, mllm_results, mllm_scores, ann_data) for vid in ann_data}

    # Get val/test labels
    val_y = np.array([cfg["label_map"][labels[v]] for v in splits["valid"]])
    test_y = np.array([cfg["label_map"][labels[v]] for v in splits["test"]])

    # Component 1: MoE multimodal classifier
    print("    Training MoE...", flush=True)
    moe_val_p, moe_test_p = get_moe_probs(ds_name, splits, device, seed)

    # Component 2: BERT text classifier
    print("    Training BERT...", flush=True)
    bert_val_p, bert_test_p = get_bert_probs(
        ds_name, splits, device, seed, texts, labels, cfg["label_map"])

    # Component 3: MLLM scores
    score_val_f, score_test_f = get_score_features(ds_name, splits)

    # Stack features: [moe_prob, bert_prob, score_features(7)]
    val_stack = np.column_stack([moe_val_p, bert_val_p, score_val_f])
    test_stack = np.column_stack([moe_test_p, bert_test_p, score_test_f])

    # Learn stacking weights with logistic regression
    stacker = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    stacker.fit(val_stack, val_y)

    test_pred = stacker.predict(test_stack)
    test_acc = accuracy_score(test_y, test_pred)
    test_f1 = f1_score(test_y, test_pred, average="macro")

    # Also try simple averaging
    avg_prob = (moe_test_p + bert_test_p) / 2
    avg_pred = (avg_prob > 0.5).astype(int)
    avg_acc = accuracy_score(test_y, avg_pred)

    print(f"    MoE only: {accuracy_score(test_y, (moe_test_p > 0.5).astype(int))*100:.1f}%", flush=True)
    print(f"    BERT only: {accuracy_score(test_y, (bert_test_p > 0.5).astype(int))*100:.1f}%", flush=True)
    print(f"    Simple avg: {avg_acc*100:.1f}%", flush=True)
    print(f"    Stacked: {test_acc*100:.1f}% (F1={test_f1:.3f})", flush=True)
    print(classification_report(test_y, test_pred, target_names=["Normal", "Hateful"]), flush=True)

    return test_acc, test_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    for ds in args.datasets:
        print(f"\n{'='*60}\n{ds}\n{'='*60}", flush=True)
        accs, f1s = [], []
        for seed in args.seeds:
            acc, f1 = run_ensemble(ds, seed, args)
            accs.append(acc)
            f1s.append(f1)

        avg = np.mean(accs) * 100
        std = np.std(accs) * 100
        best = max(accs) * 100
        target = 90 if ds == "HateMM" else 85
        gap = target - avg
        status = "TARGET MET" if avg >= target else f"gap: {gap:.1f}%"
        print(f"\n  {ds} ENSEMBLE: avg={avg:.1f}% std={std:.1f}% best={best:.1f}% — {status}", flush=True)


if __name__ == "__main__":
    main()
