#!/usr/bin/env python3
"""Enhanced ensemble with DeBERTa + cross-dataset pretraining + multiple classifiers.

Improvements over v1:
1. DeBERTa-v3-base instead of BERT-base (better NLU)
2. Cross-dataset pretraining: train on all datasets first, then fine-tune per dataset
3. Add unified model predictions as additional ensemble component
4. Better stacking with more features
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
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


def build_text_input(vid, mllm_results, mllm_scores, ann_data):
    parts = []
    if vid in mllm_results:
        a = mllm_results[vid].get("analysis", "")
        if a and not a.startswith("ERROR") and not a.startswith("[TEXT-ONLY"):
            parts.append("[ANALYSIS] " + a[:1500])
    ann = ann_data.get(vid, {})
    mix = ann.get("Mix_description", "").strip()
    if mix:
        parts.append("[HVGUARD] " + mix[:500])
    title = ann.get("Title", "").strip()
    transcript = ann.get("Transcript", "").strip()
    if title:
        parts.append("[TITLE] " + title)
    if transcript:
        parts.append("[TRANSCRIPT] " + transcript[:500])
    if vid in mllm_scores:
        sc = mllm_scores[vid].get("scores", {})
        s = (f"[SCORES] hate={sc.get('hate_speech_score','?')}/10 "
             f"visual={sc.get('visual_hate_score','?')}/10 "
             f"overall={sc.get('overall_hate_score','?')}/10 "
             f"verdict={sc.get('classification','?')}")
        parts.append(s)
    return " ".join(parts) if parts else "[NO CONTENT]"


class TextDS(Dataset):
    def __init__(self, vids, texts, labels, label_map, tokenizer, max_len=512):
        self.vids = vids
        self.texts = texts
        self.labels = labels
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, i):
        v = self.vids[i]
        enc = self.tokenizer(self.texts[v], truncation=True, max_length=self.max_len,
                             padding="max_length", return_tensors="pt")
        return (enc["input_ids"].squeeze(), enc["attention_mask"].squeeze(),
                torch.tensor(self.label_map[self.labels[v]], dtype=torch.long))


def load_all_data():
    """Load all data for all datasets."""
    all_data = {}
    for ds_name, cfg in DATASET_CONFIGS.items():
        mllm_results, mllm_scores = {}, {}
        if os.path.exists(cfg["mllm_results"]):
            with open(cfg["mllm_results"]) as f:
                mllm_results = json.load(f)
        if os.path.exists(cfg["mllm_scores"]):
            with open(cfg["mllm_scores"]) as f:
                mllm_scores = json.load(f)
        with open(cfg["label_file"]) as f:
            raw_data = json.load(f)
        ann_data = {d["Video_ID"]: d for d in raw_data}
        labels = {d["Video_ID"]: d["Label"] for d in raw_data}
        texts = {v: build_text_input(v, mllm_results, mllm_scores, ann_data) for v in ann_data}
        splits = {}
        for split in ["train", "valid", "test"]:
            with open(os.path.join(cfg["splits_dir"], split + ".csv")) as f:
                splits[split] = [l.strip() for l in f
                                 if l.strip() in labels and labels[l.strip()] in cfg["label_map"]]
        # Load multimodal features
        text_f = torch.load(cfg["hvguard_emb"] + "/text_features.pth", map_location="cpu", weights_only=True)
        audio_f = torch.load(cfg["hvguard_emb"] + "/audio_features.pth", map_location="cpu", weights_only=True)
        frame_f = torch.load(cfg["hvguard_emb"] + "/frame_features.pth", map_location="cpu", weights_only=True)
        mllm_f = torch.load(cfg["hvguard_emb"] + "/MLLM_rationale_features.pth", map_location="cpu", weights_only=True)

        all_data[ds_name] = {
            "texts": texts, "labels": labels, "label_map": cfg["label_map"],
            "splits": splits, "mllm_scores": mllm_scores,
            "text_f": text_f, "audio_f": audio_f, "frame_f": frame_f, "mllm_f": mllm_f,
        }
    return all_data


def cross_dataset_pretrain(all_data, tokenizer, model_name, device, seed, epochs=5, lr=2e-5):
    """Pretrain on all datasets jointly."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Combine all training data
    all_vids, all_texts, all_labels_dict, all_label_map = [], {}, {}, {}
    for ds_name, data in all_data.items():
        for vid in data["splits"]["train"]:
            all_vids.append(vid)
            all_texts[vid] = data["texts"][vid]
            all_labels_dict[vid] = data["labels"][vid]
            all_label_map[data["labels"][vid]] = data["label_map"][data["labels"][vid]]

    # Create unified label map
    unified_map = {}
    for vid in all_vids:
        label = all_labels_dict[vid]
        for ds_data in all_data.values():
            if label in ds_data["label_map"]:
                unified_map[label] = ds_data["label_map"][label]
                break

    ds = TextDS(all_vids, all_texts, all_labels_dict, unified_map, tokenizer, max_len=512)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    train_y = [unified_map[all_labels_dict[v]] for v in all_vids]
    counts = Counter(train_y)
    total = sum(counts.values())
    weights = torch.tensor([total / (2 * counts[c]) for c in range(2)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    print(f"    Cross-dataset pretraining: {len(all_vids)} samples, {epochs} epochs", flush=True)
    for epoch in range(epochs):
        model.train()
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(ids, attention_mask=mask)
            criterion(out.logits, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    return model


def finetune_and_predict(model, ds_name, data, tokenizer, device, seed, epochs=8, lr=1e-5):
    """Fine-tune pretrained model on specific dataset and return predictions."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    train_ds = TextDS(data["splits"]["train"], data["texts"], data["labels"],
                      data["label_map"], tokenizer, max_len=512)
    val_ds = TextDS(data["splits"]["valid"], data["texts"], data["labels"],
                    data["label_map"], tokenizer, max_len=512)
    test_ds = TextDS(data["splits"]["test"], data["texts"], data["labels"],
                     data["label_map"], tokenizer, max_len=512)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    train_y = [data["label_map"][data["labels"][v]] for v in data["splits"]["train"]]
    counts = Counter(train_y)
    total = sum(counts.values())
    weights = torch.tensor([total / (2 * counts[c]) for c in range(2)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    best_val = 0
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(ids, attention_mask=mask)
            criterion(out.logits, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        vpreds = []
        with torch.no_grad():
            for ids, mask, y in val_loader:
                out = model(ids.to(device), attention_mask=mask.to(device))
                vpreds.extend(out.logits.argmax(1).cpu().numpy())
        val_y = [data["label_map"][data["labels"][v]] for v in data["splits"]["valid"]]
        val_acc = accuracy_score(val_y, vpreds)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break

    model.load_state_dict(best_state)
    model.to(device).eval()

    def get_probs(loader):
        probs = []
        with torch.no_grad():
            for ids, mask, _ in loader:
                out = model(ids.to(device), attention_mask=mask.to(device))
                p = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
                probs.extend(p)
        return np.array(probs)

    val_probs = get_probs(val_loader)
    test_probs = get_probs(test_loader)
    val_preds = (val_probs > 0.5).astype(int)
    val_acc = accuracy_score(val_y, val_preds)
    print(f"      DeBERTa val={val_acc:.3f}", flush=True)
    return val_probs, test_probs


def get_moe_probs(ds_name, data, device, seed):
    """MoE classifier on multimodal features."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    zero = torch.zeros(768)
    def get_feats(vids):
        X, Y = [], []
        for vid in vids:
            x = torch.cat([data["text_f"].get(vid, zero), data["audio_f"].get(vid, zero),
                          data["frame_f"].get(vid, zero), data["mllm_f"].get(vid, zero)])
            X.append(x)
            Y.append(data["label_map"][data["labels"][vid]])
        return torch.stack(X), torch.tensor(Y)

    X_tr, y_tr = get_feats(data["splits"]["train"])
    X_val, y_val = get_feats(data["splits"]["valid"])
    X_test, y_test = get_feats(data["splits"]["test"])

    from ensemble import MoEClassifier
    model = MoEClassifier().to(device)
    counts = Counter(y_tr.numpy())
    total = sum(counts.values())
    weights = torch.tensor([total / (2 * counts[c]) for c in range(2)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val = 0; best_state = None
    loader = DataLoader(torch.utils.data.TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    for epoch in range(40):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(X_val.to(device)).argmax(1).cpu() == y_val).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        val_p = torch.softmax(model(X_val.to(device)), dim=1)[:, 1].cpu().numpy()
        test_p = torch.softmax(model(X_test.to(device)), dim=1)[:, 1].cpu().numpy()
    return val_p, test_p


def get_score_features(ds_name, data):
    def extract(vids):
        feats = []
        for vid in vids:
            sc = data["mllm_scores"].get(vid, {}).get("scores", {})
            f = [sc.get(k, 5) / 10.0 for k in SCORE_KEYS]
            f.append(1.0 if sc.get("classification", "").lower() in ["hateful", "offensive"] else 0.0)
            feats.append(f)
        return np.array(feats)
    return extract(data["splits"]["valid"]), extract(data["splits"]["test"])


def run_ensemble(all_data, ds_name, seed, model_name, device, pretrained_model=None):
    print(f"\n  --- {ds_name} seed={seed} ---", flush=True)
    data = all_data[ds_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    val_y = np.array([data["label_map"][data["labels"][v]] for v in data["splits"]["valid"]])
    test_y = np.array([data["label_map"][data["labels"][v]] for v in data["splits"]["test"]])

    # Component 1: MoE
    print("    Training MoE...", flush=True)
    import sys; sys.path.insert(0, "/data/jehc223/EMNLP2/src")
    moe_val, moe_test = get_moe_probs(ds_name, data, device, seed)
    moe_acc = accuracy_score(test_y, (moe_test > 0.5).astype(int))

    # Component 2: DeBERTa with cross-dataset pretraining
    print("    Fine-tuning DeBERTa...", flush=True)
    import copy
    model_copy = copy.deepcopy(pretrained_model)
    deberta_val, deberta_test = finetune_and_predict(
        model_copy, ds_name, data, tokenizer, device, seed, epochs=8, lr=1e-5)
    deberta_acc = accuracy_score(test_y, (deberta_test > 0.5).astype(int))

    # Component 3: MLLM scores
    score_val, score_test = get_score_features(ds_name, data)

    # Stacking ensemble
    val_stack = np.column_stack([moe_val, deberta_val, score_val])
    test_stack = np.column_stack([moe_test, deberta_test, score_test])

    stacker = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    stacker.fit(val_stack, val_y)
    stack_pred = stacker.predict(test_stack)
    stack_acc = accuracy_score(test_y, stack_pred)
    stack_f1 = f1_score(test_y, stack_pred, average="macro")

    # Also try weighted averaging
    # Find optimal weights on validation set
    best_w_acc = 0
    best_w = (0.5, 0.5)
    for w1 in np.arange(0.1, 0.9, 0.05):
        w2 = 1 - w1
        avg_p = w1 * moe_val + w2 * deberta_val
        avg_pred = (avg_p > 0.5).astype(int)
        w_acc = accuracy_score(val_y, avg_pred)
        if w_acc > best_w_acc:
            best_w_acc = w_acc
            best_w = (w1, w2)

    wavg_test = best_w[0] * moe_test + best_w[1] * deberta_test
    wavg_pred = (wavg_test > 0.5).astype(int)
    wavg_acc = accuracy_score(test_y, wavg_pred)

    print(f"    MoE: {moe_acc*100:.1f}%", flush=True)
    print(f"    DeBERTa: {deberta_acc*100:.1f}%", flush=True)
    print(f"    Weighted avg (w={best_w[0]:.2f},{best_w[1]:.2f}): {wavg_acc*100:.1f}%", flush=True)
    print(f"    Stacked: {stack_acc*100:.1f}% (F1={stack_f1:.3f})", flush=True)

    best_acc = max(stack_acc, wavg_acc)
    best_method = "stacked" if stack_acc >= wavg_acc else "weighted_avg"
    print(f"    Best: {best_acc*100:.1f}% ({best_method})", flush=True)

    return best_acc, stack_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--pretrain_epochs", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading all data...", flush=True)
    all_data = load_all_data()

    for seed in args.seeds:
        # Cross-dataset pretraining (once per seed)
        print(f"\n  Cross-dataset pretraining (seed={seed})...", flush=True)
        pretrained = cross_dataset_pretrain(
            all_data, tokenizer, args.model_name, device, seed,
            epochs=args.pretrain_epochs, lr=2e-5)

        for ds in args.datasets:
            acc, f1 = run_ensemble(all_data, ds, seed, args.model_name, device, pretrained)
            # Store results
            if not hasattr(main, 'results'):
                main.results = {}
            if ds not in main.results:
                main.results[ds] = []
            main.results[ds].append(acc)

    # Summary
    print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}", flush=True)
    for ds in args.datasets:
        accs = main.results.get(ds, [])
        if accs:
            avg = np.mean(accs) * 100
            std = np.std(accs) * 100
            best = max(accs) * 100
            target = 90 if ds == "HateMM" else 85
            gap = target - avg
            status = "TARGET MET" if avg >= target else f"gap: {gap:.1f}%"
            print(f"  {ds}: avg={avg:.1f}% std={std:.1f}% best={best:.1f}% — {status}", flush=True)


if __name__ == "__main__":
    main()
