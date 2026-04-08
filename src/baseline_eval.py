"""Quick baseline evaluation using HVGuard's pre-extracted features."""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import random
import os
import sys

HVGUARD_DIR = "/data/jehc223/EMNLP2/baseline/HVGuard"

DATASETS = {
    "HateMM": {
        "text": f"{HVGUARD_DIR}/embeddings/HateMM/text_features.pth",
        "audio": f"{HVGUARD_DIR}/embeddings/HateMM/audio_features.pth",
        "frame": f"{HVGUARD_DIR}/embeddings/HateMM/frame_features.pth",
        "mllm": f"{HVGUARD_DIR}/embeddings/HateMM/MLLM_rationale_features.pth",
        "label": f"{HVGUARD_DIR}/datasets/HateMM/annotation(new).json",
        "splits_dir": "/data/jehc223/HateMM/splits",
        "label_map": {"Hate": 1, "Non Hate": 0},
    },
    "MultiHateClip_CN": {
        "text": f"{HVGUARD_DIR}/embeddings/Multihateclip/Chinese/text_features.pth",
        "audio": f"{HVGUARD_DIR}/embeddings/Multihateclip/Chinese/audio_features.pth",
        "frame": f"{HVGUARD_DIR}/embeddings/Multihateclip/Chinese/frame_features.pth",
        "mllm": f"{HVGUARD_DIR}/embeddings/Multihateclip/Chinese/MLLM_rationale_features.pth",
        "label": f"{HVGUARD_DIR}/datasets/Multihateclip/Chinese/annotation(new).json",
        "splits_dir": "/data/jehc223/Multihateclip/Chinese/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
    "MultiHateClip_EN": {
        "text": f"{HVGUARD_DIR}/embeddings/Multihateclip/English/text_features.pth",
        "audio": f"{HVGUARD_DIR}/embeddings/Multihateclip/English/audio_features.pth",
        "frame": f"{HVGUARD_DIR}/embeddings/Multihateclip/English/frame_features.pth",
        "mllm": f"{HVGUARD_DIR}/embeddings/Multihateclip/English/MLLM_rationale_features.pth",
        "label": f"{HVGUARD_DIR}/datasets/Multihateclip/English/annotation(new).json",
        "splits_dir": "/data/jehc223/Multihateclip/English/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
}


class FeatureDataset(Dataset):
    def __init__(self, video_ids, features_dict, labels_dict, label_map):
        self.video_ids = video_ids
        self.features = features_dict
        self.labels = labels_dict
        self.label_map = label_map

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        feats = []
        for mod in ["text", "audio", "frame", "mllm"]:
            if vid in self.features[mod]:
                feats.append(self.features[mod][vid])
            else:
                feats.append(torch.zeros(768))
        x = torch.cat(feats, dim=-1)  # 768*4 = 3072
        y = self.label_map[self.labels[vid]]
        return x, torch.tensor(y, dtype=torch.long)


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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        gate_weights = self.gate_dropout(self.gate(x))
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)
        fused = torch.sum(gate_weights.unsqueeze(-1) * expert_outs, dim=1)
        return self.classifier(fused)


def load_dataset(ds_name):
    cfg = DATASETS[ds_name]
    features = {}
    for mod in ["text", "audio", "frame", "mllm"]:
        features[mod] = torch.load(cfg[mod], map_location="cpu", weights_only=True)

    with open(cfg["label"]) as f:
        raw = json.load(f)
    labels = {d["Video_ID"]: d["Label"] for d in raw}

    splits = {}
    for split in ["train", "valid", "test"]:
        fpath = f"{cfg['splits_dir']}/{split}.csv"
        if not os.path.exists(fpath):
            # Try without splits subdir
            fpath = f"{cfg['splits_dir']}/../{split}.csv"
        ids = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.append(line)
        # Filter to IDs that have all features
        valid_ids = [v for v in ids if v in labels and labels[v] in cfg["label_map"]]
        splits[split] = valid_ids

    return features, labels, splits, cfg["label_map"]


def train_and_eval(ds_name, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    features, labels, splits, label_map = load_dataset(ds_name)

    train_ds = FeatureDataset(splits["train"], features, labels, label_map)
    valid_ds = FeatureDataset(splits["valid"], features, labels, label_map)
    test_ds = FeatureDataset(splits["test"], features, labels, label_map)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoEClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(30):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                out = model(x)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_labels.extend(y.numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            test_preds.extend(out.argmax(1).cpu().numpy())
            test_labels.extend(y.numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    print(f"\n{ds_name} (seed={seed}):")
    print(f"  Val acc: {best_val_acc:.4f}")
    print(f"  Test acc: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Test F1 (macro): {test_f1:.4f}")
    print(classification_report(test_labels, test_preds, target_names=["Normal", "Hateful"]))
    return test_acc


if __name__ == "__main__":
    for ds in ["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"]:
        # Run with multiple seeds
        accs = []
        for seed in [42, 123, 456]:
            acc = train_and_eval(ds, seed=seed)
            accs.append(acc)
        print(f"\n{ds} avg acc: {np.mean(accs)*100:.1f}% (std: {np.std(accs)*100:.1f}%)")
        print("=" * 60)
