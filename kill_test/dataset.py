"""
Dataset and data loading for kill test.

Loads:
- unit_features.pth: {video_id: [K, 768]} per-unit BERT embeddings
- generic_rationale_features.pth: {video_id: [768]} whole rationale embedding
- frame_features.pth: {video_id: [768]}
- wavlm_audio_features.pth: {video_id: [768]}
- annotation(new).json: labels
- splits/{train,valid,test}.csv: split IDs
"""
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

LABEL_MAP_BINARY = {"Non Hate": 0, "Hate": 1}
LABEL_MAP_TERNARY = {"Normal": 0, "Offensive": 1, "Hateful": 2}
LABEL_MAP_TERNARY_AS_BINARY = {"Normal": 0, "Offensive": 1, "Hateful": 1}
K = 5  # number of evidence units


def load_split_ids(split_dir: str) -> dict:
    """Load train/valid/test video IDs from CSV files."""
    splits = {}
    for name in ["train", "valid", "test"]:
        df = pd.read_csv(Path(split_dir) / f"{name}.csv", header=None)
        splits[name] = df.iloc[:, 0].tolist()
    return splits


def load_features(emb_dir: str) -> dict:
    """Load all feature dicts."""
    emb = Path(emb_dir)
    return {
        "text": torch.load(emb / "generic_rationale_features.pth", map_location="cpu"),
        "units": torch.load(emb / "unit_features.pth", map_location="cpu"),
        "audio": torch.load(emb / "wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(emb / "frame_features.pth", map_location="cpu"),
    }


def load_labels(ann_path: str, force_binary: bool = False) -> dict:
    """Load video_id → int label. Auto-detects binary vs ternary.
    If force_binary=True, merge Offensive+Hateful→1 for ternary datasets."""
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_labels = set(d["Label"] for d in data)
    if all_labels <= {"Non Hate", "Hate"}:
        label_map = LABEL_MAP_BINARY
    elif all_labels <= {"Normal", "Offensive", "Hateful"}:
        label_map = LABEL_MAP_TERNARY_AS_BINARY if force_binary else LABEL_MAP_TERNARY
    else:
        raise ValueError(f"Unknown label set: {all_labels}")
    return {d["Video_ID"]: label_map[d["Label"]] for d in data}


class KillTestDataset(Dataset):
    def __init__(self, video_ids, features, labels):
        # Filter to IDs that exist in all feature dicts
        valid = set(video_ids)
        for feat_dict in features.values():
            valid &= set(feat_dict.keys())
        valid &= set(labels.keys())
        self.ids = [vid for vid in video_ids if vid in valid]
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        vid = self.ids[idx]
        units = self.features["units"][vid]  # [K, 768]
        # Create mask (all 1s since we have exactly K=5 fields for every video)
        mask = torch.ones(units.shape[0])

        return {
            "video_id": vid,
            "text": self.features["text"][vid],  # [768]
            "units": units,  # [K, 768]
            "unit_mask": mask,  # [K]
            "audio": self.features["audio"][vid],  # [768]
            "frame": self.features["frame"][vid],  # [768]
            "label": self.labels[vid],
        }


def collate_fn(batch):
    """Collate into tensors."""
    return {
        "video_id": [b["video_id"] for b in batch],
        "text": torch.stack([b["text"] for b in batch]),
        "units": torch.stack([b["units"] for b in batch]),
        "unit_mask": torch.stack([b["unit_mask"] for b in batch]),
        "audio": torch.stack([b["audio"] for b in batch]),
        "frame": torch.stack([b["frame"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def get_dataloaders(
    emb_dir: str,
    ann_path: str,
    split_dir: str,
    batch_size: int = 32,
) -> tuple:
    """Return train, valid, test DataLoaders + features dict."""
    features = load_features(emb_dir)
    labels = load_labels(ann_path)
    splits = load_split_ids(split_dir)

    train_ds = KillTestDataset(splits["train"], features, labels)
    valid_ds = KillTestDataset(splits["valid"], features, labels)
    test_ds = KillTestDataset(splits["test"], features, labels)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_dl, valid_dl, test_dl
