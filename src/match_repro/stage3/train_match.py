"""MATCH stage 3 training + evaluation driver.

Ports upstream `external_repos/match_hvd/src/main.py:45-303`
(Trainer class + _train / _valid loops) with these simplifications
per the brief:

- **No Hydra**: argparse instead, hardcoded defaults from
  `src/config/HateMM_MATCH.yaml` — BERT-base text encoder,
  `fea_dim=128`, `batch_size=128`, `num_epoch=50`, AdamW `lr=5e-4`,
  `weight_decay=5e-5`, patience 8 early stopping, seed 2025
- **No wandb**: plain logging only
- **No t-SNE save**: the model still emits `tsne_tensor` for
  interface compatibility but the trainer never persists it
- **Single fold type** (`default` only)
- **Loss**: `F.cross_entropy(pred, labels)` — upstream
  `main.py:237`, non-HVD branch
- **Optimizer**: `torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-5)`
- **Scheduler**: none (upstream's `DummyLR` is a no-op)
- **Early stopping**: on valid accuracy, patience 8 — upstream
  `main.py:129-131` + `_valid(split="valid", use_earlystop=True)`
- **Output**: best model loaded after training, run on test split,
  emit `{video_id, pred, score}` jsonl at
  `results/match_qwen2vl_7b/<dataset>/test_match.jsonl`
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import dataset_loaders  # noqa: E402
import match_model  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]

# Defaults from upstream `src/config/HateMM_MATCH.yaml`.
DEFAULT_TEXT_ENCODER = "bert-base-uncased"
DEFAULT_FEA_DIM = 128
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_EPOCH = 50
DEFAULT_LR = 5e-4
DEFAULT_WEIGHT_DECAY = 5e-5
DEFAULT_PATIENCE = 8
DEFAULT_SEED = 2025


def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Upstream `EarlyStopping` in `core_utils.py`, ported verbatim:
    monitors a scalar (valid accuracy) for improvement, saves the
    best model state, flips `early_stop=True` after `patience`
    non-improving epochs.
    """

    def __init__(self, patience: int, path: str):
        import torch

        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self._torch = torch

    def __call__(self, score: float, model):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            self._torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def build_loaders(dataset: str, text_encoder: str, batch_size: int, seed: int):
    """Build train / valid / test dataloaders + collator for one dataset."""
    import torch
    from torch.utils.data import DataLoader

    collator = dataset_loaders.MATCH_Collator(text_encoder=text_encoder)
    generator = torch.Generator().manual_seed(seed)
    train_ds = dataset_loaders.get_dataset(dataset, split="train")
    valid_ds = dataset_loaders.get_dataset(dataset, split="valid")
    test_ds = dataset_loaders.get_dataset(dataset, split="test")

    num_workers = min(8, batch_size // 2)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        shuffle=True,
        generator=generator,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        shuffle=False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        shuffle=False,
    )
    return train_dl, valid_dl, test_dl


def move_batch_to(batch: Dict, device: str) -> Dict:
    """Move all tensors + BatchEncoding contents to `device`. Leaves
    the `vids` list untouched (it's popped by the training loop).
    """
    import torch
    out = {}
    for k, v in batch.items():
        if k == "vids":
            out[k] = v
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif hasattr(v, "to"):
            # BatchEncoding from the tokenizer
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Minimal accuracy + macro-F1 calc without extra deps."""
    if not y_true:
        return {"acc": 0.0, "macro_f1": 0.0}
    n = len(y_true)
    correct = sum(int(p == t) for p, t in zip(y_pred, y_true))
    acc = correct / n
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for lbl in labels:
        tp = sum(1 for p, t in zip(y_pred, y_true) if p == lbl and t == lbl)
        fp = sum(1 for p, t in zip(y_pred, y_true) if p == lbl and t != lbl)
        fn = sum(1 for p, t in zip(y_pred, y_true) if p != lbl and t == lbl)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"acc": acc, "macro_f1": macro_f1}


def train_one_epoch(model, loader, optimizer, device):
    import torch
    import torch.nn.functional as F

    model.train()
    losses = []
    for batch in loader:
        batch = move_batch_to(batch, device)
        _ = batch.pop("vids")
        labels = batch.pop("labels")
        output = model(**batch)
        pred = output["pred"]
        loss = F.cross_entropy(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


def evaluate(model, loader, device, collect_per_video: bool = False):
    """Run the model on a loader. Returns `(metrics, [records])` where
    records is only populated if `collect_per_video=True` (test-set
    prediction export).
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    y_true, y_pred = [], []
    records: List[Dict] = []
    with torch.no_grad():
        for batch in loader:
            vids = batch.get("vids", [])
            batch_on_device = move_batch_to(batch, device)
            _ = batch_on_device.pop("vids")
            labels = batch_on_device.pop("labels")
            output = model(**batch_on_device)
            pred = output["pred"]
            probs = F.softmax(pred, dim=-1)
            pred_label = torch.argmax(pred, dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred_label.cpu().tolist())
            if collect_per_video:
                for i, vid in enumerate(vids):
                    records.append(
                        {
                            "video_id": vid,
                            "pred": int(pred_label[i].item()),
                            "score": float(probs[i, 1].item()),
                            "label": int(labels[i].item()),
                        }
                    )
    metrics = compute_metrics(y_true, y_pred)
    return metrics, records


def run_one_dataset(dataset: str, args):
    import torch

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(
        PROJECT_ROOT, "results", "match_qwen2vl_7b", dataset
    )
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "stage3")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    test_out_path = os.path.join(out_dir, "test_match.jsonl")
    log_path = os.path.join(ckpt_dir, "train.log")

    logging.info(
        f"[{dataset}] text_encoder={args.text_encoder} fea_dim={args.fea_dim} "
        f"bs={args.batch_size} epochs={args.num_epoch} lr={args.lr} "
        f"wd={args.weight_decay} patience={args.patience} seed={args.seed}"
    )

    train_dl, valid_dl, test_dl = build_loaders(
        dataset, args.text_encoder, args.batch_size, args.seed
    )
    logging.info(
        f"[{dataset}] train={len(train_dl.dataset)} valid={len(valid_dl.dataset)} "
        f"test={len(test_dl.dataset)}"
    )

    model = match_model.MATCH(
        text_encoder=args.text_encoder,
        fea_dim=args.fea_dim,
        num_classes=2,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    early = EarlyStopping(patience=args.patience, path=best_path)

    for epoch in range(args.num_epoch):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, device)
        valid_metrics, _ = evaluate(model, valid_dl, device)
        logging.info(
            f"[{dataset}] epoch {epoch}: train_loss={train_loss:.4f}  "
            f"valid_acc={valid_metrics['acc']:.4f}  "
            f"valid_mf1={valid_metrics['macro_f1']:.4f}  "
            f"epoch_time={time.time()-t0:.1f}s"
        )
        early(valid_metrics["acc"], model)
        if early.early_stop:
            logging.info(f"[{dataset}] early stopping at epoch {epoch}")
            break

    # Reload best model state and run on test split.
    logging.info(f"[{dataset}] reloading best model from {best_path}")
    model.load_state_dict(torch.load(best_path, weights_only=False))
    test_metrics, records = evaluate(
        model, test_dl, device, collect_per_video=True
    )
    logging.info(
        f"[{dataset}] TEST  acc={test_metrics['acc']:.4f}  "
        f"macro_f1={test_metrics['macro_f1']:.4f}  n={len(records)}"
    )

    # Emit per-video jsonl (schema per brief: `{video_id, pred, score}`
    # plus `label` for eval_generative_predictions compatibility).
    tmp = test_out_path + ".tmp"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, test_out_path)
    logging.info(f"[{dataset}] wrote {test_out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MATCH stage 3 training + eval (reconstructed)"
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--fea-dim", type=int, default=DEFAULT_FEA_DIM)
    parser.add_argument("--num-epoch", type=int, default=DEFAULT_NUM_EPOCH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    datasets = ALL_DATASETS if args.all else [args.dataset]
    for ds in datasets:
        run_one_dataset(ds, args)

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
