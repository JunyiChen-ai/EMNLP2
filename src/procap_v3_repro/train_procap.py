"""Pro-Cap V3 — supervised RoBERTa-large PromptHateModel trainer.

Source: `external_repos/procap/codes/scr/train.py:48-204` (the
`pbm` branch only — wandb / Hydra / FIX_LAYERS / KLDivLoss / multi-
query inference are all stripped). Hyperparameters from upstream
`scr/config.py` (lr=1e-5, weight_decay=0.01, AdamW eps=1e-8,
EPOCHS=10, BATCH_SIZE=16, MAX_LENGTH=320, USE_DEMO=True,
NUM_SAMPLE=1) are kept as defaults.

What we do per dataset:
  1. Load Pro-Cap V3 captions (`results/procap_v3/<dataset>/captions_<split>.pkl`)
     for both train and test splits via `dataset_procap.build_entries`.
  2. Stratified 80/20 split of train_clean.csv → (train, valid)
     subsets (`stratified_train_valid_split`, seed=2025).
  3. Wrap into `Multimodal_Data` with the train subset as the
     in-context support pool for both train + valid + test phases.
     Upstream `dataset.py:76` uses `self.support_examples = self.load_entries('train')`
     unconditionally, regardless of the current `mode` — same here.
  4. Train PBM with AdamW + BCELoss for `--num-epoch` epochs with
     early stopping on valid accuracy (`--patience` epochs).
  5. Reload best epoch state, run inference on test split, write
     per-video predictions to
     `results/procap_v3/<dataset>/test_procap.jsonl` with schema
     `{video_id, pred, score}`.

Loss: upstream `train.py:13-16` `bce_for_loss` =
`F.binary_cross_entropy_with_logits(logits, labels) * num_classes`.
We use the same call (the `* num_classes` factor mirrors upstream).

Score: `softmax(logits, dim=-1)[:, 1]` is the upstream "harmful
probability" used for AUC (`train.py:195-197`). We use the same
quantity as the per-video `score` column of the output jsonl.
"""

import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dataset_procap import (  # noqa: E402
    Multimodal_Data,
    build_entries,
    stratified_train_valid_split,
    LABEL_WORDS,
)
from pbm import PromptHateModel  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bce_for_loss(logits, labels):
    """Upstream `train.py:13-16` verbatim."""
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score(logits, labels):
    """Upstream `train.py:25-32` verbatim."""
    pred = torch.max(logits, 1)[1]
    one_hot = torch.zeros_like(labels)
    one_hot.scatter_(1, pred.view(-1, 1), 1)
    score = one_hot * labels
    return score.sum().float()


def collate_fn(batch):
    """Stack the items returned by `Multimodal_Data.__getitem__` into a
    batch dict. Text fields are kept as a Python list (RoBERTa
    tokenization happens inside the model's `forward`). Tensor fields
    are stacked.
    """
    keys_text = ("prompt_all_text", "test_all_text", "test_text", "img")
    keys_tensor = ("target", "label")
    out = {}
    for k in keys_text:
        out[k] = [b[k] for b in batch]
    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch])
    return out


def make_loader(entries, support_entries, dataset, batch_size, shuffle):
    ds = Multimodal_Data(
        dataset=dataset,
        entries=entries,
        support_entries=support_entries,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )


def evaluate(model, loader, device, use_demo=True):
    """Run inference on `loader` and return
    `(accuracy, [(vid, pred, score)...])` where score is the harmful
    probability (softmax index 1).
    """
    model.eval()
    correct = 0
    total = 0
    rows = []
    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(device)
            text = batch["prompt_all_text"] if use_demo else batch["test_all_text"]
            logits = model(text)
            prob = F.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == batch["label"].to(device)).sum().item()
            total += target.size(0)
            for i, vid in enumerate(batch["img"]):
                rows.append({
                    "video_id": vid,
                    "pred": int(pred[i].item()),
                    "score": float(prob[i, 1].item()),
                })
    acc = correct / max(total, 1)
    return acc, rows


def train_one_dataset(dataset, args):
    set_seed(args.seed)
    logging.info(f"=== {dataset} === seed={args.seed}")

    train_entries, train_missing = build_entries(dataset, "train")
    test_entries, test_missing = build_entries(dataset, "test")
    logging.info(
        f"  train={len(train_entries)} (missing caps {len(train_missing)})  "
        f"test={len(test_entries)} (missing caps {len(test_missing)})"
    )

    train_subset, valid_subset = stratified_train_valid_split(
        train_entries, valid_frac=args.valid_frac, seed=args.seed
    )
    logging.info(
        f"  stratified split → train={len(train_subset)} valid={len(valid_subset)}"
    )

    # Upstream `dataset.py:76` always loads the *training* split as
    # the in-context support pool, regardless of which mode is active.
    # We follow the same convention: the train subset (post 80/20
    # split) is the support pool for train, valid, AND test phases.
    support_pool = train_subset

    train_loader = make_loader(
        train_subset, support_pool, dataset, args.batch_size, shuffle=True
    )
    valid_loader = make_loader(
        valid_subset, support_pool, dataset, args.batch_size, shuffle=False
    )
    test_loader = make_loader(
        test_entries, support_pool, dataset, args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptHateModel(
        label_words=LABEL_WORDS,
        max_length=args.max_length,
        model_name=args.text_encoder,
    )
    model.to(device)

    # Upstream `train.py:68-90` FIX_LAYERS filter (default FIX_LAYERS=2
    # per `config.py:33`). Freezes:
    #   - all `embeddings.*` parameters
    #   - `encoder.layer.<N>.*` for N < fix_layers
    # Trains:
    #   - `encoder.layer.<N>.*` for N >= fix_layers
    #   - everything else (lm_head, pooler, etc.)
    # When fix_layers <= 0 the filter is bypassed and all parameters
    # are trained, matching upstream's `else: params[n] = p` branch.
    params = {}
    for n, p in model.named_parameters():
        if args.fix_layers > 0:
            if "encoder.layer" in n:
                try:
                    layer_num = int(
                        n[n.find("encoder.layer") + 14:].split(".")[0]
                    )
                except Exception:
                    raise Exception(
                        f"could not parse encoder layer number from {n}"
                    )
                if layer_num >= args.fix_layers:
                    params[n] = p
            elif "embeddings" in n:
                pass
            else:
                params[n] = p
        else:
            params[n] = p
    n_train_p = len(params)
    n_total_p = sum(1 for _ in model.named_parameters())
    logging.info(
        f"  FIX_LAYERS={args.fix_layers}: training {n_train_p}/{n_total_p} "
        f"named parameters"
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optim_groups = [
        {
            "params": [
                p for n, p in params.items()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in params.items()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # transformers.AdamW was removed in transformers>=4.46; use torch.optim.AdamW.
    from torch.optim import AdamW
    optimizer = AdamW(optim_groups, lr=args.lr, eps=args.eps)

    out_dir = os.path.join(PROJECT_ROOT, "results", "procap_v3", dataset)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_pbm.pth")

    best_valid_acc = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_improvement = 0
    t0 = time.time()

    for epoch in range(args.num_epoch):
        model.train()
        total_loss = 0.0
        train_correct = 0
        n_train = 0
        for i, batch in enumerate(train_loader):
            target = batch["target"].to(device)
            text = batch["prompt_all_text"]
            logits = model(text)
            loss = bce_for_loss(logits, target)
            score = compute_score(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_correct += score.item()
            n_train += target.size(0)

        train_acc = train_correct / max(n_train, 1)
        valid_acc, _ = evaluate(model, valid_loader, device)
        elapsed = time.time() - t0
        logging.info(
            f"  epoch {epoch:02d} | train_loss={total_loss:.3f} "
            f"train_acc={train_acc:.4f} valid_acc={valid_acc:.4f} "
            f"elapsed={elapsed:.0f}s"
        )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
            torch.save(best_state, best_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                logging.info(
                    f"  early stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

    logging.info(
        f"  best epoch={best_epoch} valid_acc={best_valid_acc:.4f} → {best_path}"
    )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_rows = evaluate(model, test_loader, device)
    logging.info(f"  test_acc={test_acc:.4f}  rows={len(test_rows)}")

    out_jsonl = os.path.join(out_dir, "test_procap.jsonl")
    with open(out_jsonl, "w") as f:
        for r in test_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    logging.info(f"  wrote {out_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="Pro-Cap V3 supervised PBM trainer (RoBERTa-large)"
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--text-encoder", default="roberta-large")
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)
    # Upstream `config.py:33` --FIX_LAYERS default=2. Freezes
    # `embeddings.*` and `encoder.layer.0/1.*`; trains layers 2..23 +
    # lm_head + everything else. Set to 0 to disable freezing.
    parser.add_argument("--fix-layers", type=int, default=2)
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    datasets = ALL_DATASETS if args.all else [args.dataset]
    logging.info(
        f"Pro-Cap V3 trainer: datasets={datasets} text_encoder={args.text_encoder} "
        f"lr={args.lr} wd={args.weight_decay} bs={args.batch_size} "
        f"epochs={args.num_epoch} patience={args.patience} seed={args.seed}"
    )

    for ds in datasets:
        train_one_dataset(ds, args)

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
