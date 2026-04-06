"""
Unified runner for 5 idea pilots + baseline on HateMM.

Models:
  baseline    — WholeRationaleMLP (text only)
  gtt         — GroundedTokenTrust (token-level AV trust)
  vmbt        — VMBTModel + VMBTTrainer (stability regularization)
  borf        — BORF two-stage (text base + residual)
  rcd         — RCD three-stage (text teacher + audio/frame residual distillation)
  cmde        — CrossModalDescriptionEditor (AV-gated text editing)

Usage:
  python run_gpt_experiments.py --models baseline gtt vmbt borf rcd cmde
  python run_gpt_experiments.py --models baseline borf  # run subset
"""
import argparse
import copy
import json
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from dataset import get_dataloaders, load_features, load_labels, load_split_ids, KillTestDataset, collate_fn
from torch.utils.data import DataLoader
from collections import Counter

from models import WholeRationaleMLP
from model_gtt import GroundedTokenTrust
from model_vmbt import VMBTModel, VMBTTrainer
from model_borf import BORFTextBase, BORFResidual, BORFFullModel
from model_rcd import RCDTextTeacher, RCDResidualBranch, RCDCombined
from model_cmde import CrossModalDescriptionEditor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Standard training loop (shared) ───────────────────────────────

def compute_class_weights(labels_dict, num_classes):
    """Compute inverse-frequency class weights."""
    counts = Counter(labels_dict.values())
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        freq = counts.get(c, 1)
        weights.append(total / (num_classes * freq))
    # Normalize so min weight = 1.0
    min_w = min(weights)
    return torch.tensor([w / min_w for w in weights], dtype=torch.float)


def make_optimizer_and_scheduler(model, epochs, steps_per_epoch, lr=2e-4, weight_decay=0.02):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-2, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_standard(model, train_dl, valid_dl, epochs=50, lr=2e-4, patience=10,
                   extra_loss_fn=None, class_weight=None):
    """Standard training with optional extra loss. Returns best model."""
    model = model.to(DEVICE)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, epochs, len(train_dl), lr=lr
    )
    if class_weight is None:
        class_weight = torch.tensor([1.0, 1.5], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(DEVICE))

    best_val_acc = -1
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in valid_dl:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                logits = model(batch)
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(batch["label"].cpu().numpy())

        val_acc = accuracy_score(labels, preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


def train_vmbt(vmbt_trainer, train_dl, valid_dl, epochs=50, lr=2e-4, patience=10,
               class_weight=None):
    """Training loop with VMBT trainer (custom loss computation)."""
    model = vmbt_trainer.model.to(DEVICE)
    vmbt_trainer.teacher = vmbt_trainer.teacher.to(DEVICE)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, epochs, len(train_dl), lr=lr
    )
    if class_weight is None:
        class_weight = torch.tensor([1.0, 1.5], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(DEVICE))

    best_val_acc = -1
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()
            loss, logits = vmbt_trainer.training_step(batch, criterion)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            vmbt_trainer.step_ema()

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in valid_dl:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                logits = model(batch)
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(batch["label"].cpu().numpy())

        val_acc = accuracy_score(labels, preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


def train_borf(train_dl, valid_dl, epochs=50, lr=2e-4, patience=10,
               num_classes=2, class_weight=None):
    """Two-stage BORF training."""
    # Stage 1: Train text-only base
    text_base = BORFTextBase(num_classes=num_classes)
    text_base = train_standard(text_base, train_dl, valid_dl, epochs=epochs, lr=lr,
                              patience=patience, class_weight=class_weight)

    # Freeze text base
    text_base.eval()
    for p in text_base.parameters():
        p.requires_grad = False

    # Stage 2: Train residual branches
    residual = BORFResidual(num_classes=num_classes).to(DEVICE)
    full_model = BORFFullModel(text_base, residual).to(DEVICE)

    # Only optimize residual parameters
    optimizer, scheduler = make_optimizer_and_scheduler(
        residual, epochs, len(train_dl), lr=lr
    )
    if class_weight is None:
        class_weight = torch.tensor([1.0, 1.5], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(DEVICE))

    best_val_acc = -1
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        residual.train()
        for batch in train_dl:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            logits = full_model(batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(residual.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validate
        residual.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in valid_dl:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                logits = full_model(batch)
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(batch["label"].cpu().numpy())

        val_acc = accuracy_score(labels, preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in residual.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    residual.load_state_dict(best_state)
    return full_model


def train_rcd(train_dl, valid_dl, test_dl, epochs=50, lr=2e-4, patience=10,
              num_classes=2, class_weight=None):
    """Three-stage RCD training."""
    # Stage 1: Train text teacher
    text_teacher = RCDTextTeacher(num_classes=num_classes)
    text_teacher = train_standard(text_teacher, train_dl, valid_dl, epochs=epochs, lr=lr,
                                 patience=patience, class_weight=class_weight)
    text_teacher.eval()
    for p in text_teacher.parameters():
        p.requires_grad = False

    # Stage 2: Compute residual targets on training set
    residual_targets = {}
    text_teacher_dev = text_teacher.to(DEVICE)
    with torch.no_grad():
        for batch in train_dl:
            batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            text_logits = text_teacher_dev(batch_dev)
            text_probs = F.softmax(text_logits, dim=-1)  # [B, 2]
            labels = batch_dev["label"]
            one_hot = F.one_hot(labels, num_classes=num_classes).float()
            residuals = one_hot - text_probs  # [B, 2]

            for i, vid in enumerate(batch["video_id"]):
                residual_targets[vid] = residuals[i].cpu()

    # Compute sample weights (text uncertainty)
    sample_weights = {}
    with torch.no_grad():
        for batch in train_dl:
            batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            text_logits = text_teacher_dev(batch_dev)
            text_probs = F.softmax(text_logits, dim=-1)
            confidence = text_probs.gather(1, batch_dev["label"].unsqueeze(1)).squeeze(1)
            weights = 1.0 / (confidence + 1e-6)
            for i, vid in enumerate(batch["video_id"]):
                sample_weights[vid] = weights[i].cpu().item()

    # Normalize weights
    mean_w = np.mean(list(sample_weights.values()))
    sample_weights = {k: v / mean_w for k, v in sample_weights.items()}

    # Stage 3a: Train audio residual branch
    audio_branch = RCDResidualBranch(input_dim=768, num_classes=num_classes).to(DEVICE)
    audio_opt, audio_sched = make_optimizer_and_scheduler(
        audio_branch, epochs, len(train_dl), lr=lr
    )
    best_val_loss = float('inf')
    best_audio_state = None
    no_improve = 0

    for epoch in range(epochs):
        audio_branch.train()
        for batch in train_dl:
            batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            audio_opt.zero_grad()
            pred_res = audio_branch(batch_dev["audio"])  # [B, 2]
            targets = torch.stack([residual_targets[vid] for vid in batch["video_id"]]).to(DEVICE)
            weights = torch.tensor([sample_weights[vid] for vid in batch["video_id"]],
                                   dtype=torch.float).to(DEVICE)
            loss = (weights.unsqueeze(1) * (pred_res - targets).pow(2)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(audio_branch.parameters(), 1.0)
            audio_opt.step()
            audio_sched.step()

        # Validate with MSE
        audio_branch.eval()
        val_loss = 0
        n = 0
        with torch.no_grad():
            for batch in valid_dl:
                batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                text_logits = text_teacher_dev(batch_dev)
                text_probs = F.softmax(text_logits, dim=-1)
                labels = batch_dev["label"]
                one_hot = F.one_hot(labels, num_classes=num_classes).float()
                targets = one_hot - text_probs
                pred = audio_branch(batch_dev["audio"])
                val_loss += (pred - targets).pow(2).sum().item()
                n += labels.size(0)
        val_loss /= max(n, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_audio_state = {k: v.clone() for k, v in audio_branch.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    audio_branch.load_state_dict(best_audio_state)

    # Stage 3b: Train frame residual branch (same procedure)
    frame_branch = RCDResidualBranch(input_dim=768, num_classes=num_classes).to(DEVICE)
    frame_opt, frame_sched = make_optimizer_and_scheduler(
        frame_branch, epochs, len(train_dl), lr=lr
    )
    best_val_loss = float('inf')
    best_frame_state = None
    no_improve = 0

    for epoch in range(epochs):
        frame_branch.train()
        for batch in train_dl:
            batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            frame_opt.zero_grad()
            pred_res = frame_branch(batch_dev["frame"])
            targets = torch.stack([residual_targets[vid] for vid in batch["video_id"]]).to(DEVICE)
            weights = torch.tensor([sample_weights[vid] for vid in batch["video_id"]],
                                   dtype=torch.float).to(DEVICE)
            loss = (weights.unsqueeze(1) * (pred_res - targets).pow(2)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(frame_branch.parameters(), 1.0)
            frame_opt.step()
            frame_sched.step()

        frame_branch.eval()
        val_loss = 0
        n = 0
        with torch.no_grad():
            for batch in valid_dl:
                batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                text_logits = text_teacher_dev(batch_dev)
                text_probs = F.softmax(text_logits, dim=-1)
                labels = batch_dev["label"]
                one_hot = F.one_hot(labels, num_classes=num_classes).float()
                targets = one_hot - text_probs
                pred = frame_branch(batch_dev["frame"])
                val_loss += (pred - targets).pow(2).sum().item()
                n += labels.size(0)
        val_loss /= max(n, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_frame_state = {k: v.clone() for k, v in frame_branch.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    frame_branch.load_state_dict(best_frame_state)

    # Find best alpha, beta on validation set
    best_combo_f1 = -1
    best_alpha, best_beta = 0.3, 0.3
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7]:
        for beta in [0.1, 0.2, 0.3, 0.5, 0.7]:
            preds, labels = [], []
            with torch.no_grad():
                for batch in valid_dl:
                    batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
                    t_logit = text_teacher_dev(batch_dev)
                    a_res = audio_branch(batch_dev["audio"])
                    f_res = frame_branch(batch_dev["frame"])
                    combined = t_logit + alpha * a_res + beta * f_res
                    preds.extend(combined.argmax(1).cpu().numpy())
                    labels.extend(batch_dev["label"].cpu().numpy())
            f1 = f1_score(labels, preds, average="macro")
            if f1 > best_combo_f1:
                best_combo_f1 = f1
                best_alpha, best_beta = alpha, beta

    combined = RCDCombined(text_teacher, audio_branch, frame_branch,
                           alpha=best_alpha, beta=best_beta).to(DEVICE)
    return combined, best_alpha, best_beta


def evaluate(model, test_dl):
    """Evaluate model, return acc, macro-F1."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_dl:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits = model(batch)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return round(acc * 100, 2), round(f1 * 100, 2)


# ─── Dataset with optional token features ──────────────────────────

class ExtendedDataset(KillTestDataset):
    """Extends KillTestDataset with optional token-level features for GTT."""

    def __init__(self, video_ids, features, labels, token_features=None, token_masks=None):
        super().__init__(video_ids, features, labels)
        self.token_features = token_features
        self.token_masks = token_masks
        # Further filter to IDs that have token features if provided
        if token_features is not None:
            self.ids = [vid for vid in self.ids if vid in token_features]

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        vid = item["video_id"]
        if self.token_features is not None:
            item["unit_tokens"] = self.token_features[vid].float()  # [5, T, 768]
            item["unit_token_mask"] = self.token_masks[vid].float()  # [5, T]
        return item


def extended_collate_fn(batch):
    """Collate with optional token features."""
    result = {
        "video_id": [b["video_id"] for b in batch],
        "text": torch.stack([b["text"] for b in batch]),
        "units": torch.stack([b["units"] for b in batch]),
        "unit_mask": torch.stack([b["unit_mask"] for b in batch]),
        "audio": torch.stack([b["audio"] for b in batch]),
        "frame": torch.stack([b["frame"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }
    if "unit_tokens" in batch[0]:
        result["unit_tokens"] = torch.stack([b["unit_tokens"] for b in batch])
        result["unit_token_mask"] = torch.stack([b["unit_token_mask"] for b in batch])
    return result


# ─── Main ──────────────────────────────────────────────────────────

def run_one_seed(model_name, seed, train_dl, valid_dl, test_dl, epochs, logger,
                 num_classes=2, class_weight=None):
    """Run a single seed for a single model. Returns result dict."""
    set_seed(seed)

    if model_name == "baseline":
        model = WholeRationaleMLP(text_dim=768, num_classes=num_classes)
        model = train_standard(model, train_dl, valid_dl, epochs=epochs,
                              class_weight=class_weight)
        acc, f1 = evaluate(model, test_dl)

    elif model_name == "gtt":
        model = GroundedTokenTrust(text_dim=768, av_dim=768, num_classes=num_classes)
        model = train_standard(model, train_dl, valid_dl, epochs=epochs,
                              class_weight=class_weight)
        acc, f1 = evaluate(model, test_dl)

    elif model_name == "vmbt":
        base_model = VMBTModel(text_dim=768, av_dim=768, num_classes=num_classes)
        trainer = VMBTTrainer(base_model, ema_decay=0.999, lambda_fda=0.1, device=DEVICE)
        train_vmbt(trainer, train_dl, valid_dl, epochs=epochs, class_weight=class_weight)
        acc, f1 = evaluate(base_model, test_dl)

    elif model_name == "borf":
        full_model = train_borf(train_dl, valid_dl, epochs=epochs,
                               num_classes=num_classes, class_weight=class_weight)
        acc, f1 = evaluate(full_model, test_dl)

    elif model_name == "rcd":
        combined, alpha, beta = train_rcd(train_dl, valid_dl, test_dl, epochs=epochs,
                                          num_classes=num_classes, class_weight=class_weight)
        acc, f1 = evaluate(combined, test_dl)
        logger.info(f"    RCD best alpha={alpha}, beta={beta}")

    elif model_name == "cmde":
        model = CrossModalDescriptionEditor(text_dim=768, av_dim=768, num_classes=num_classes)
        extra_loss = lambda batch: model.edit_regularization_loss(batch)
        model = train_standard(model, train_dl, valid_dl, epochs=epochs,
                              extra_loss_fn=extra_loss, class_weight=class_weight)
        acc, f1 = evaluate(model, test_dl)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return {"model": model_name, "seed": seed, "acc": acc, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="GPT Experiment Pilot Runner")
    parser.add_argument("--emb_dir", default="/home/junyi/EMNLP2/embeddings/HateMM")
    parser.add_argument("--ann_path", default="/home/junyi/EMNLP2/datasets/HateMM/annotation(new).json")
    parser.add_argument("--split_dir", default="/home/junyi/EMNLP2/datasets/HateMM/splits")
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--seed_offset", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="/home/junyi/EMNLP2/kill_test/results/gpt_experiments")
    parser.add_argument("--models", nargs="+",
                        default=["baseline", "gtt", "vmbt", "borf", "rcd", "cmde"])
    parser.add_argument("--token_feature_path", default=None,
                        help="Path to unit_token_features.pth for GTT")
    parser.add_argument("--token_mask_path", default=None,
                        help="Path to unit_token_masks.pth for GTT")
    parser.add_argument("--force_binary", action="store_true",
                        help="Merge Offensive+Hateful→Hate for ternary datasets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"{args.output_dir}/pilot_{ts}.log"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    seeds = [args.seed_offset + i for i in range(args.num_seeds)]
    logger.info(f"GPT Experiment Pilot — models: {args.models}")
    logger.info(f"Seeds: {seeds}, Epochs: {args.epochs}")

    # Load base features
    features = load_features(args.emb_dir)
    labels = load_labels(args.ann_path, force_binary=args.force_binary)
    splits = load_split_ids(args.split_dir)

    # Auto-detect num_classes
    num_classes = max(labels.values()) + 1
    class_weight = compute_class_weights(labels, num_classes)
    logger.info(f"Detected {num_classes} classes, weights: {class_weight.tolist()}")

    # Load token features if available (for GTT)
    token_features = None
    token_masks = None
    if args.token_feature_path and os.path.exists(args.token_feature_path):
        logger.info(f"Loading token features from {args.token_feature_path}")
        token_features = torch.load(args.token_feature_path, map_location="cpu")
        token_masks = torch.load(args.token_mask_path, map_location="cpu")
        logger.info(f"  Loaded token features for {len(token_features)} videos")
    elif "gtt" in args.models:
        # Auto-detect
        auto_path = os.path.join(args.emb_dir, "unit_token_features.pth")
        auto_mask = os.path.join(args.emb_dir, "unit_token_masks.pth")
        if os.path.exists(auto_path):
            logger.info(f"Auto-detected token features at {auto_path}")
            token_features = torch.load(auto_path, map_location="cpu")
            token_masks = torch.load(auto_mask, map_location="cpu")
        else:
            logger.warning("GTT requested but no token features found. Skipping GTT.")
            args.models = [m for m in args.models if m != "gtt"]

    # Create dataloaders
    def make_loaders(need_tokens=False):
        tf = token_features if need_tokens else None
        tm = token_masks if need_tokens else None
        train_ds = ExtendedDataset(splits["train"], features, labels, tf, tm)
        valid_ds = ExtendedDataset(splits["valid"], features, labels, tf, tm)
        test_ds = ExtendedDataset(splits["test"], features, labels, tf, tm)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=extended_collate_fn)
        valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=extended_collate_fn)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=extended_collate_fn)
        return train_dl, valid_dl, test_dl

    # Standard loaders (no token features)
    train_dl, valid_dl, test_dl = make_loaders(need_tokens=False)
    logger.info(f"Data: train={len(train_dl.dataset)}, valid={len(valid_dl.dataset)}, test={len(test_dl.dataset)}")

    # Token loaders (for GTT)
    if "gtt" in args.models and token_features is not None:
        gtt_train_dl, gtt_valid_dl, gtt_test_dl = make_loaders(need_tokens=True)
    else:
        gtt_train_dl = gtt_valid_dl = gtt_test_dl = None

    all_results = []

    for model_name in args.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {model_name}")
        logger.info(f"{'='*60}")

        # Use token loaders for GTT
        if model_name == "gtt":
            m_train, m_valid, m_test = gtt_train_dl, gtt_valid_dl, gtt_test_dl
        else:
            m_train, m_valid, m_test = train_dl, valid_dl, test_dl

        model_results = []
        for seed in seeds:
            logger.info(f"  Seed {seed}...")
            try:
                result = run_one_seed(model_name, seed, m_train, m_valid, m_test,
                                      args.epochs, logger,
                                      num_classes=num_classes, class_weight=class_weight)
                model_results.append(result)
                logger.info(f"  Seed {seed}: ACC={result['acc']:.2f}, F1={result['f1']:.2f}")
            except Exception as e:
                logger.error(f"  Seed {seed} FAILED: {e}")
                model_results.append({
                    "model": model_name, "seed": seed,
                    "acc": None, "f1": None, "error": str(e)
                })

        # Summary
        valid_results = [r for r in model_results if r["f1"] is not None]
        if valid_results:
            accs = [r["acc"] for r in valid_results]
            f1s = [r["f1"] for r in valid_results]
            logger.info(f"\n  Summary for {model_name} ({len(valid_results)} seeds):")
            logger.info(f"    ACC: mean={np.mean(accs):.2f}, std={np.std(accs):.2f}")
            logger.info(f"    F1:  mean={np.mean(f1s):.2f}, std={np.std(f1s):.2f}, "
                        f"worst={np.min(f1s):.2f}, best={np.max(f1s):.2f}")

        all_results.extend(model_results)

    # Save all results
    result_path = f"{args.output_dir}/seed_results_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAll results saved to {result_path}")

    # Generate summary table
    generate_summary(all_results, args.output_dir, ts, logger)


def generate_summary(all_results, output_dir, ts, logger):
    """Generate summary table with verdicts."""
    by_model = {}
    for r in all_results:
        by_model.setdefault(r["model"], []).append(r)

    # Baseline stats
    baseline_results = [r for r in by_model.get("baseline", []) if r["f1"] is not None]
    if not baseline_results:
        logger.error("No baseline results! Cannot compute verdicts.")
        return

    bl_f1s = [r["f1"] for r in baseline_results]
    bl_mean_f1 = np.mean(bl_f1s)
    bl_std_f1 = np.std(bl_f1s)
    bl_worst_f1 = np.min(bl_f1s)

    lines = []
    lines.append("| Variant | Mean ACC | Std ACC | Mean F1 | Std F1 | Worst F1 | Best F1 | Δ Mean F1 | Verdict |")
    lines.append("|---------|----------|---------|---------|--------|----------|---------|-----------|---------|")

    for model_name in ["baseline", "gtt", "vmbt", "borf", "rcd", "cmde"]:
        results = [r for r in by_model.get(model_name, []) if r["f1"] is not None]
        if not results:
            lines.append(f"| {model_name} | — | — | — | — | — | — | — | Incomplete |")
            continue

        accs = [r["acc"] for r in results]
        f1s = [r["f1"] for r in results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        worst_f1 = np.min(f1s)
        best_f1 = np.max(f1s)
        delta_f1 = mean_f1 - bl_mean_f1

        # Verdict
        if model_name == "baseline":
            verdict = "Baseline"
        elif delta_f1 > 1.0:
            verdict = "Positive signal"
        elif (delta_f1 >= -0.5 and std_f1 <= 0.8 * bl_std_f1
              and worst_f1 >= bl_worst_f1 + 1.0):
            verdict = "Stability-only signal"
        elif delta_f1 > 0:
            verdict = "Borderline"
        else:
            verdict = "No signal"

        lines.append(
            f"| {model_name} | {mean_acc:.2f} | {std_acc:.2f} | {mean_f1:.2f} | "
            f"{std_f1:.2f} | {worst_f1:.2f} | {best_f1:.2f} | "
            f"{delta_f1:+.2f} | {verdict} |"
        )

    summary_md = "\n".join(lines)
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"\n{summary_md}")

    # Save
    with open(f"{output_dir}/summary_{ts}.md", "w") as f:
        f.write(f"# GPT Experiment Pilot Results\n\n")
        f.write(f"**Date**: {ts}\n")
        f.write(f"**Dataset**: HateMM\n")
        f.write(f"**Seeds**: 42-51 (10 seeds)\n\n")
        f.write(summary_md + "\n")

    logger.info(f"\nSummary saved to {output_dir}/summary_{ts}.md")


if __name__ == "__main__":
    main()
