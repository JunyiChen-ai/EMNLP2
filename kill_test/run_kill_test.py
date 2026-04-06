"""
Kill Test: Compare 4 model variants on HateMM, 5 seeds each.

Variants:
  1. whole_mlp: WholeRationaleMLP (baseline)
  2. text_unit_attn: TextUnitAttention (text-only sentence attention)
  3. pos_av_support: PositiveAVSupport (AV-conditioned positive weights)
  4. signed_auditor: SignedEvidenceAuditor (AV-conditioned signed accept/reject)

Kill criteria:
  - If text_unit_attn <= whole_mlp → unit decomposition doesn't help → kill all
  - If signed_auditor <= pos_av_support → signed mechanism adds nothing → kill signed
  - If reject fraction < 5% → rejection not used → signed is just positive attention
  - If gains < 1pp over whole_mlp → not worth the complexity
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
from sklearn.metrics import accuracy_score, f1_score

from dataset import get_dataloaders
from models import (
    WholeRationaleMLP,
    TextUnitAttention,
    PositiveAVSupport,
    SignedEvidenceAuditor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_model(model, train_dl, valid_dl, epochs=50, lr=2e-4, weight_decay=0.02, patience=10):
    """Train model, return best state dict (by valid acc)."""
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Class weights for imbalanced data (Non Hate: 639, Hate: 427 → ~1.5x)
    class_weight = torch.tensor([1.0, 1.5], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    # Cosine schedule with warmup
    total_steps = epochs * len(train_dl)
    warmup_steps = 5 * len(train_dl)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-2, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = -1
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            batch = {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in valid_dl:
                batch = {
                    k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
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


def evaluate(model, test_dl):
    """Evaluate model, return acc, macro-F1, per-sample logits."""
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            batch = {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
            all_logits.append(logits.cpu())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, torch.cat(all_logits)


def analyze_audit_scores(model, test_dl):
    """Analyze signed audit score distribution (for SignedEvidenceAuditor only).
    Returns sample-averaged statistics, not batch-averaged."""
    model.eval()
    all_scores = []
    all_masks = []
    with torch.no_grad():
        for batch in test_dl:
            batch = {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            s = model.get_audit_scores(batch)  # [B, K]
            all_scores.append(s.cpu())
            all_masks.append(batch["unit_mask"].cpu())

    scores = torch.cat(all_scores, dim=0)  # [N, K]
    masks = torch.cat(all_masks, dim=0)  # [N, K]
    valid_per_sample = masks.sum(dim=-1).clamp(min=1)  # [N]

    accept_per_sample = ((scores > 0.2) & (masks == 1)).float().sum(dim=-1) / valid_per_sample
    ignore_per_sample = ((scores.abs() <= 0.2) & (masks == 1)).float().sum(dim=-1) / valid_per_sample
    reject_per_sample = ((scores < -0.2) & (masks == 1)).float().sum(dim=-1) / valid_per_sample
    valid_scores = scores[masks == 1]

    return {
        "accept_frac": accept_per_sample.mean().item(),
        "ignore_frac": ignore_per_sample.mean().item(),
        "reject_frac": reject_per_sample.mean().item(),
        "mean_score": valid_scores.mean().item(),
        "std_score": valid_scores.std().item(),
    }


def analyze_attention_weights(model, test_dl):
    """Analyze attention weight entropy (for TextUnitAttention or PositiveAVSupport)."""
    model.eval()
    all_entropy = []
    with torch.no_grad():
        for batch in test_dl:
            batch = {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            weights = model.get_weights(batch)  # [B, K]
            # Entropy: -sum(w * log(w))
            eps = 1e-8
            entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)  # [B]
            all_entropy.extend(entropy.cpu().numpy())

    max_entropy = np.log(5)  # uniform over 5 units
    return {
        "mean_entropy": np.mean(all_entropy),
        "max_entropy": max_entropy,
        "normalized_entropy": np.mean(all_entropy) / max_entropy,
    }


def run_variant(variant_name, model_cls, train_dl, valid_dl, test_dl, seed, epochs=50, **kwargs):
    """Train and evaluate one variant for one seed."""
    set_seed(seed)
    model = model_cls(**kwargs)
    model = train_one_model(model, train_dl, valid_dl, epochs=epochs)
    acc, f1, logits = evaluate(model, test_dl)

    result = {
        "variant": variant_name,
        "seed": seed,
        "acc": round(acc * 100, 2),
        "f1": round(f1 * 100, 2),
    }

    # Variant-specific analysis
    if isinstance(model, SignedEvidenceAuditor):
        audit_stats = analyze_audit_scores(model, test_dl)
        result["audit_stats"] = audit_stats
    elif isinstance(model, (TextUnitAttention, PositiveAVSupport)):
        attn_stats = analyze_attention_weights(model, test_dl)
        result["attn_stats"] = attn_stats

    return result


def main():
    parser = argparse.ArgumentParser(description="Kill Test: Signed Evidence Auditing")
    parser.add_argument("--emb_dir", default="/home/junyi/EMNLP2/embeddings/HateMM")
    parser.add_argument(
        "--ann_path",
        default="/home/junyi/EMNLP2/datasets/HateMM/annotation(new).json",
    )
    parser.add_argument(
        "--split_dir", default="/home/junyi/EMNLP2/datasets/HateMM/splits"
    )
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--seed_offset", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="/home/junyi/EMNLP2/kill_test/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check that unit_features.pth exists
    unit_path = os.path.join(args.emb_dir, "unit_features.pth")
    if not os.path.exists(unit_path):
        print(f"ERROR: {unit_path} not found.")
        print("Run prepare_unit_embeddings.py first:")
        print(f"  conda run -n HVGuard python kill_test/prepare_unit_embeddings.py")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"{args.output_dir}/kill_test_{ts}.log"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    seeds = [args.seed_offset + i for i in range(args.num_seeds)]
    logger.info(f"Kill Test — {args.num_seeds} seeds: {seeds}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

    # Load data once
    train_dl, valid_dl, test_dl = get_dataloaders(
        args.emb_dir, args.ann_path, args.split_dir, args.batch_size
    )
    logger.info(
        f"Data loaded: train={len(train_dl.dataset)}, "
        f"valid={len(valid_dl.dataset)}, test={len(test_dl.dataset)}"
    )

    variants = [
        ("whole_mlp", WholeRationaleMLP, {"text_dim": 768}),
        ("text_unit_attn", TextUnitAttention, {"text_dim": 768}),
        (
            "pos_av_support",
            PositiveAVSupport,
            {"text_dim": 768, "av_dim": 768},
        ),
        (
            "signed_auditor",
            SignedEvidenceAuditor,
            {"text_dim": 768, "av_dim": 768},
        ),
    ]

    all_results = []

    for variant_name, model_cls, kwargs in variants:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running variant: {variant_name}")
        logger.info(f"{'='*60}")

        variant_results = []
        for seed in seeds:
            logger.info(f"  Seed {seed}...")
            result = run_variant(
                variant_name, model_cls, train_dl, valid_dl, test_dl, seed,
                epochs=args.epochs, **kwargs
            )
            variant_results.append(result)
            logger.info(
                f"  Seed {seed}: ACC={result['acc']:.2f}, F1={result['f1']:.2f}"
            )
            if "audit_stats" in result:
                stats = result["audit_stats"]
                logger.info(
                    f"    Audit: accept={stats['accept_frac']:.3f}, "
                    f"ignore={stats['ignore_frac']:.3f}, "
                    f"reject={stats['reject_frac']:.3f}, "
                    f"mean_s={stats['mean_score']:.3f}, std_s={stats['std_score']:.3f}"
                )
            if "attn_stats" in result:
                stats = result["attn_stats"]
                logger.info(
                    f"    Attn entropy: {stats['mean_entropy']:.3f} "
                    f"(normalized: {stats['normalized_entropy']:.3f})"
                )

        accs = [r["acc"] for r in variant_results]
        f1s = [r["f1"] for r in variant_results]
        logger.info(f"\n  Summary for {variant_name}:")
        logger.info(
            f"    ACC: mean={np.mean(accs):.2f}, std={np.std(accs):.2f}, "
            f"worst={np.min(accs):.2f}, best={np.max(accs):.2f}"
        )
        logger.info(
            f"    F1:  mean={np.mean(f1s):.2f}, std={np.std(f1s):.2f}, "
            f"worst={np.min(f1s):.2f}, best={np.max(f1s):.2f}"
        )

        all_results.extend(variant_results)

    # Save results
    # Convert numpy types for JSON serialization
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    result_path = f"{args.output_dir}/kill_test_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(to_native(all_results), f, indent=2)
    logger.info(f"\nResults saved to {result_path}")

    # Kill test verdict
    logger.info(f"\n{'='*60}")
    logger.info("KILL TEST VERDICT")
    logger.info(f"{'='*60}")

    by_variant = {}
    for r in all_results:
        by_variant.setdefault(r["variant"], []).append(r)

    def mean_acc(name):
        return np.mean([r["acc"] for r in by_variant[name]])

    def mean_f1(name):
        return np.mean([r["f1"] for r in by_variant[name]])

    baseline_acc = mean_acc("whole_mlp")
    baseline_f1 = mean_f1("whole_mlp")

    for name in ["text_unit_attn", "pos_av_support", "signed_auditor"]:
        delta_acc = mean_acc(name) - baseline_acc
        delta_f1 = mean_f1(name) - baseline_f1
        logger.info(f"  {name}: Δacc={delta_acc:+.2f}pp, Δf1={delta_f1:+.2f}pp")

    # Check kill criteria
    kill_reasons = []

    # Kill 1: unit decomposition doesn't help
    if mean_acc("text_unit_attn") <= baseline_acc:
        kill_reasons.append(
            "KILL: text_unit_attn <= whole_mlp → unit decomposition doesn't help"
        )

    # Kill 2: signed doesn't beat positive
    if mean_acc("signed_auditor") <= mean_acc("pos_av_support"):
        kill_reasons.append(
            "KILL: signed_auditor <= pos_av_support → signed mechanism adds nothing"
        )

    # Kill 3: rejection not used
    audit_results = by_variant.get("signed_auditor", [])
    if audit_results and all("audit_stats" in r for r in audit_results):
        avg_reject = np.mean([r["audit_stats"]["reject_frac"] for r in audit_results])
        if avg_reject < 0.05:
            kill_reasons.append(
                f"KILL: reject fraction = {avg_reject:.3f} < 0.05 → rejection not used"
            )

    # Kill 4: gains too small
    best_delta = max(
        mean_acc("signed_auditor") - baseline_acc,
        mean_acc("pos_av_support") - baseline_acc,
    )
    if best_delta < 1.0:
        kill_reasons.append(
            f"KILL: best gain over baseline = {best_delta:.2f}pp < 1pp → not worth complexity"
        )

    if kill_reasons:
        logger.info("\n  ⚠️  KILL SIGNALS:")
        for reason in kill_reasons:
            logger.info(f"    {reason}")
    else:
        logger.info("\n  ✅ NO KILL SIGNALS — proceed to full experiment")

    logger.info(f"\n{'='*60}")


if __name__ == "__main__":
    main()
