"""
Error Pattern Analysis for MLLM Textualization Pipeline.

Run the baseline (whole rationale MLP) across 10 seeds, then analyze:
1. Which samples are consistently misclassified? (hard cases)
2. Which samples flip between correct/wrong across seeds? (unstable cases)
3. What do the rationale texts look like for each error pattern?
4. Per-field analysis: which rationale fields correlate with errors?
5. Transcript vs rationale divergence: does MLLM miss things?
6. Label-conditioned patterns: FP vs FN characteristics
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_dataloaders, load_labels, load_split_ids, KillTestDataset, collate_fn, load_features
from models import WholeRationaleMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_baseline(train_dl, valid_dl, seed, epochs=50, lr=2e-4):
    set_seed(seed)
    model = WholeRationaleMLP(text_dim=768).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    class_weight = torch.tensor([1.0, 1.5], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    total_steps = epochs * len(train_dl)
    warmup_steps = 5 * len(train_dl)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-2, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_acc, best_state, no_improve = -1, None, 0

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in valid_dl:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                preds.extend(model(batch).argmax(1).cpu().numpy())
                labels.extend(batch["label"].cpu().numpy())
        val_acc = accuracy_score(labels, preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break

    model.load_state_dict(best_state)
    return model


def get_per_sample_predictions(model, dataloader):
    """Return {video_id: (pred, prob, label)} for each sample."""
    model.eval()
    results = {}
    with torch.no_grad():
        for batch in dataloader:
            batch_gpu = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch_gpu)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(1).cpu().numpy()
            labels = batch["label"].numpy()
            for i, vid in enumerate(batch["video_id"]):
                results[vid] = {
                    "pred": int(preds[i]),
                    "prob_hate": float(probs[i, 1].cpu()),
                    "label": int(labels[i]),
                    "correct": int(preds[i]) == int(labels[i]),
                }
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", default="/home/junyi/EMNLP2/embeddings/HateMM")
    parser.add_argument("--ann_path", default="/home/junyi/EMNLP2/datasets/HateMM/annotation(new).json")
    parser.add_argument("--split_dir", default="/home/junyi/EMNLP2/datasets/HateMM/splits")
    parser.add_argument("--data_path", default="/home/junyi/EMNLP2/datasets/HateMM/generic_data.json")
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--output_dir", default="/home/junyi/EMNLP2/kill_test/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    features = load_features(args.emb_dir)
    labels = load_labels(args.ann_path)
    splits = load_split_ids(args.split_dir)
    train_ds = KillTestDataset(splits["train"], features, labels)
    valid_ds = KillTestDataset(splits["valid"], features, labels)
    test_ds = KillTestDataset(splits["test"], features, labels)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Load raw rationale text
    with open(args.data_path, "r") as f:
        raw_data = {d["Video_ID"]: d for d in json.load(f)}

    print(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    # Run 10 seeds
    seeds = list(range(42, 42 + args.num_seeds))
    all_seed_results = {}  # seed → {vid: result}

    for seed in seeds:
        print(f"\nTraining seed {seed}...")
        model = train_baseline(train_dl, valid_dl, seed)
        results = get_per_sample_predictions(model, test_dl)
        acc = sum(r["correct"] for r in results.values()) / len(results)
        print(f"  Seed {seed}: ACC={acc*100:.2f}")
        all_seed_results[seed] = results

    # =========================================================
    # Analysis 1: Per-sample correctness across seeds
    # =========================================================
    test_vids = list(all_seed_results[seeds[0]].keys())
    vid_correct_count = {}
    for vid in test_vids:
        correct = sum(all_seed_results[s][vid]["correct"] for s in seeds)
        vid_correct_count[vid] = correct

    # Categorize
    always_correct = [v for v, c in vid_correct_count.items() if c == len(seeds)]
    always_wrong = [v for v, c in vid_correct_count.items() if c == 0]
    mostly_wrong = [v for v, c in vid_correct_count.items() if 0 < c <= 2]
    unstable = [v for v, c in vid_correct_count.items() if 3 <= c <= len(seeds) - 3]
    mostly_correct = [v for v, c in vid_correct_count.items() if len(seeds) - 2 <= c < len(seeds)]

    total = len(test_vids)
    print(f"\n{'='*70}")
    print(f"ANALYSIS 1: Sample Stability (across {len(seeds)} seeds)")
    print(f"{'='*70}")
    print(f"  Always correct  (10/10): {len(always_correct)} ({len(always_correct)/total*100:.1f}%)")
    print(f"  Mostly correct   (8-9): {len(mostly_correct)} ({len(mostly_correct)/total*100:.1f}%)")
    print(f"  Unstable         (3-7): {len(unstable)} ({len(unstable)/total*100:.1f}%)")
    print(f"  Mostly wrong     (1-2): {len(mostly_wrong)} ({len(mostly_wrong)/total*100:.1f}%)")
    print(f"  Always wrong     (0/10): {len(always_wrong)} ({len(always_wrong)/total*100:.1f}%)")

    # =========================================================
    # Analysis 2: FP vs FN breakdown
    # =========================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 2: Error Types (FP vs FN)")
    print(f"{'='*70}")

    # Use majority vote across seeds for stable error type
    for category, vids, desc in [
        ("always_wrong", always_wrong, "Always Wrong"),
        ("mostly_wrong", mostly_wrong, "Mostly Wrong"),
        ("unstable", unstable, "Unstable"),
    ]:
        if not vids:
            continue
        fp_count = sum(1 for v in vids if labels[v] == 0)  # label=NonHate, pred=Hate
        fn_count = sum(1 for v in vids if labels[v] == 1)  # label=Hate, pred=NonHate
        print(f"\n  {desc} ({len(vids)} samples):")
        print(f"    False Positive (NonHate→Hate): {fp_count} ({fp_count/len(vids)*100:.1f}%)")
        print(f"    False Negative (Hate→NonHate): {fn_count} ({fn_count/len(vids)*100:.1f}%)")

    # =========================================================
    # Analysis 3: Rationale text patterns for error categories
    # =========================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 3: Rationale Characteristics by Error Category")
    print(f"{'='*70}")

    fields = ["content_summary", "target_analysis", "sentiment_tone", "harm_assessment", "overall_judgment"]

    for category, vids, desc in [
        ("always_wrong", always_wrong, "Always Wrong"),
        ("mostly_wrong", mostly_wrong, "Mostly Wrong"),
        ("unstable", unstable, "Unstable"),
        ("always_correct", always_correct, "Always Correct"),
    ]:
        if not vids:
            continue
        # Rationale lengths
        rationale_lens = []
        field_lens = {f: [] for f in fields}
        has_no_target = 0
        judgment_says_hate = 0
        judgment_says_not_hate = 0

        for vid in vids:
            if vid not in raw_data:
                continue
            resp = raw_data[vid].get("generic_response", {})
            full_text = " ".join(resp.get(f, "") or "" for f in fields)
            rationale_lens.append(len(full_text))
            for f in fields:
                field_lens[f].append(len(resp.get(f, "") or ""))

            # Check if target_analysis says "No"
            ta = (resp.get("target_analysis", "") or "").lower()
            if ta.startswith("no"):
                has_no_target += 1

            # Check judgment
            oj = (resp.get("overall_judgment", "") or "").lower()
            if "yes" in oj[:20] or "hateful" in oj[:50]:
                judgment_says_hate += 1
            elif "no" in oj[:20] or "not hateful" in oj[:50]:
                judgment_says_not_hate += 1

        n = len(vids)
        print(f"\n  {desc} ({n} samples):")
        print(f"    Avg rationale length: {np.mean(rationale_lens):.0f} chars")
        for f in fields:
            print(f"    Avg {f} length: {np.mean(field_lens[f]):.0f} chars")
        print(f"    target_analysis starts with 'No': {has_no_target} ({has_no_target/n*100:.1f}%)")
        print(f"    overall_judgment says hateful: {judgment_says_hate} ({judgment_says_hate/n*100:.1f}%)")
        print(f"    overall_judgment says not hateful: {judgment_says_not_hate} ({judgment_says_not_hate/n*100:.1f}%)")

    # =========================================================
    # Analysis 4: MLLM judgment vs ground truth
    # =========================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 4: MLLM Judgment Alignment with Ground Truth")
    print(f"{'='*70}")

    mllm_correct = 0
    mllm_fp = 0  # MLLM says hate but label is non-hate
    mllm_fn = 0  # MLLM says non-hate but label is hate
    mllm_ambiguous = 0
    total_test = 0

    for vid in test_vids:
        if vid not in raw_data:
            continue
        resp = raw_data[vid].get("generic_response", {})
        oj = (resp.get("overall_judgment", "") or "").lower()
        label = labels[vid]
        total_test += 1

        mllm_says_hate = "yes" in oj[:20] or ("hateful" in oj[:50] and "not hateful" not in oj[:50])
        mllm_says_not_hate = "no" in oj[:5] or "not hateful" in oj[:50]

        if mllm_says_hate and not mllm_says_not_hate:
            if label == 1:
                mllm_correct += 1
            else:
                mllm_fp += 1
        elif mllm_says_not_hate and not mllm_says_hate:
            if label == 0:
                mllm_correct += 1
            else:
                mllm_fn += 1
        else:
            mllm_ambiguous += 1

    print(f"  MLLM judgment correct: {mllm_correct} ({mllm_correct/total_test*100:.1f}%)")
    print(f"  MLLM FP (says hate, label non-hate): {mllm_fp} ({mllm_fp/total_test*100:.1f}%)")
    print(f"  MLLM FN (says non-hate, label hate): {mllm_fn} ({mllm_fn/total_test*100:.1f}%)")
    print(f"  MLLM ambiguous/unclear: {mllm_ambiguous} ({mllm_ambiguous/total_test*100:.1f}%)")

    # Cross with classifier errors
    print(f"\n  Cross-analysis: MLLM judgment × classifier error pattern:")
    for category, vids, desc in [
        ("always_wrong", always_wrong, "Always Wrong"),
        ("mostly_wrong", mostly_wrong, "Mostly Wrong"),
        ("unstable", unstable, "Unstable"),
    ]:
        if not vids:
            continue
        mllm_agree_label = 0
        mllm_disagree_label = 0
        mllm_unclear = 0
        for vid in vids:
            if vid not in raw_data:
                continue
            resp = raw_data[vid].get("generic_response", {})
            oj = (resp.get("overall_judgment", "") or "").lower()
            label = labels[vid]
            mllm_says_hate = "yes" in oj[:20] or ("hateful" in oj[:50] and "not hateful" not in oj[:50])
            mllm_says_not_hate = "no" in oj[:5] or "not hateful" in oj[:50]

            if mllm_says_hate and not mllm_says_not_hate:
                if label == 1:
                    mllm_agree_label += 1
                else:
                    mllm_disagree_label += 1
            elif mllm_says_not_hate and not mllm_says_hate:
                if label == 0:
                    mllm_agree_label += 1
                else:
                    mllm_disagree_label += 1
            else:
                mllm_unclear += 1

        n = len(vids)
        print(f"    {desc}: MLLM agrees with label={mllm_agree_label} ({mllm_agree_label/n*100:.1f}%), "
              f"disagrees={mllm_disagree_label} ({mllm_disagree_label/n*100:.1f}%), "
              f"unclear={mllm_unclear} ({mllm_unclear/n*100:.1f}%)")

    # =========================================================
    # Analysis 5: Transcript availability and length
    # =========================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 5: Transcript Characteristics by Error Category")
    print(f"{'='*70}")

    for category, vids, desc in [
        ("always_wrong", always_wrong, "Always Wrong"),
        ("mostly_wrong", mostly_wrong, "Mostly Wrong"),
        ("unstable", unstable, "Unstable"),
        ("always_correct", always_correct, "Always Correct"),
    ]:
        if not vids:
            continue
        transcript_lens = []
        empty_transcripts = 0
        for vid in vids:
            if vid not in raw_data:
                continue
            t = raw_data[vid].get("Transcript", "") or ""
            transcript_lens.append(len(t))
            if len(t.strip()) < 5:
                empty_transcripts += 1

        n = len(vids)
        print(f"  {desc} ({n} samples):")
        print(f"    Avg transcript length: {np.mean(transcript_lens):.0f} chars")
        print(f"    Empty/very short transcripts: {empty_transcripts} ({empty_transcripts/n*100:.1f}%)")

    # =========================================================
    # Analysis 6: Print example always-wrong and unstable cases
    # =========================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 6: Example Cases")
    print(f"{'='*70}")

    for category, vids, desc in [
        ("always_wrong", always_wrong[:8], "Always Wrong (first 8)"),
        ("unstable", unstable[:8], "Unstable (first 8)"),
    ]:
        if not vids:
            continue
        print(f"\n--- {desc} ---")
        for vid in vids:
            if vid not in raw_data:
                continue
            entry = raw_data[vid]
            resp = entry.get("generic_response", {})
            label = "HATE" if labels[vid] == 1 else "NON-HATE"
            correct_count = vid_correct_count[vid]

            # Average predicted prob across seeds
            avg_prob = np.mean([all_seed_results[s][vid]["prob_hate"] for s in seeds])

            print(f"\n  [{vid}] Label={label}, Correct={correct_count}/{len(seeds)}, Avg P(hate)={avg_prob:.3f}")
            print(f"    Transcript: {(entry.get('Transcript', '') or '')[:200]}...")
            print(f"    Content Summary: {(resp.get('content_summary', '') or '')[:200]}...")
            print(f"    Target Analysis: {(resp.get('target_analysis', '') or '')[:150]}...")
            print(f"    Sentiment/Tone: {(resp.get('sentiment_tone', '') or '')[:100]}")
            print(f"    Harm Assessment: {(resp.get('harm_assessment', '') or '')[:150]}...")
            print(f"    Overall Judgment: {(resp.get('overall_judgment', '') or '')[:150]}...")

    # =========================================================
    # Save structured results
    # =========================================================
    output = {
        "summary": {
            "num_seeds": len(seeds),
            "test_size": len(test_vids),
            "always_correct": len(always_correct),
            "mostly_correct": len(mostly_correct),
            "unstable": len(unstable),
            "mostly_wrong": len(mostly_wrong),
            "always_wrong": len(always_wrong),
        },
        "per_sample": {
            vid: {
                "label": labels[vid],
                "correct_count": vid_correct_count[vid],
                "avg_prob_hate": float(np.mean([all_seed_results[s][vid]["prob_hate"] for s in seeds])),
                "category": (
                    "always_correct" if vid_correct_count[vid] == len(seeds)
                    else "always_wrong" if vid_correct_count[vid] == 0
                    else "mostly_wrong" if vid_correct_count[vid] <= 2
                    else "unstable" if vid_correct_count[vid] <= len(seeds) - 3
                    else "mostly_correct"
                ),
            }
            for vid in test_vids
        },
    }

    out_path = os.path.join(args.output_dir, "error_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
