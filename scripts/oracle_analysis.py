"""
Oracle analysis: Is the easy-to-hard cascade hypothesis valid?

Small model: MLP on BERT text embeddings (Title + Transcript, NOT MLLM rationale)
Large model: MLLM zero-shot judgment (from generic_response.overall_judgment)

Questions:
1. What % of small-model errors does the large model get right? (complementarity)
2. What % of large-model errors does the small model get right?
3. Oracle upper bound: if we could perfectly route, what's the ceiling?
4. What do "hard for both" cases look like?
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


DATASET_PATHS = {
    "HateMM": "datasets/HateMM",
    "MHClip_EN": "datasets/Multihateclip/English",
    "MHClip_ZH": "datasets/Multihateclip/Chinese",
}


def load_data(dataset_name="HateMM"):
    base = DATASET_PATHS[dataset_name]
    with open(f"{base}/generic_data.json") as f:
        data = json.load(f)

    with open(f"{base}/splits/train.csv") as f:
        train_ids = [l.strip() for l in f if l.strip()]
    with open(f"{base}/splits/test.csv") as f:
        test_ids = [l.strip() for l in f if l.strip()]

    label_map = {"Hate": 1, "Non Hate": 0, "Hateful": 1, "Offensive": 1, "Normal": 0}

    id2sample = {s["Video_ID"]: s for s in data}
    return id2sample, train_ids, test_ids, label_map


def get_text(sample):
    """Raw text: Title + Transcript (NOT MLLM rationale)."""
    title = sample.get("Title", "") or ""
    transcript = sample.get("Transcript", "") or ""
    text = f"{title} {transcript}".strip()
    if not text:
        text = "[empty]"
    return text


def get_mllm_pred(sample):
    """Extract MLLM zero-shot prediction from generic_response."""
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
    return -1  # unclear


class TextDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_small_model(train_embs, train_labels, test_embs, test_labels, n_seeds=20):
    """Train simple MLP on text embeddings, return per-sample predictions across seeds."""
    dim = train_embs.shape[1]
    all_test_preds = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        train_ds = TextDataset(train_embs, train_labels)
        loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(50):
            for X, y in loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            test_X = torch.tensor(test_embs, dtype=torch.float32)
            logits = model(test_X)
            preds = logits.argmax(dim=1).numpy()
        all_test_preds.append(preds)

    return np.array(all_test_preds)  # [n_seeds, n_test]


def run_one_dataset(dataset_name):
    print(f"\n{'=' * 60}")
    print(f"Oracle Analysis: {dataset_name}")
    print(f"{'=' * 60}")

    id2sample, train_ids, test_ids, label_map = load_data(dataset_name)

    # Encode text with sentence-transformers (NOT MLLM rationale)
    print("\nEncoding text with sentence-transformers...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    train_texts, train_labels = [], []
    for vid in train_ids:
        if vid not in id2sample:
            continue
        s = id2sample[vid]
        gt = label_map.get(s["Label"], -1)
        if gt == -1:
            continue
        train_texts.append(get_text(s))
        train_labels.append(gt)

    test_texts, test_labels, test_vids, test_mllm_preds = [], [], [], []
    for vid in test_ids:
        if vid not in id2sample:
            continue
        s = id2sample[vid]
        gt = label_map.get(s["Label"], -1)
        if gt == -1:
            continue
        mllm_pred = get_mllm_pred(s)
        test_texts.append(get_text(s))
        test_labels.append(gt)
        test_vids.append(vid)
        test_mllm_preds.append(mllm_pred)

    train_embs = encoder.encode(train_texts, show_progress_bar=False)
    test_embs = encoder.encode(test_texts, show_progress_bar=False)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    test_mllm_preds = np.array(test_mllm_preds)

    print(f"Train: {len(train_labels)}, Test: {len(test_labels)}")

    # Filter out MLLM unclear predictions
    valid_mask = test_mllm_preds != -1
    print(f"MLLM valid predictions: {valid_mask.sum()}/{len(valid_mask)}")

    # Train small model (20 seeds)
    print("\nTraining small model (MLP on BERT text embeddings, 20 seeds)...")
    all_preds = train_small_model(train_embs, train_labels, test_embs, test_labels, n_seeds=20)

    # Use majority vote across seeds for stable small-model prediction
    majority_preds = np.round(all_preds.mean(axis=0)).astype(int)

    # Also compute per-seed variance as confidence proxy
    pred_mean = all_preds.mean(axis=0)  # fraction of seeds predicting 1
    small_confidence = np.abs(pred_mean - 0.5) * 2  # 0 = maximally uncertain, 1 = all seeds agree

    # === Analysis ===
    mask = valid_mask
    gt = test_labels[mask]
    small_pred = majority_preds[mask]
    mllm_pred = test_mllm_preds[mask]
    conf = small_confidence[mask]
    vids = np.array(test_vids)[mask]

    small_correct = (small_pred == gt)
    mllm_correct = (mllm_pred == gt)

    print(f"\n{'=' * 60}")
    print(f"RESULTS (n={len(gt)})")
    print(f"{'=' * 60}")

    small_acc = accuracy_score(gt, small_pred)
    small_f1 = f1_score(gt, small_pred, average="macro")
    mllm_acc = accuracy_score(gt, mllm_pred)
    mllm_f1 = f1_score(gt, mllm_pred, average="macro")

    print(f"\nSmall model (MLP on text): Acc={small_acc:.3f}, F1={small_f1:.3f}")
    print(f"MLLM zero-shot:           Acc={mllm_acc:.3f}, F1={mllm_f1:.3f}")

    # Complementarity
    small_wrong = ~small_correct
    mllm_wrong = ~mllm_correct

    both_correct = (small_correct & mllm_correct).sum()
    small_only_correct = (small_correct & mllm_wrong).sum()
    mllm_only_correct = (mllm_correct & small_wrong).sum()
    both_wrong = (small_wrong & mllm_wrong).sum()

    print(f"\n--- Complementarity Matrix ---")
    print(f"                    MLLM correct  MLLM wrong")
    print(f"Small correct       {both_correct:>8}       {small_only_correct:>8}")
    print(f"Small wrong         {mllm_only_correct:>8}       {both_wrong:>8}")

    print(f"\n--- Key Numbers ---")
    n_small_errors = small_wrong.sum()
    n_mllm_errors = mllm_wrong.sum()
    print(f"Small model errors: {n_small_errors}")
    print(f"  → MLLM gets right: {mllm_only_correct} ({mllm_only_correct/max(n_small_errors,1)*100:.1f}%)")
    print(f"  → Both wrong:      {both_wrong} ({both_wrong/max(n_small_errors,1)*100:.1f}%)")
    print(f"MLLM errors: {n_mllm_errors}")
    print(f"  → Small gets right: {small_only_correct} ({small_only_correct/max(n_mllm_errors,1)*100:.1f}%)")
    print(f"  → Both wrong:       {both_wrong} ({both_wrong/max(n_mllm_errors,1)*100:.1f}%)")

    # Oracle
    oracle_correct = (small_correct | mllm_correct).sum()
    oracle_acc = oracle_correct / len(gt)
    print(f"\nOracle (perfect routing): Acc={oracle_acc:.3f} ({oracle_correct}/{len(gt)})")
    print(f"Oracle ceiling lift over small: +{(oracle_acc - small_acc)*100:.1f}pp")
    print(f"Oracle ceiling lift over MLLM:  +{(oracle_acc - mllm_acc)*100:.1f}pp")

    # Confidence analysis
    print(f"\n--- Confidence vs Correctness ---")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        high_conf = conf >= threshold
        low_conf = conf < threshold
        if high_conf.sum() > 0 and low_conf.sum() > 0:
            high_acc = (small_pred[high_conf] == gt[high_conf]).mean()
            low_acc = (small_pred[low_conf] == gt[low_conf]).mean()
            mllm_on_low = (mllm_pred[low_conf] == gt[low_conf]).mean()
            print(f"  conf>={threshold:.1f}: n={high_conf.sum():>4}, small_acc={high_acc:.3f}")
            print(f"  conf< {threshold:.1f}: n={low_conf.sum():>4}, small_acc={low_acc:.3f}, mllm_acc={mllm_on_low:.3f}")

    # "Both wrong" analysis
    print(f"\n--- Both Wrong Cases ({both_wrong} samples) ---")
    both_wrong_mask = small_wrong & mllm_wrong
    for i, (vid, is_bw) in enumerate(zip(vids, both_wrong_mask)):
        if not is_bw:
            continue
        s = id2sample[vid]
        resp = s.get("generic_response", {})
        print(f"\n[{vid}] GT={s['Label']}, Small={small_pred[list(vids).index(vid)]}, MLLM={mllm_pred[list(vids).index(vid)]}")
        print(f"  Title: {s.get('Title', '')[:80]}")
        print(f"  Transcript: {s.get('Transcript', '')[:120]}")
        if isinstance(resp, dict):
            print(f"  MLLM judgment: {resp.get('overall_judgment', '')[:150]}")


def main():
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["HateMM", "MHClip_EN", "MHClip_ZH"]
    for ds in datasets:
        run_one_dataset(ds)


if __name__ == "__main__":
    main()
