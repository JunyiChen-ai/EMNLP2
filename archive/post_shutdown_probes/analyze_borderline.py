"""Analyze score distribution by class (Hateful/Offensive/Normal).

Questions:
1. Where do each class's scores concentrate?
2. For borderline samples (middle score bins), what class dominates?
3. What does the confusion look like at different score ranges?
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_annotations


def load_scores(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vid = r.get("video_id")
            s = r.get("score")
            if vid and s is not None:
                out[vid] = float(s)
    return out


def analyze(dataset, score_path, label):
    print(f"\n=== {dataset} — {label} ===")
    scores_dict = load_scores(score_path)
    ann = load_annotations(dataset)

    # Build (score, class) pairs
    data = []
    for vid, s in scores_dict.items():
        if vid not in ann:
            continue
        cls = ann[vid]["label"]
        data.append((s, cls))

    data.sort()
    print(f"  N={len(data)}")
    cls_counts = {}
    for _, c in data:
        cls_counts[c] = cls_counts.get(c, 0) + 1
    print(f"  Class counts: {cls_counts}")

    # Per-class score stats
    for c in ["Hateful", "Offensive", "Normal"]:
        cs = [s for s, cl in data if cl == c]
        if cs:
            cs = np.array(cs)
            print(f"  {c:<10} n={len(cs):>3}  mean={cs.mean():.4f}  median={np.median(cs):.4f}  std={cs.std():.4f}  min={cs.min():.4f}  max={cs.max():.4f}")

    # Score bins: what class composition?
    bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    print(f"\n  Score range → Class composition:")
    print(f"  {'Range':<12} {'Total':<6} {'Hate':<6} {'Offn':<6} {'Norm':<6} {'H+O%':<8}")
    for lo, hi in bins:
        bin_data = [c for s, c in data if lo <= s < hi]
        if not bin_data:
            continue
        h = sum(1 for c in bin_data if c == "Hateful")
        o = sum(1 for c in bin_data if c == "Offensive")
        n = sum(1 for c in bin_data if c == "Normal")
        total = len(bin_data)
        pos_pct = 100 * (h + o) / total if total > 0 else 0
        print(f"  [{lo:.2f},{hi:.2f}) {total:<6} {h:<6} {o:<6} {n:<6} {pos_pct:<7.1f}%")

    # Borderline identification: score in [0.15, 0.5] (the "uncertainty" region)
    borderline_low, borderline_high = 0.15, 0.5
    border = [c for s, c in data if borderline_low <= s < borderline_high]
    if border:
        h = sum(1 for c in border if c == "Hateful")
        o = sum(1 for c in border if c == "Offensive")
        n = sum(1 for c in border if c == "Normal")
        total = len(border)
        print(f"\n  Borderline region [{borderline_low}, {borderline_high}): N={total}")
        print(f"    Hateful: {h} ({100*h/total:.1f}%)")
        print(f"    Offensive: {o} ({100*o/total:.1f}%)")
        print(f"    Normal: {n} ({100*n/total:.1f}%)")
        if (h + o) > n:
            print(f"    → borderline samples are MORE positive than negative")
        else:
            print(f"    → borderline samples are MORE negative than positive")


def main():
    # 2B Binary (validated)
    print("### 2B BINARY SCORES (EN Raw+Otsu 76.40%, ZH Raw+GMM 77.85%)")
    analyze("MHClip_EN",
            "/data/jehc223/EMNLP2/results/holistic_2b/MHClip_EN/test_binary.jsonl.backup",
            "Binary 2B")
    analyze("MHClip_ZH",
            "/data/jehc223/EMNLP2/results/holistic_2b/MHClip_ZH/test_binary.jsonl.backup",
            "Binary 2B")

    # 2B Triclass
    print("\n\n### 2B TRICLASS SCORES (H/O/N → score = p_h + p_o)")
    analyze("MHClip_EN",
            "/data/jehc223/EMNLP2/results/holistic_2b/MHClip_EN/test_triclass.jsonl",
            "Triclass 2B")
    analyze("MHClip_ZH",
            "/data/jehc223/EMNLP2/results/holistic_2b/MHClip_ZH/test_triclass.jsonl",
            "Triclass 2B")


if __name__ == "__main__":
    main()
