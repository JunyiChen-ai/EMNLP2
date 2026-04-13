"""Test: within each score cluster (rounded at 4 decimals), do higher sub-FP
values correlate with positive labels?

If yes: sub-FP ordering is a real label-free signal.
If no: my MAD rule's strict-beat is just luck on 2 clusters.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from collections import defaultdict
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations

for d in ["MHClip_EN", "MHClip_ZH"]:
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)

    # Group test samples by 4-decimal atom
    clusters = defaultdict(list)
    for x, y in zip(test_x, test_y):
        a = round(float(x), 4)
        clusters[a].append((float(x), int(y)))

    print(f"\n=== {d} ===")
    # For each cluster with mixed labels, measure rank correlation
    # between sub-FP value and label
    tot_agree = 0
    tot_pairs = 0
    n_mixed_cluster = 0
    for a, items in sorted(clusters.items()):
        if len(items) < 2:
            continue
        labels = [y for _, y in items]
        if len(set(labels)) < 2:
            continue  # not mixed
        n_mixed_cluster += 1
        items.sort(key=lambda p: p[0])
        # count concordant pairs: higher x => higher y
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                xi, yi = items[i]
                xj, yj = items[j]
                if yi != yj:
                    tot_pairs += 1
                    # concordant if higher x has higher label
                    if (yj - yi) > 0:  # xj > xi, yj > yi => concordant
                        tot_agree += 1
    if tot_pairs > 0:
        print(f"  n_mixed_clusters={n_mixed_cluster}")
        print(f"  concordant sub-FP pairs: {tot_agree}/{tot_pairs} = {tot_agree/tot_pairs:.3f}")
        print(f"  (0.5 = random; higher = sub-FP ordering carries label signal)")

    # Also: for each mixed cluster, what's the best within-cluster split?
    print(f"  Per-cluster within-split analysis:")
    for a, items in sorted(clusters.items()):
        if len(set(y for _, y in items)) < 2:
            continue
        items.sort(key=lambda p: p[0])
        labels = [y for _, y in items]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        # best split: flag top-k samples; find k that maximizes (TP - FP)
        best_k = 0
        best_gain = 0
        for k in range(len(labels)+1):
            flagged = labels[-k:] if k > 0 else []
            tp = sum(flagged)
            fp = k - tp
            gain = tp - fp
            if gain > best_gain:
                best_gain = gain
                best_k = k
        print(f"    atom={a:.4f}  n={len(labels)}  pos={n_pos}  neg={n_neg}  best_top_k={best_k}  gain={best_gain}")
