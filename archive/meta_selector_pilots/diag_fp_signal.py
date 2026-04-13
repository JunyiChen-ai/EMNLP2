"""Test whether sub-atom FP-drift ordering correlates with labels.

For each atom (4-decimal), look at test samples within the atom and test if their
raw (unrounded) scores correlate with labels. If HIGHER FP-bits indicate HIGHER
positive probability, we can use the 8-decimal scores directly and a clean
quantile or threshold rule.
"""
import os, sys, json, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays, otsu_threshold, gmm_threshold
from data_utils import load_annotations


def load(dataset):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(dataset)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, train_arr, test_x, test_y


for d in ["MHClip_EN", "MHClip_ZH"]:
    print(f"\n=== {d} ===")
    pool, train_arr, test_x, test_y = load(d)
    # Group test by 4-decimal atom
    atoms = {}
    for x, y in zip(test_x, test_y):
        k = round(float(x), 4)
        atoms.setdefault(k, []).append((float(x), int(y)))
    # For each atom with mixed labels, check if sub-atom score order correlates
    signal = []
    for k, entries in sorted(atoms.items()):
        if len(entries) < 2:
            continue
        ys = [y for _, y in entries]
        if sum(ys) == 0 or sum(ys) == len(ys):
            continue  # pure atom
        # Sort entries by raw score
        entries.sort(key=lambda r: r[0])
        # Spearman-ish: does rank correlate with label?
        xs = np.array([e[0] for e in entries])
        ys = np.array([e[1] for e in entries])
        # check if pos samples have higher average sub-atom offset
        mean_xs_pos = xs[ys == 1].mean()
        mean_xs_neg = xs[ys == 0].mean()
        diff = mean_xs_pos - mean_xs_neg
        print(f"  atom {k:.4f}: n={len(entries)}, pos_mean-neg_mean sub-atom = {diff:+.2e}")
        signal.append(diff)
    print(f"  atoms with mixed labels: {len(signal)}")
    print(f"  frac with diff > 0: {np.mean([s > 0 for s in signal]):.3f}")
    print(f"  mean diff: {np.mean(signal):+.2e}")

    # A different question: if we use RAW sub-atom scores (no rounding) and apply
    # a standard Otsu threshold, do we get different behavior?
    ot_raw = otsu_threshold(test_x)
    gm_raw = gmm_threshold(test_x)
    print(f"  test-Otsu raw t={ot_raw:.10f}")
    print(f"  test-GMM raw t={gm_raw:.10f}")
