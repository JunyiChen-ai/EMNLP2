"""Full atom-level oracle ceiling for EN: does ANY subset of 32 atoms
strict-beat the EN baseline (acc>0.7640 AND mf>0.6532)?

Uses DP over atoms: state = (pos_selected, neg_selected) across selected
atoms. Since total_pos=49 and total_neg=112, state space is ~49*112=5488.
Full enumeration in O(K * P * N) time.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations

for d in ["MHClip_EN", "MHClip_ZH"]:
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)

    acc_b = 0.7639751552795031 if d == "MHClip_EN" else 0.8120805369127517
    mf_b = 0.6531746031746032 if d == "MHClip_EN" else 0.7871428571428571

    n_te = len(test_x)
    print(f"\n=== {d} === n_te={n_te}, baseline {acc_b:.4f}/{mf_b:.4f}")

    rounded_te = np.round(test_x, 4)
    atoms_vals = np.array(sorted(set(rounded_te)))
    K = len(atoms_vals)

    pos = np.array([int(((rounded_te == v) & (test_y == 1)).sum()) for v in atoms_vals])
    neg = np.array([int(((rounded_te == v) & (test_y == 0)).sum()) for v in atoms_vals])
    total_pos = int(test_y.sum())
    total_neg = n_te - total_pos

    # DP: dp[p][n] = True if there's a subset with exactly (p pos, n neg) selected
    dp = np.zeros((total_pos + 1, total_neg + 1), dtype=bool)
    dp[0, 0] = True
    for i in range(K):
        pi, ni = pos[i], neg[i]
        # Iterate in reverse to avoid double-counting
        for p in range(total_pos, pi - 1, -1):
            for n in range(total_neg, ni - 1, -1):
                if dp[p - pi, n - ni]:
                    dp[p, n] = True

    # Enumerate achievable (tp, fp), compute metrics, count strict-beats
    passes = 0
    best = (0, 0, None)
    for tp in range(total_pos + 1):
        for fp in range(total_neg + 1):
            if not dp[tp, fp]:
                continue
            fn = total_pos - tp
            tn = total_neg - fp
            acc = (tp + tn) / n_te
            pp_pred = tp + fp
            pn_pred = tn + fn
            if pp_pred == 0 or pn_pred == 0:
                continue
            prec_p = tp / pp_pred
            rec_p = tp / total_pos if total_pos else 0
            f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0
            prec_n = tn / pn_pred
            rec_n = tn / total_neg if total_neg else 0
            f1_n = 2 * prec_n * rec_n / (prec_n + rec_n) if (prec_n + rec_n) > 0 else 0
            mf = (f1_p + f1_n) / 2
            if acc > acc_b and mf > mf_b:
                passes += 1
                if (acc, mf) > best[:2]:
                    best = (acc, mf, (tp, fp))

    print(f"  # (tp,fp) pairs that beat strict-both: {passes}")
    print(f"  Best achievable: acc={best[0]:.4f} mf={best[1]:.4f} at {best[2]}")
    # Also show the ceiling regardless of strict-both
    ceil = (0, 0)
    ceil_at = None
    for tp in range(total_pos + 1):
        for fp in range(total_neg + 1):
            if not dp[tp, fp]:
                continue
            fn = total_pos - tp
            tn = total_neg - fp
            acc = (tp + tn) / n_te
            if acc > ceil[0]:
                ceil = (acc, 0)
                ceil_at = (tp, fp)
    print(f"  Max achievable acc: {ceil[0]:.4f} at (tp,fp)={ceil_at}")
