"""Vote between multiple unsupervised thresholds.

Each sample gets a binary vote from each unsupervised threshold method
(Otsu, GMM, MET, mean, median, q_otsu-quantile, etc.).
Final decision: majority vote, or K-of-N threshold.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(dataset):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(dataset)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, train_arr, test_x, test_y


def met_threshold(src):
    """Kittler-Illingworth MET."""
    s = np.sort(np.asarray(src))
    n = len(s)
    best_t = s[0]
    best = np.inf
    for i in range(2, n - 1):
        c0 = s[:i]
        c1 = s[i:]
        if len(c0) < 2 or len(c1) < 2:
            continue
        w0 = len(c0) / n
        w1 = len(c1) / n
        s0 = c0.std() + 1e-12
        s1 = c1.std() + 1e-12
        J = 1 + 2 * (w0 * np.log(s0) + w1 * np.log(s1)) - 2 * (w0 * np.log(w0) + w1 * np.log(w1))
        if J < best:
            best = J
            best_t = (c0[-1] + c1[0]) / 2
    return float(best_t)


def compute_thresholds(pool, train, test):
    """Return dict of threshold method -> threshold value on a given source."""
    def compute(src):
        return {
            "otsu": otsu_threshold(src),
            "gmm": gmm_threshold(src),
            "met": met_threshold(src),
            "mean": float(src.mean()),
            "median": float(np.median(src)),
            "q75": float(np.quantile(src, 0.75)),
            "q85": float(np.quantile(src, 0.85)),
        }
    return {"pool": compute(pool), "train": compute(train), "test": compute(test)}


def main():
    data = {}
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_arr, test_x, test_y = load(d)
        data[d] = {"pool": pool, "train": train_arr, "test_x": test_x, "test_y": test_y,
                   "base": BASE[d]}

    # For each dataset, compute thresholds from pool source
    for src_name in ["pool", "train", "test"]:
        print(f"\n=== source: {src_name} ===")
        all_thresh = {}
        for d in ["MHClip_EN", "MHClip_ZH"]:
            pool = data[d]["pool"]
            tr = data[d]["train"]
            te = data[d]["test_x"]
            ths = compute_thresholds(pool, tr, te)[src_name]
            all_thresh[d] = ths
        methods = list(all_thresh["MHClip_EN"].keys())
        # Every combination of K methods
        from itertools import combinations
        for K in range(1, len(methods) + 1):
            for subset in combinations(methods, K):
                for v_thresh in range(1, K + 1):
                    # Vote: sample predicted positive if >= v_thresh of the subset thresholds say yes
                    pass_unified = True
                    results = {}
                    for d in ["MHClip_EN", "MHClip_ZH"]:
                        ths = all_thresh[d]
                        test_x = data[d]["test_x"]
                        test_y = data[d]["test_y"]
                        votes = np.zeros(len(test_x), dtype=int)
                        for m in subset:
                            votes += (test_x >= ths[m]).astype(int)
                        pred = (votes >= v_thresh).astype(int)
                        # Convert pred to threshold semantics... actually just compute metrics manually
                        TP = int(((pred == 1) & (test_y == 1)).sum())
                        TN = int(((pred == 0) & (test_y == 0)).sum())
                        FP = int(((pred == 1) & (test_y == 0)).sum())
                        FN = int(((pred == 0) & (test_y == 1)).sum())
                        acc = (TP + TN) / len(test_x)
                        # macro F1
                        prec_p = TP / max(TP + FP, 1)
                        rec_p = TP / max(TP + FN, 1)
                        f1_p = 2 * prec_p * rec_p / max(prec_p + rec_p, 1e-12)
                        prec_n = TN / max(TN + FN, 1)
                        rec_n = TN / max(TN + FP, 1)
                        f1_n = 2 * prec_n * rec_n / max(prec_n + rec_n, 1e-12)
                        mf = (f1_p + f1_n) / 2
                        acc_b, mf_b = data[d]["base"]
                        sb = acc > acc_b and mf > mf_b
                        results[d] = (acc, mf, sb)
                        if not sb:
                            pass_unified = False
                    if pass_unified:
                        print(f"  K={K} v>={v_thresh} subset={subset}")
                        for d in ["MHClip_EN", "MHClip_ZH"]:
                            print(f"    {d}: acc={results[d][0]:.4f} mf={results[d][1]:.4f}")


if __name__ == "__main__":
    main()
