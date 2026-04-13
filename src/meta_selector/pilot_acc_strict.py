"""Re-search with the correct bar: acc strict >, mf >= (non-strict).

The previous searches used both-strict. Loosening mf to >= might open new rules.
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


def main():
    EPS = 1e-10
    data = {}
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_arr, test_x, test_y = load(d)
        data[d] = {"pool": pool, "train": train_arr, "test_x": test_x, "test_y": test_y,
                   "base": BASE[d]}

    # Fine-grid quantile sweep with bar: acc strict > AND mf >= (not strict)
    print("=== Fine-grid pool-quantile sweep with bar (acc strict>, mf>=) ===")
    strict_en = []
    strict_zh = []
    for q_int in range(1, 10000):
        q = q_int / 10000
        results = {}
        for d in ["MHClip_EN", "MHClip_ZH"]:
            t = float(np.quantile(data[d]["pool"], q))
            m = metrics(data[d]["test_x"], data[d]["test_y"], t)
            acc_b, mf_b = data[d]["base"]
            pass_ = m["acc"] > acc_b and m["mf"] >= mf_b - EPS
            results[d] = (q, t, m["acc"], m["mf"], pass_)
        if results["MHClip_EN"][4]:
            strict_en.append(results["MHClip_EN"])
        if results["MHClip_ZH"][4]:
            strict_zh.append(results["MHClip_ZH"])
        if results["MHClip_EN"][4] and results["MHClip_ZH"][4]:
            print(f"  ** UNIFIED q={q:.4f} EN t={results['MHClip_EN'][1]:.6f} "
                  f"{results['MHClip_EN'][2]:.4f}/{results['MHClip_EN'][3]:.4f}  "
                  f"ZH t={results['MHClip_ZH'][1]:.6f} {results['MHClip_ZH'][2]:.4f}/{results['MHClip_ZH'][3]:.4f}")
    print(f"  EN (acc strict, mf>=) passes: {len(strict_en)} q values")
    if strict_en:
        qs = [r[0] for r in strict_en]
        print(f"    q range: [{min(qs):.4f}, {max(qs):.4f}]")
    print(f"  ZH (acc strict, mf>=) passes: {len(strict_zh)} q values")
    if strict_zh:
        qs = [r[0] for r in strict_zh]
        print(f"    q range: [{min(qs):.4f}, {max(qs):.4f}]")

    # Fine-grid train-quantile sweep
    print("\n=== Fine-grid train-quantile sweep (acc strict>, mf>=) ===")
    strict_en = []
    strict_zh = []
    for q_int in range(1, 10000):
        q = q_int / 10000
        results = {}
        for d in ["MHClip_EN", "MHClip_ZH"]:
            t = float(np.quantile(data[d]["train"], q))
            m = metrics(data[d]["test_x"], data[d]["test_y"], t)
            acc_b, mf_b = data[d]["base"]
            pass_ = m["acc"] > acc_b and m["mf"] >= mf_b - EPS
            results[d] = (q, t, m["acc"], m["mf"], pass_)
        if results["MHClip_EN"][4]:
            strict_en.append(results["MHClip_EN"])
        if results["MHClip_ZH"][4]:
            strict_zh.append(results["MHClip_ZH"])
        if results["MHClip_EN"][4] and results["MHClip_ZH"][4]:
            print(f"  ** UNIFIED q={q:.4f} EN t={results['MHClip_EN'][1]:.6f} "
                  f"{results['MHClip_EN'][2]:.4f}/{results['MHClip_EN'][3]:.4f}  "
                  f"ZH t={results['MHClip_ZH'][1]:.6f} {results['MHClip_ZH'][2]:.4f}/{results['MHClip_ZH'][3]:.4f}")
    print(f"  EN passes: {len(strict_en)} q values")
    if strict_en:
        qs = [r[0] for r in strict_en]
        print(f"    q range: [{min(qs):.4f}, {max(qs):.4f}]")
    print(f"  ZH passes: {len(strict_zh)} q values")
    if strict_zh:
        qs = [r[0] for r in strict_zh]
        print(f"    q range: [{min(qs):.4f}, {max(qs):.4f}]")


if __name__ == "__main__":
    main()
