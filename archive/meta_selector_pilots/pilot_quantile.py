"""Quantile thresholds: sweep the positive-fraction q and check metrics.

For each q in (0, 1), threshold at q-th quantile of pool (so fraction 1-q of test >= t).
"""
import os, sys, json, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
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
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        pool, train_arr, test_x, test_y = load(dataset)
        print(f"\n=== {dataset} ===  baseline {acc_b:.4f}/{mf_b:.4f}")
        for src_name, src in [("pool", pool), ("train", train_arr), ("test", test_x)]:
            for q in np.arange(0.50, 1.00, 0.01):
                t = float(np.quantile(src, q))
                m = metrics(test_x, test_y, t)
                flag = ""
                if m["acc"] > acc_b and m["mf"] > mf_b: flag = " ***STRICT-BOTH***"
                elif m["acc"] > acc_b: flag = " [acc+]"
                elif m["mf"] > mf_b: flag = " [mf+]"
                if flag and "STRICT" in flag:
                    print(f"  src={src_name}  q={q:.2f}  t={t:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}{flag}")


if __name__ == "__main__":
    main()
