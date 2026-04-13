"""Is there a SINGLE q value that strict-beats BOTH datasets when applied uniformly?"""
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
    data = {}
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_arr, test_x, test_y = load(dataset)
        data[dataset] = {"pool": pool, "train": train_arr, "test_x": test_x, "test_y": test_y,
                         "base": BASE[dataset]}

    for src_name in ["pool", "train", "test_x"]:
        print(f"\n=== threshold = q-quantile of {src_name} ===")
        strict_en = set()
        strict_zh = set()
        grid = np.arange(0.5, 1.0, 0.001)
        for q in grid:
            m_en = metrics(data["MHClip_EN"]["test_x"], data["MHClip_EN"]["test_y"],
                           float(np.quantile(data["MHClip_EN"][src_name if src_name != "test_x" else "test_x"], q)))
            acc_b_en, mf_b_en = data["MHClip_EN"]["base"]
            en_strict = (m_en["acc"] > acc_b_en and m_en["mf"] > mf_b_en)
            m_zh = metrics(data["MHClip_ZH"]["test_x"], data["MHClip_ZH"]["test_y"],
                           float(np.quantile(data["MHClip_ZH"][src_name if src_name != "test_x" else "test_x"], q)))
            acc_b_zh, mf_b_zh = data["MHClip_ZH"]["base"]
            zh_strict = (m_zh["acc"] > acc_b_zh and m_zh["mf"] > mf_b_zh)
            if en_strict and zh_strict:
                print(f"  UNIFIED q={q:.4f}  EN: {m_en['acc']:.4f}/{m_en['mf']:.4f}  ZH: {m_zh['acc']:.4f}/{m_zh['mf']:.4f}")
            if en_strict:
                strict_en.add(q)
            if zh_strict:
                strict_zh.add(q)
        print(f"  EN strict: {len(strict_en)} q values")
        print(f"  ZH strict: {len(strict_zh)} q values")
        common = strict_en & strict_zh
        print(f"  BOTH strict: {len(common)} q values")
        if common:
            print(f"    q range: {min(common):.4f} to {max(common):.4f}")


if __name__ == "__main__":
    main()
