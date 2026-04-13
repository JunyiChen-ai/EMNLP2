"""Detailed quantile sweep to understand strict-both regions."""
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


def sweep(dataset, src_name, src, test_x, test_y, acc_b, mf_b):
    strict = []
    best = None
    for q_int in range(100, 10001):  # fine grid q in [0.01, 1.0]
        q = q_int / 10000
        t = float(np.quantile(src, q))
        m = metrics(test_x, test_y, t)
        rec = {"q": q, "t": t, "acc": m["acc"], "mf": m["mf"], "npos": int((test_x >= t).sum())}
        if m["acc"] > acc_b and m["mf"] > mf_b:
            strict.append(rec)
        if best is None or (m["acc"] + m["mf"] > best["acc"] + best["mf"]):
            best = rec

    # Find contiguous strict-both q-ranges
    if strict:
        qs = sorted([r["q"] for r in strict])
        print(f"  {src_name}: strict-both at {len(strict)} distinct q values (range q in [{min(qs):.4f}, {max(qs):.4f}])")
        # Dedup by (acc, mf, t)
        seen = set()
        uniq = []
        for r in sorted(strict, key=lambda r: r["q"]):
            k = (round(r["acc"], 5), round(r["mf"], 5), round(r["t"], 8))
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        print(f"  unique strict-both cells: {len(uniq)}")
        for r in uniq[:15]:
            print(f"    q={r['q']:.4f}  t={r['t']:.10f}  acc={r['acc']:.4f}  mf={r['mf']:.4f}  npos={r['npos']}")
    else:
        print(f"  {src_name}: no strict-both")
    return strict


def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        pool, train_arr, test_x, test_y = load(dataset)
        print(f"\n=== {dataset} === baseline {acc_b:.6f}/{mf_b:.6f}")
        for src_name, src in [("pool", pool), ("train", train_arr), ("test", test_x)]:
            sweep(dataset, src_name, src, test_x, test_y, acc_b, mf_b)


if __name__ == "__main__":
    main()
