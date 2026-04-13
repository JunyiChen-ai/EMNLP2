"""Re-verify current baselines on the 4 allowed score files and dump pool/atom structure."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations


OUT = "/data/jehc223/EMNLP2/results/meta_selector/diag_baselines.json"


def load_pool(dataset):
    base = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base}/train_binary.jsonl")
    test = load_scores_file(f"{base}/test_binary.jsonl")
    ann = load_annotations(dataset)
    test_x, test_y = build_arrays(test, ann)
    # Pool: all train values + all test values (train unlabeled-ok, test labels only used at final metric)
    pool = np.array(list(train.values()) + list(test.values()), dtype=float)
    train_x = np.array(list(train.values()), dtype=float)
    return pool, train_x, test_x, test_y


def atom_structure(arr):
    u = np.array(sorted(set(np.round(arr.astype(float), 8).tolist())))
    counts = {float(v): int(np.sum(np.round(arr, 8) == v)) for v in u}
    return u, counts


def main():
    out = {}
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_x, test_x, test_y = load_pool(dataset)

        tf_otsu_t = otsu_threshold(test_x)
        tf_gmm_t = gmm_threshold(test_x)
        tr_otsu_t = otsu_threshold(train_x)
        tr_gmm_t = gmm_threshold(train_x)
        pool_otsu_t = otsu_threshold(pool)
        pool_gmm_t = gmm_threshold(pool)

        u_pool, counts_pool = atom_structure(pool)
        u_train, _ = atom_structure(train_x)
        u_test, _ = atom_structure(test_x)

        out[dataset] = {
            "n_train": int(len(train_x)),
            "n_test": int(len(test_x)),
            "n_test_pos": int(test_y.sum()),
            "pool_uniques": len(u_pool),
            "train_uniques": len(u_train),
            "test_uniques": len(u_test),
            "tf_otsu": {"t": tf_otsu_t, **metrics(test_x, test_y, tf_otsu_t)},
            "tf_gmm": {"t": tf_gmm_t, **metrics(test_x, test_y, tf_gmm_t)},
            "tr_otsu": {"t": tr_otsu_t, **metrics(test_x, test_y, tr_otsu_t)},
            "tr_gmm": {"t": tr_gmm_t, **metrics(test_x, test_y, tr_gmm_t)},
            "pool_otsu": {"t": pool_otsu_t, **metrics(test_x, test_y, pool_otsu_t)},
            "pool_gmm": {"t": pool_gmm_t, **metrics(test_x, test_y, pool_gmm_t)},
            "top_atoms": sorted(counts_pool.items(), key=lambda kv: -kv[1])[:12],
        }

        print(f"\n=== {dataset} ===")
        print(f"  n_train={len(train_x)}  n_test={len(test_x)}  n_test_pos={int(test_y.sum())}")
        print(f"  pool uniques={len(u_pool)}  train uniques={len(u_train)}  test uniques={len(u_test)}")
        for k in ["tf_otsu","tf_gmm","tr_otsu","tr_gmm","pool_otsu","pool_gmm"]:
            m = out[dataset][k]
            print(f"  {k:>10}  t={m['t']:.6f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}")
        print(f"  top atoms: {out[dataset]['top_atoms'][:6]}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
