"""Where do the strict-both atoms sit in the unlabeled pool? Rank, percentile, neighborhood."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

TARGET_T = {"MHClip_EN": 0.32082128, "MHClip_ZH": 0.02931223}
BASE = {"MHClip_EN": (0.7640, 0.6532), "MHClip_ZH": (0.8121, 0.7871)}


def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
        train = load_scores_file(f"{base_d}/train_binary.jsonl")
        test = load_scores_file(f"{base_d}/test_binary.jsonl")
        ann = load_annotations(dataset)
        test_x, test_y = build_arrays(test, ann)
        train_arr = np.array(list(train.values()), dtype=float)
        test_arr = np.array(list(test.values()), dtype=float)
        pool = np.concatenate([train_arr, test_arr])

        t_star = TARGET_T[dataset]
        pool_u = sorted(set(round(float(s), 8) for s in pool))
        # Atoms with at least one sample: rank of t_star atom
        rank = None
        for i, a in enumerate(pool_u):
            if abs(a - t_star) < 1e-5:
                rank = i
                break
        n_atoms = len(pool_u)
        pool_below = int(np.sum(pool < t_star))
        pool_equal_or_above = int(np.sum(pool >= t_star))
        train_below = int(np.sum(train_arr < t_star))
        train_equal_or_above = int(np.sum(train_arr >= t_star))
        test_below = int(np.sum(test_arr < t_star))
        test_equal_or_above = int(np.sum(test_arr >= t_star))

        print(f"\n=== {dataset} ===")
        print(f"  t* = {t_star:.8f}")
        print(f"  atom rank within pool uniques = {rank}/{n_atoms-1}  (={rank/max(1,n_atoms-1):.3f})")
        print(f"  pool: below {pool_below}/{len(pool)}={pool_below/len(pool):.3f}; above-or-eq {pool_equal_or_above}")
        print(f"  train: below {train_below}/{len(train_arr)}={train_below/len(train_arr):.3f}; above-or-eq {train_equal_or_above}")
        print(f"  test: below {test_below}/{len(test_arr)}={test_below/len(test_arr):.3f}; above-or-eq {test_equal_or_above}")

        # Neighborhood atoms with sample counts
        print(f"  pool atoms around t*:")
        for a in pool_u:
            if abs(a - t_star) < 0.02:
                cnt = int(np.sum(np.round(pool, 8) == a))
                print(f"    atom={a:.8f}  pool_count={cnt}")


if __name__ == "__main__":
    main()
