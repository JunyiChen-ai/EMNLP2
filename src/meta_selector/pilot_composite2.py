"""Sample-level composite + frozen Otsu.

For each sample, composite(s) = transform(s, pool_features).
Pool-composite-otsu threshold applied to test-composite.
Try many dataset-agnostic transforms.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7640, 0.6532), "MHClip_ZH": (0.8121, 0.7871)}


def load(dataset):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(dataset)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    test_arr = np.array(list(test.values()), dtype=float)
    pool = np.concatenate([train_arr, test_arr])
    return pool, train_arr, test_x, test_y


def nbhd_count(x, pool, window):
    return np.array([int(((pool != xi) & (np.abs(pool - xi) < window)).sum()) for xi in x])


def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        pool, train_arr, test_x, test_y = load(dataset)
        n = len(pool)
        # Use relative window (2% of score range)
        window = 0.05
        nbhd_pool = nbhd_count(pool, pool, window)
        nbhd_test = nbhd_count(test_x, pool, window)

        print(f"\n=== {dataset} ===  baseline {acc_b:.4f}/{mf_b:.4f}  pool={len(pool)}")

        for name, fn in [
            ("raw", lambda x, nb: x),
            ("x - 0.001*nb", lambda x, nb: x - 0.001*nb),
            ("x - 0.005*nb", lambda x, nb: x - 0.005*nb),
            ("x - 0.01*nb", lambda x, nb: x - 0.01*nb),
            ("x - 0.02*nb", lambda x, nb: x - 0.02*nb),
            ("x - 0.005*sqrt(nb)", lambda x, nb: x - 0.005*np.sqrt(nb)),
            ("x - 0.01*sqrt(nb)", lambda x, nb: x - 0.01*np.sqrt(nb)),
            ("x - 0.05*log1p(nb)", lambda x, nb: x - 0.05*np.log1p(nb)),
            ("x - 0.1*log1p(nb)", lambda x, nb: x - 0.1*np.log1p(nb)),
            ("x - 0.15*log1p(nb)", lambda x, nb: x - 0.15*np.log1p(nb)),
            ("x - 0.2*log1p(nb)", lambda x, nb: x - 0.2*np.log1p(nb)),
            ("x * exp(-0.01*nb)", lambda x, nb: x * np.exp(-0.01*nb)),
            ("x * exp(-0.02*nb)", lambda x, nb: x * np.exp(-0.02*nb)),
            ("x * exp(-0.05*nb)", lambda x, nb: x * np.exp(-0.05*nb)),
            ("x + 0.2/(1+nb)", lambda x, nb: x + 0.2/(1+nb)),
            ("x + 0.5/(1+nb)", lambda x, nb: x + 0.5/(1+nb)),
            ("x + 1.0/(1+nb)", lambda x, nb: x + 1.0/(1+nb)),
        ]:
            c_pool = fn(pool, nbhd_pool)
            c_test = fn(test_x, nbhd_test)
            t = otsu_threshold(c_pool)
            m = metrics(c_test, test_y, t)
            flag = ""
            if m["acc"] > acc_b and m["mf"] > mf_b: flag = " ***STRICT-BOTH***"
            elif m["acc"] > acc_b: flag = " [acc+]"
            elif m["mf"] > mf_b: flag = " [mf+]"
            print(f"  {name:<28}  t={t:+.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}{flag}")


if __name__ == "__main__":
    main()
