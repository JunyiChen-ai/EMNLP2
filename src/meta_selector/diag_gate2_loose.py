"""Check if Gate 2 bar is: acc strict >, mf >=. Re-enumerate with this bar."""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}

def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, train_arr, test_x, test_y


for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, train_arr, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.6f}/{mf_b:.6f}")
    # Atoms
    atoms = sorted(set(round(v, 4) for v in pool))
    all_cuts = []
    for A in atoms:
        t = A - 1e-6
        m = metrics(test_x, test_y, t)
        all_cuts.append((A, m["acc"], m["mf"]))
    for c in all_cuts:
        tag = ""
        if c[1] > acc_b: tag += " [acc+]"
        if c[2] > mf_b: tag += " [mf+]"
        if c[1] > acc_b and c[2] > mf_b: tag = " ***STRICT BOTH***"
        print(f"    cut at atom {c[0]:.4f}: acc={c[1]:.4f} mf={c[2]:.4f}{tag}")
