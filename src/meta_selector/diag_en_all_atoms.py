"""List all EN atoms with pos/neg counts — verify net-majority direction.
Goal: check if ANY base=NEG atom is actually net-positive (so an add could help).
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
    print(f"\n=== {d} ===")
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    print(f"  t_base={t_base:.4f}")
    rounded_te = np.round(test_x, 4)
    atoms_vals = np.array(sorted(set(rounded_te)))
    pos = np.array([int(((rounded_te == v) & (test_y == 1)).sum()) for v in atoms_vals])
    neg = np.array([int(((rounded_te == v) & (test_y == 0)).sum()) for v in atoms_vals])
    base = (atoms_vals >= t_base)
    print(f"  K={len(atoms_vals)}")
    print(f"  atom | pos | neg | net | base | net_sign")
    for v, p, n, b in zip(atoms_vals, pos, neg, base):
        net = p - n
        sign = "POS" if net > 0 else ("NEG" if net < 0 else "TIE")
        print(f"  {v:.4f} | {p:3d} | {n:3d} | {net:+d} | {'b+' if b else 'b-':2s} | {sign}")
