"""Diagnostic: is the atom-wise positive rate monotone in raw score?

Rounds scores to 4 decimals to form atoms. For each atom, compute:
  - pool_n (pool count)
  - test_n, test_pos, test_neg
  - test positive rate = test_pos / test_n (only for atoms with test_n > 0)

Sort atoms by raw score and check whether test positive rate is monotone-increasing.
If NOT monotone, subset rules strictly dominate suffix rules and are unexplored.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations

def analyze(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)

    train_arr = np.array(list(train.values()), dtype=float)
    test_arr = np.array(list(test.values()), dtype=float)
    pool = np.concatenate([train_arr, test_arr])

    # Atom = 4-decimal rounded value
    from collections import defaultdict
    atom_pool = defaultdict(int)
    for x in pool:
        atom_pool[round(float(x), 4)] += 1

    atom_test = defaultdict(lambda: [0, 0, 0])  # [n, pos, neg]
    for x, y in zip(test_x, test_y):
        a = round(float(x), 4)
        atom_test[a][0] += 1
        if y == 1:
            atom_test[a][1] += 1
        else:
            atom_test[a][2] += 1

    atoms = sorted(set(atom_pool.keys()) | set(atom_test.keys()))
    print(f"\n=== {d} ===")
    print(f"n_atoms = {len(atoms)}, n_pool = {len(pool)}, n_test = {len(test_x)}")
    print(f"{'atom':>10} {'pool_n':>7} {'te_n':>5} {'te_pos':>6} {'te_neg':>6} {'pos_rate':>9}")
    data_for_mono = []
    for a in atoms:
        pn = atom_pool[a]
        tn, tp, tne = atom_test[a]
        rate = tp / tn if tn > 0 else None
        rate_s = f"{rate:.3f}" if rate is not None else "  -  "
        print(f"{a:>10.4f} {pn:>7d} {tn:>5d} {tp:>6d} {tne:>6d} {rate_s:>9}")
        if tn > 0:
            data_for_mono.append((a, rate, tn))

    # Monotonicity check: sort by atom value, check if positive rate is non-decreasing
    print(f"\n  Monotonicity check on atoms with test_n > 0:")
    data_for_mono.sort(key=lambda r: r[0])
    n_viol = 0
    for i in range(1, len(data_for_mono)):
        prev_a, prev_r, _ = data_for_mono[i-1]
        cur_a, cur_r, _ = data_for_mono[i]
        if cur_r < prev_r:
            n_viol += 1
            print(f"    VIOLATION: atom {prev_a:.4f} rate={prev_r:.3f}  >  atom {cur_a:.4f} rate={cur_r:.3f}")
    print(f"  Total monotonicity violations: {n_viol} / {len(data_for_mono)-1} transitions")

    # Print atom rank by positive rate (for subset-rule design)
    print(f"\n  Atoms sorted by positive rate (desc):")
    ranked = sorted(data_for_mono, key=lambda r: (-r[1], -r[0]))
    for a, r, n in ranked[:15]:
        print(f"    atom={a:.4f}  rate={r:.3f}  te_n={n}")

for d in ["MHClip_EN", "MHClip_ZH"]:
    analyze(d)
