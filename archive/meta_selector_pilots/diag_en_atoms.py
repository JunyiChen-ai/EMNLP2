"""Inspect EN atom structure near 0.3."""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations

for d in ["MHClip_EN", "MHClip_ZH"]:
    base = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    tr = load_scores_file(f"{base}/train_binary.jsonl")
    te = load_scores_file(f"{base}/test_binary.jsonl")
    ann = load_annotations(d)
    tex, tey = build_arrays(te, ann)
    tra = np.array(list(tr.values()), dtype=float)
    pool = np.concatenate([tra, np.array(list(te.values()), dtype=float)])
    print(f"\n=== {d} === pool N={len(pool)}, test N={len(tex)}, pos={int(tey.sum())}")
    # Group by 4-decimal atom
    atoms = {}
    for v in pool:
        k = round(float(v), 4)
        atoms[k] = atoms.get(k, 0) + 1
    sorted_atoms = sorted(atoms.items())
    print(f"   4-decimal atoms: {len(sorted_atoms)}")
    for a, n in sorted_atoms:
        # test composition
        matches_te = [(tex[i], int(tey[i])) for i in range(len(tex)) if round(tex[i], 4) == a]
        n_te = len(matches_te)
        n_te_pos = sum(y for _, y in matches_te)
        n_tr = sum(1 for v in tra if round(v, 4) == a)
        marker = " <-- STRICT FLIP" if (d == "MHClip_EN" and a == 0.3208) or (d == "MHClip_ZH" and a == 0.0474) else ""
        print(f"    a={a:.4f}  pool_n={n:4d}  tr_n={n_tr:4d}  te_n={n_te:3d}  te_pos={n_te_pos:2d}  te_neg={n_te-n_te_pos:2d}{marker}")
