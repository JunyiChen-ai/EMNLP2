"""For each atom, compute label-free features and correlate with labeled positive rate.

Label-free features per atom:
  - pool_n: count in pool
  - train_frac: fraction of pool from train split
  - raw_score: the atom value itself
  - rel_density: local pool density vs global
  - gap_below / gap_above: distance to neighboring atoms
  - rank_in_pool: rank of atom among all atoms
  - percentile: percentile of atom in pool distribution

Label (diagnostic only, NOT in selection path):
  - te_pos_rate: positive rate on test subset in this atom

Find label-free features that correlate with positive rate (high = good subset candidate).
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from collections import defaultdict
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

    atom_pool = defaultdict(int)
    atom_train = defaultdict(int)
    for x in pool:
        atom_pool[round(float(x), 4)] += 1
    for x in train_arr:
        atom_train[round(float(x), 4)] += 1

    atom_test = defaultdict(lambda: [0, 0, 0])
    for x, y in zip(test_x, test_y):
        a = round(float(x), 4)
        atom_test[a][0] += 1
        if y == 1:
            atom_test[a][1] += 1
        else:
            atom_test[a][2] += 1

    atoms = sorted(atom_pool.keys())
    n_pool = len(pool)
    feats = []
    for i, a in enumerate(atoms):
        pn = atom_pool[a]
        tr = atom_train[a]
        train_frac = tr / pn if pn else 0
        gap_below = atoms[i] - atoms[i-1] if i > 0 else atoms[i]
        gap_above = atoms[i+1] - atoms[i] if i < len(atoms)-1 else 1 - atoms[i]
        local_density = pn / (gap_below + gap_above + 1e-12)
        # percentile by pool
        pct = (np.asarray(pool) <= a).mean()
        tn, tp, tne = atom_test[a]
        rate = tp / tn if tn > 0 else np.nan
        feats.append({
            "atom": a, "pool_n": pn, "train_frac": train_frac,
            "gap_below": gap_below, "gap_above": gap_above,
            "local_density": local_density, "pct": pct,
            "te_n": tn, "te_pos": tp, "te_neg": tne, "rate": rate,
        })

    print(f"\n=== {d} ===")
    # Print atoms with test_n >= 3 sorted by positive rate
    print("\nAtoms with te_n>=3, sorted by positive rate desc:")
    print(f"{'atom':>7} {'pct':>6} {'pn':>4} {'tr_f':>5} {'lden':>7} {'gap-':>7} {'gap+':>7} {'te_n':>5} {'rate':>5}")
    data = [f for f in feats if f["te_n"] >= 3]
    data.sort(key=lambda r: -r["rate"])
    for f in data:
        print(f"{f['atom']:>7.4f} {f['pct']:>6.3f} {f['pool_n']:>4d} {f['train_frac']:>5.2f} "
              f"{f['local_density']:>7.1f} {f['gap_below']:>7.4f} {f['gap_above']:>7.4f} "
              f"{f['te_n']:>5d} {f['rate']:>5.2f}")

    # Correlation of label-free features with rate (on test-covered atoms only)
    valid = [f for f in feats if f["te_n"] >= 3]
    if len(valid) >= 5:
        keys = ["pool_n", "train_frac", "local_density", "pct", "atom", "gap_below", "gap_above"]
        rates = np.array([f["rate"] for f in valid])
        print(f"\nPearson correlation with positive rate (n={len(valid)}):")
        for k in keys:
            xs = np.array([f[k] for f in valid])
            if xs.std() > 0 and rates.std() > 0:
                r = np.corrcoef(xs, rates)[0,1]
                print(f"  {k:>15}: r = {r:+.3f}")

for d in ["MHClip_EN", "MHClip_ZH"]:
    analyze(d)
