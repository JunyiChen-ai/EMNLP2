"""What features correlate with atom positive rate — using ONLY pool data (no labels)?

For each atom a, compute:
  - test count n(a)
  - pool count N(a)  (train + test)
  - train count Nt(a)
  - train/pool mass ratio
  - score value a
  - atom rank within pool uniques
  - local density (count of atoms within a window)

Compare against (post-hoc, for analysis only) conditional positive rate.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations


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

        test_atoms = np.round(test_x, 4)
        train_atoms = np.round(train_arr, 4)
        pool_atoms_arr = np.round(pool, 4)

        uniques = sorted(set(pool_atoms_arr.tolist()))
        rows = []
        for a in uniques:
            nt_test = int((test_atoms == a).sum())
            nt_train = int((train_atoms == a).sum())
            nt_pool = nt_test + nt_train
            pos = int(test_y[test_atoms == a].sum()) if nt_test > 0 else 0
            rate = pos / nt_test if nt_test > 0 else float('nan')
            # pool rank
            rows.append({"a": a, "n_test": nt_test, "n_train": nt_train, "n_pool": nt_pool, "pos": pos, "rate": rate})

        # For atoms with test presence, compute Spearman-style correlation of rate with various features
        have_test = [r for r in rows if r["n_test"] >= 2]  # need multiple to get stable rate

        print(f"\n=== {dataset} ===  ({len(have_test)} atoms with n_test>=2)")
        # Feature: atom score a  ---- already equivalent to threshold
        # Feature: log(n_pool)
        # Feature: train/pool ratio
        # Feature: neighborhood density

        # Compute corr of rate vs atom and vs log-count
        import scipy.stats as st
        atoms_arr = np.array([r["a"] for r in have_test])
        rates = np.array([r["rate"] for r in have_test])
        npools = np.array([r["n_pool"] for r in have_test])
        lognp = np.log1p(npools)
        ntrains = np.array([r["n_train"] for r in have_test])
        ntests = np.array([r["n_test"] for r in have_test])
        train_frac = ntrains / np.maximum(1, ntrains + ntests)

        # Local density: count of pool atoms within ±0.05 (excluding self)
        nbhd = []
        for r in have_test:
            nbhd.append(sum(1 for o in rows if o["a"] != r["a"] and abs(o["a"] - r["a"]) < 0.05))
        nbhd = np.array(nbhd, dtype=float)

        def corr(x, y, name):
            try:
                s = st.spearmanr(x, y).statistic
                p = st.pearsonr(x, y).statistic
                print(f"    {name:>20}   spearman={s:+.3f}  pearson={p:+.3f}")
            except Exception as e:
                print(f"    {name:>20}   ERR {e}")

        corr(atoms_arr, rates, "atom value")
        corr(lognp, rates, "log(n_pool)")
        corr(npools, rates, "n_pool")
        corr(train_frac, rates, "train_frac")
        corr(nbhd, rates, "nbhd density")
        corr(atoms_arr * (1.0/np.maximum(1, lognp)), rates, "atom/log(n_pool)")

        # Print top-5 and bottom-5 atoms by rate
        print(f"  atoms by rate (top 5):")
        for r in sorted(have_test, key=lambda r: -r["rate"])[:8]:
            print(f"    a={r['a']:.4f}  n_test={r['n_test']:3d}  n_train={r['n_train']:3d}  pos={r['pos']:3d}  rate={r['rate']:.3f}")
        print(f"  atoms by rate (bottom 5):")
        for r in sorted(have_test, key=lambda r: r["rate"])[:8]:
            print(f"    a={r['a']:.4f}  n_test={r['n_test']:3d}  n_train={r['n_train']:3d}  pos={r['pos']:3d}  rate={r['rate']:.3f}")


if __name__ == "__main__":
    main()
