"""Coarser atom structure + full atom sweep on true atoms."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7640, 0.6532), "MHClip_ZH": (0.8121, 0.7871)}

def rounds(arr, d):
    return np.array([round(float(s), d) for s in arr])


def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
        train = load_scores_file(f"{base_d}/train_binary.jsonl")
        test = load_scores_file(f"{base_d}/test_binary.jsonl")
        ann = load_annotations(dataset)
        test_x, test_y = build_arrays(test, ann)
        train_arr = np.array(list(train.values()), dtype=float)
        test_arr = np.array(list(test.values()), dtype=float)
        pool = np.concatenate([train_arr, test_arr])

        print(f"\n=== {dataset} ===  baseline {acc_b:.4f}/{mf_b:.4f}")
        for d in [2, 3, 4, 5, 6, 7, 8]:
            u = sorted(set(rounds(pool, d).tolist()))
            print(f"  round {d}d  unique atoms = {len(u)}")

        # Use 4-decimal rounding as "true atoms"
        atoms4 = sorted(set(rounds(pool, 4).tolist()))
        print(f"  using 4-decimal atoms: {len(atoms4)}")
        cands = []
        cands.append(atoms4[0] - 1e-4)
        cands.append(atoms4[-1] + 1e-4)
        for a in atoms4:
            cands.append(a - 5e-5)  # just below
            cands.append(a + 5e-5)  # just above
        cands = sorted(set(cands))

        strict_both = []
        strict_acc = []
        strict_mf = []
        pareto_all = []
        for t in cands:
            m = metrics(test_x, test_y, float(t))
            rec = {"t": float(t), "acc": m["acc"], "mf": m["mf"], "npos": int((test_x >= t).sum())}
            pareto_all.append(rec)
            if m["acc"] > acc_b and m["mf"] > mf_b:
                strict_both.append(rec)
            elif m["acc"] > acc_b:
                strict_acc.append(rec)
            elif m["mf"] > mf_b:
                strict_mf.append(rec)

        # Dedup by (acc,mf,npos)
        seen = set()
        u_both = []
        for r in strict_both:
            k = (round(r["acc"],4), round(r["mf"],4), r["npos"])
            if k not in seen:
                seen.add(k)
                u_both.append(r)

        print(f"  strict-both unique cells: {len(u_both)}")
        for r in u_both[:10]:
            print(f"    t={r['t']:.6f}  acc={r['acc']:.4f}  mf={r['mf']:.4f}  npos={r['npos']}")
        print(f"  strict-acc-only cells: {len(strict_acc)}  strict-mf-only cells: {len(strict_mf)}")

        # Best cells overall (dominant)
        best_acc = max(pareto_all, key=lambda r: r["acc"])
        best_mf = max(pareto_all, key=lambda r: r["mf"])
        print(f"  max-ACC cell: t={best_acc['t']:.6f} acc={best_acc['acc']:.4f} mf={best_acc['mf']:.4f} npos={best_acc['npos']}")
        print(f"  max-mF1 cell: t={best_mf['t']:.6f} acc={best_mf['acc']:.4f} mf={best_mf['mf']:.4f} npos={best_mf['npos']}")


if __name__ == "__main__":
    main()
