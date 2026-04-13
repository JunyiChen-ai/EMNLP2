"""Atom-boundary Pareto enumeration on current pool + sub-FP strict-both search."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations


BASE = {"MHClip_EN": (0.7640, 0.6532), "MHClip_ZH": (0.8121, 0.7871)}
OUT = "/data/jehc223/EMNLP2/results/meta_selector/diag_pareto.json"


def main():
    out = {}
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
        train = load_scores_file(f"{base_d}/train_binary.jsonl")
        test = load_scores_file(f"{base_d}/test_binary.jsonl")
        ann = load_annotations(dataset)
        test_x, test_y = build_arrays(test, ann)
        pool = np.array(list(train.values()) + list(test.values()), dtype=float)

        atoms = sorted(set(round(float(s), 8) for s in pool))
        # Candidate thresholds: every atom + midpoints + one below min and above max.
        cands = set()
        cands.add(atoms[0] - 1e-6)
        cands.add(atoms[-1] + 1e-6)
        for a in atoms:
            cands.add(a)
        for i in range(len(atoms) - 1):
            cands.add((atoms[i] + atoms[i+1]) / 2.0)
        cands = sorted(cands)

        strict_both, strict_acc, strict_mf, neigh = [], [], [], []
        for t in cands:
            m = metrics(test_x, test_y, float(t))
            rec = {"t": float(t), "acc": m["acc"], "mf": m["mf"], "npos": int((test_x >= t).sum())}
            if m["acc"] > acc_b and m["mf"] > mf_b:
                strict_both.append(rec)
            if m["acc"] > acc_b and m["mf"] <= mf_b:
                strict_acc.append(rec)
            if m["acc"] <= acc_b and m["mf"] > mf_b:
                strict_mf.append(rec)
            if abs(t - (0.273354 if dataset == "MHClip_EN" else 0.036233)) < 0.15:
                neigh.append(rec)

        neigh = sorted(neigh, key=lambda r: r["t"])
        out[dataset] = {
            "n_cands": len(cands),
            "n_strict_both": len(strict_both),
            "strict_both_sample": strict_both[:20],
            "n_strict_acc_only": len(strict_acc),
            "n_strict_mf_only": len(strict_mf),
        }
        print(f"\n=== {dataset} ===")
        print(f"  baseline: {acc_b:.4f} / {mf_b:.4f}")
        print(f"  candidates: {len(cands)}")
        print(f"  strict-both: {len(strict_both)}")
        for r in strict_both[:8]:
            print(f"    t={r['t']:.6f}  acc={r['acc']:.4f}  mf={r['mf']:.4f}  n={r['npos']}")
        print(f"  strict-acc-only: {len(strict_acc)}  strict-mf-only: {len(strict_mf)}")
        print(f"  neighborhood around baseline:")
        for r in neigh:
            flag = ""
            if r["acc"] > acc_b and r["mf"] > mf_b: flag = " *STRICT-BOTH*"
            print(f"    t={r['t']:.6f}  acc={r['acc']:.4f}  mf={r['mf']:.4f}  n={r['npos']}{flag}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
