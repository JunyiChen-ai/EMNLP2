"""Director-independent verification of meta-selector v2's infeasibility claim.

Enumerates every bucket-aligned threshold on MHClip_EN and MHClip_ZH 2B binary_nodef
test scores and asks: does any threshold strictly beat BOTH baseline ACC AND baseline
macro-F1?

Baselines (frozen, current):
  EN: ACC 0.7640, mF1 0.6532 (TF-Otsu bucket ≥ 0.3208)
  ZH: ACC 0.8121, mF1 0.7871 (TF-GMM  bucket ≥ 0.0373)

This script is a one-shot director diagnostic. It is NOT under src/meta_selector/
and does not influence either teammate's scope. It calls the frozen metrics()
function from quick_eval_all.py — no reimplementation.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations


BASELINE = {
    "MHClip_EN": {"acc": 0.7640, "mf": 0.6532},
    "MHClip_ZH": {"acc": 0.8121, "mf": 0.7871},
}


def enumerate_buckets(dataset):
    ann = load_annotations(dataset)
    base = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base}/train_binary.jsonl")
    test = load_scores_file(f"{base}/test_binary.jsonl")

    test_x, test_y = build_arrays(test, ann)
    pool_x, _ = build_arrays(
        {**train, **{k: v for k, v in test.items()}}, ann
    )
    pool_arr = np.concatenate(
        [
            np.fromiter((s for s in train.values()), dtype=float),
            np.fromiter((s for s in test.values()), dtype=float),
        ]
    )

    # All unique score values on the pool, rounded to 8 decimals to fuse FP noise.
    unique = sorted(set(round(float(s), 8) for s in pool_arr))
    # Bucket boundaries are thresholds halfway between adjacent unique values,
    # plus one just below the minimum and one just above the maximum.
    boundaries = []
    for i in range(len(unique)):
        # threshold = this unique value (>= t flags this bucket and above)
        boundaries.append(unique[i])
    # Also test thresholds exactly between adjacent uniques to be safe.
    for i in range(len(unique) - 1):
        boundaries.append((unique[i] + unique[i + 1]) / 2.0)
    boundaries = sorted(set(boundaries))

    rows = []
    for t in boundaries:
        m = metrics(test_x, test_y, float(t))
        rows.append(
            {
                "t": float(t),
                "acc": m["acc"],
                "mf": m["mf"],
                "n_pred_pos": int((test_x >= t).sum()),
            }
        )
    return rows


def find_strict_beats(rows, dataset):
    base = BASELINE[dataset]
    strict_both = [
        r for r in rows if r["acc"] > base["acc"] and r["mf"] > base["mf"]
    ]
    strict_acc_only = [
        r for r in rows if r["acc"] > base["acc"] and r["mf"] <= base["mf"]
    ]
    strict_mf_only = [
        r for r in rows if r["acc"] <= base["acc"] and r["mf"] > base["mf"]
    ]
    return strict_both, strict_acc_only, strict_mf_only


def main():
    out = {}
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        rows = enumerate_buckets(dataset)
        strict_both, strict_acc, strict_mf = find_strict_beats(rows, dataset)

        # Pull a neighborhood around the baseline threshold.
        base_t = 0.3208 if dataset == "MHClip_EN" else 0.0373
        nbh = [r for r in rows if abs(r["t"] - base_t) < 0.15]
        nbh_sorted = sorted(nbh, key=lambda r: r["t"])

        print(f"\n=== {dataset} ===")
        print(f"  baseline: ACC {BASELINE[dataset]['acc']:.4f}  mF1 {BASELINE[dataset]['mf']:.4f}")
        print(f"  unique score values (bucket count): {len([r for r in rows if r['t'] in set(r2['t'] for r2 in rows)])}")
        print(f"  candidates evaluated: {len(rows)}")
        print(f"  thresholds strictly beating BOTH ACC and mF1: {len(strict_both)}")
        if strict_both:
            print("    --- STRICT-BOTH HITS ---")
            for r in strict_both[:20]:
                print(f"    t={r['t']:.6f}  acc={r['acc']:.4f}  mf={r['mf']:.4f}  n_pred_pos={r['n_pred_pos']}")
        print(f"  thresholds strict-ACC-only (mf regresses): {len(strict_acc)}")
        print(f"  thresholds strict-mF1-only (acc regresses): {len(strict_mf)}")

        print("  --- neighborhood around baseline ---")
        for r in nbh_sorted:
            flag = ""
            if r["acc"] > BASELINE[dataset]["acc"] and r["mf"] > BASELINE[dataset]["mf"]:
                flag = " **STRICT-BOTH**"
            elif abs(r["t"] - base_t) < 1e-6:
                flag = " (baseline)"
            print(f"    t={r['t']:.6f}  acc={r['acc']:.4f}  mf={r['mf']:.4f}  n_pred_pos={r['n_pred_pos']}{flag}")

        out[dataset] = {
            "baseline": BASELINE[dataset],
            "n_candidates": len(rows),
            "strict_both_count": len(strict_both),
            "strict_both_examples": strict_both[:10],
            "strict_acc_only_count": len(strict_acc),
            "strict_mf_only_count": len(strict_mf),
        }

    outpath = "/data/jehc223/EMNLP2/results/analysis/director_pareto_check.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
