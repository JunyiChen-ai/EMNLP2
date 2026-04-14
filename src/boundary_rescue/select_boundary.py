"""
Task A: label-free boundary candidate selection.

Rule S1 — Top-K-nearest-to-threshold per side (primary).
  For each dataset:
    below_candidates = top-K pred=0 videos by score (highest first)
    above_candidates = top-K pred=1 videos by score (lowest first)
    K = min(pool_size, TOPK_PER_SIDE)
  These are the "closest-to-threshold" videos per side — most likely
  to flip under a re-examination. TOPK_PER_SIDE defaults to 10.

Rule S1-atom (diagnostic only) — adjacent-atom cardinality is logged
alongside S1 for comparison. On binary_nodef 2B scores the lattice is
near-continuous (~0.95 unique atoms per video), so the atom rule gives
only ~1 candidate per side and is too tight. S1 top-K is the actual
method.

Both rules are fully score-only; no labels.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))

from data_utils import SKIP_VIDEOS  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "boundary_rescue")
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM"]

# Default K values. Frozen method uses K_BELOW=0, K_ABOVE=2 — only the
# two pred=1 candidates nearest the threshold are rescued. Below-side
# (FN-hunt) rescue is disabled because the 2B model's below-side
# extraction is unreliable at this boundary scale.
K_BELOW_DEFAULT = 0
K_ABOVE_DEFAULT = 2
ATOM_TOL = 1e-9


def load_baseline(dataset):
    path = os.path.join(OUT_ROOT, dataset, "baseline_preds.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def select_S1(rows, dataset, k_below, k_above):
    """Top-K-nearest-to-threshold per side. Returns (candidates, diag)."""
    skip = SKIP_VIDEOS.get(dataset, set())
    rows_eligible = [r for r in rows if r["video_id"] not in skip]
    if not rows_eligible:
        return [], {"n": 0}

    threshold = rows_eligible[0]["threshold"]

    below_pool = [r for r in rows_eligible if int(r["pred_baseline"]) == 0]
    above_pool = [r for r in rows_eligible if int(r["pred_baseline"]) == 1]

    # below_pool: sort by score DESC (highest-scored negatives closest to threshold)
    below_pool.sort(key=lambda r: -float(r["score"]))
    # above_pool: sort by score ASC (lowest-scored positives closest to threshold)
    above_pool.sort(key=lambda r: float(r["score"]))

    k0 = min(k_below, len(below_pool))
    k1 = min(k_above, len(above_pool))

    candidates = []
    for r in below_pool[:k0]:
        candidates.append(
            {
                "video_id": r["video_id"],
                "score": float(r["score"]),
                "threshold": threshold,
                "pred_baseline": 0,
                "side": "below",
                "distance": abs(float(r["score"]) - threshold),
            }
        )
    for r in above_pool[:k1]:
        candidates.append(
            {
                "video_id": r["video_id"],
                "score": float(r["score"]),
                "threshold": threshold,
                "pred_baseline": 1,
                "side": "above",
                "distance": abs(float(r["score"]) - threshold),
            }
        )

    diag = {
        "dataset": dataset,
        "threshold": threshold,
        "K_0_below": k0,
        "K_1_above": k1,
        "total": k0 + k1,
        "n_below_pool": len(below_pool),
        "n_above_pool": len(above_pool),
        "n_eligible": len(rows_eligible),
        "min_dist_below": candidates[0]["distance"] if k0 else None,
        "max_dist_below": candidates[k0 - 1]["distance"] if k0 else None,
        "min_dist_above": candidates[k0]["distance"] if k1 else None,
        "max_dist_above": candidates[k0 + k1 - 1]["distance"] if k1 else None,
    }
    return candidates, diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="final")
    parser.add_argument("--k-below", type=int, default=K_BELOW_DEFAULT)
    parser.add_argument("--k-above", type=int, default=K_ABOVE_DEFAULT)
    args = parser.parse_args()

    print(f"k_below = {args.k_below}, k_above = {args.k_above}")
    print()
    print(
        f"{'dataset':<10} {'t':>8} {'K0':>4} {'K1':>4} {'total':>6} "
        f"{'|below|':>8} {'|above|':>8}  "
        f"{'bMinD':>7} {'bMaxD':>7} {'aMinD':>7} {'aMaxD':>7}"
    )
    print("-" * 90)
    for ds in ALL_DATASETS:
        rows = load_baseline(ds)
        candidates, diag = select_S1(rows, ds, args.k_below, args.k_above)
        out_dir = os.path.join(OUT_ROOT, ds)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"candidates_{args.version}.jsonl")
        with open(out_path, "w") as f:
            for c in candidates:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        def fmt(v):
            return f"{v:.4f}" if v is not None else "   -   "

        print(
            f"{ds:<10} {diag['threshold']:>8.4f} "
            f"{diag['K_0_below']:>4d} {diag['K_1_above']:>4d} {diag['total']:>6d} "
            f"{diag['n_below_pool']:>8d} {diag['n_above_pool']:>8d}  "
            f"{fmt(diag['min_dist_below']):>7} {fmt(diag['max_dist_below']):>7} "
            f"{fmt(diag['min_dist_above']):>7} {fmt(diag['max_dist_above']):>7}"
        )


if __name__ == "__main__":
    main()
