"""
Task A (v2): GMM-posterior uncertainty boundary selector (logit space).

For each dataset:
  1. Apply logit transform to the threshold-source 2B scores:
     z = log(s/(1-s)). The 2B score is a sigmoid of an underlying
     logit — logit space is the natural parameterization.
  2. Fit a 2-component GaussianMixture on the logit-transformed
     scores.
  3. Identify the high-mean component as "hi".
  4. For each test score s_i, compute z_i = logit(s_i) and the
     posterior q_i = P(class=hi | z_i) via gmm.predict_proba.
  5. Define the boundary band as q_i ∈ [α, 1-α].
  6. Emit `candidates_band_alpha{α}.jsonl` per dataset.

The band is fully label-free at decide time. A diagnostic block at
the end uses labels (flagged "diagnostic only") to print:
  - band recall: fraction of 2B errors that fall inside the band
  - band precision: fraction of in-band videos that are 2B errors
  - band_error_ceiling: # in-band errors (used as Goal-2 ceiling)

Why logit-space posterior-cut:
  In raw-score space the 2B score lattice is highly polarized
  (near-0 or near-1 posteriors) which gives a degenerate band.
  Logit space is the underlying natural parameterization of the
  binary scoring model — the GMM in logit space has meaningful
  spread, and α is a probability cut on the *natural-space*
  uncertainty under the density model. Scale-invariant,
  asymmetry-aware, scientifically grounded.
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.mixture import GaussianMixture

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
sys.path.insert(0, os.path.join(_HERE, "..", "naive_baseline"))

from quick_eval_all import load_scores_file  # noqa: E402
from data_utils import load_annotations, SKIP_VIDEOS  # noqa: E402
from eval_generative_predictions import collapse_label  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "boundary_rescue")
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM"]


def load_v2_baseline():
    with open(os.path.join(OUT_ROOT, "v2_baseline.json")) as f:
        return json.load(f)


def load_baseline_preds(dataset):
    """Return list of dicts in file order from baseline_preds_v2.jsonl."""
    path = os.path.join(OUT_ROOT, dataset, "baseline_preds_v2.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_logit(scores, eps=1e-6):
    s = np.clip(np.asarray(scores, dtype=float), eps, 1.0 - eps)
    return np.log(s / (1.0 - s))


def fit_gmm(scores):
    """Fit a 2-component GMM in logit space; return (gmm, hi_idx)."""
    z = to_logit(scores).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(z)
    means = gmm.means_.flatten()
    hi_idx = int(np.argmax(means))
    return gmm, hi_idx


def select_band(dataset, alpha, v2_baseline):
    info = v2_baseline[dataset]
    fit_source = info["fit_source"]
    fit_path = info["fit_path"]
    threshold = float(info["threshold"])

    fit_scores = np.array(
        list(load_scores_file(fit_path).values()), dtype=float
    )
    gmm, hi_idx = fit_gmm(fit_scores)

    base_rows = load_baseline_preds(dataset)
    test_scores = np.array([r["score"] for r in base_rows], dtype=float)
    test_z = to_logit(test_scores).reshape(-1, 1)

    posteriors = gmm.predict_proba(test_z)[:, hi_idx]

    skip = SKIP_VIDEOS.get(dataset, set())
    candidates = []
    band_rows = []
    for r, s, q in zip(base_rows, test_scores, posteriors):
        vid = r["video_id"]
        if vid in skip:
            continue
        in_band = (q >= alpha) and (q <= (1.0 - alpha))
        if in_band:
            side = "above" if int(r["pred_baseline"]) == 1 else "below"
            cand = {
                "video_id": vid,
                "score": float(s),
                "posterior_hi": float(q),
                "threshold": threshold,
                "pred_baseline": int(r["pred_baseline"]),
                "side": side,
                "fit_source": fit_source,
            }
            candidates.append(cand)
            band_rows.append(r)

    out_path = os.path.join(
        OUT_ROOT, dataset, f"candidates_band_alpha{alpha:.2f}.jsonl"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    diag = {
        "dataset": dataset,
        "alpha": alpha,
        "fit_source": fit_source,
        "n_fit": len(fit_scores),
        "threshold": threshold,
        "n_test_after_skip": sum(
            1 for r in base_rows if r["video_id"] not in skip
        ),
        "n_band_total": len(candidates),
        "n_band_below": sum(1 for c in candidates if c["side"] == "below"),
        "n_band_above": sum(1 for c in candidates if c["side"] == "above"),
        "out_path": out_path,
    }

    # ---- DIAGNOSTIC ONLY (uses labels) ----
    ann = load_annotations(dataset)
    band_ids = {c["video_id"] for c in candidates}
    in_band_errors = 0
    for c in candidates:
        if c["video_id"] not in ann:
            continue
        gt = collapse_label(dataset, ann[c["video_id"]]["label"])
        if int(c["pred_baseline"]) != gt:
            in_band_errors += 1

    total_errors = 0
    for r in base_rows:
        if r["video_id"] in skip or r["video_id"] not in ann:
            continue
        gt = collapse_label(dataset, ann[r["video_id"]]["label"])
        if int(r["pred_baseline"]) != gt:
            total_errors += 1

    diag["band_error_ceiling"] = in_band_errors
    diag["total_errors"] = total_errors
    diag["band_recall"] = (
        in_band_errors / total_errors if total_errors > 0 else 0.0
    )
    diag["band_precision"] = (
        in_band_errors / len(candidates) if len(candidates) > 0 else 0.0
    )
    diag["goal2_min_correct_flips"] = (in_band_errors + 1) // 2  # ceil(half)
    return diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.30)
    args = parser.parse_args()

    v2_baseline = load_v2_baseline()

    print(f"alpha = {args.alpha:.2f}")
    print()
    print(
        f"{'dataset':<10} {'fit':<6} {'fit_n':>5} {'thr':>8}  "
        f"{'N':>4} {'|band|':>7} {'below':>5} {'above':>5}  "
        f"{'#err_in_band':>12} {'#err_total':>10} {'recall':>7} {'prec':>7} "
        f"{'goal2_min':>9}"
    )
    print("-" * 115)
    diags = []
    for ds in ALL_DATASETS:
        d = select_band(ds, args.alpha, v2_baseline)
        diags.append(d)
        print(
            f"{ds:<10} {d['fit_source']:<6} {d['n_fit']:>5d} "
            f"{d['threshold']:>8.4f}  "
            f"{d['n_test_after_skip']:>4d} {d['n_band_total']:>7d} "
            f"{d['n_band_below']:>5d} {d['n_band_above']:>5d}  "
            f"{d['band_error_ceiling']:>12d} {d['total_errors']:>10d} "
            f"{d['band_recall']:>7.3f} {d['band_precision']:>7.3f} "
            f"{d['goal2_min_correct_flips']:>9d}"
        )
    print()
    print("(# err_in_band / # err_total / recall / prec / goal2_min are diagnostic only — use labels)")

    diag_path = os.path.join(OUT_ROOT, f"band_diag_alpha{args.alpha:.2f}.json")
    with open(diag_path, "w") as f:
        json.dump(diags, f, indent=2)
    print(f"\nDiagnostic dump → {diag_path}")


if __name__ == "__main__":
    main()
