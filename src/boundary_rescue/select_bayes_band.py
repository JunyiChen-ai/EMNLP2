"""
Option A: GMM Bayes error matching — fully unsupervised band selection.

Idea: a 2-component GMM fit on the logit-transformed scores predicts
its own Bayes error rate analytically. Pick the samples with the
highest "error contribution" min(q, 1-q) until their cumulative
contribution covers the GMM-predicted expected error mass. This
removes the α (or τ) hyperparameter entirely — band width is set
purely by the GMM's intrinsic structure of the data.

Output: `candidates_bayes_band.jsonl` per dataset, same schema as
`candidates_band_alpha*.jsonl`.

Usage:
  python src/boundary_rescue/select_bayes_band.py [--mass 1.0]
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import norm
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


def to_logit(scores, eps=1e-6):
    s = np.clip(np.asarray(scores, dtype=float), eps, 1.0 - eps)
    return np.log(s / (1.0 - s))


def fit_gmm(scores):
    z = to_logit(scores).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(z)
    means = gmm.means_.flatten()
    hi_idx = int(np.argmax(means))
    return gmm, hi_idx


def gmm_bayes_error(gmm, n_grid=8192):
    """Numerically integrate the 2-component 1D GMM Bayes error rate."""
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    span = 6.0 * stds.max()
    lo = means.min() - span
    hi = means.max() + span
    z = np.linspace(lo, hi, n_grid)
    p0 = weights[0] * norm.pdf(z, means[0], stds[0])
    p1 = weights[1] * norm.pdf(z, means[1], stds[1])
    err_density = np.minimum(p0, p1)
    return float(np.trapz(err_density, z))


def load_v2_baseline():
    with open(os.path.join(OUT_ROOT, "v2_baseline.json")) as f:
        return json.load(f)


def load_baseline_preds(dataset):
    path = os.path.join(OUT_ROOT, dataset, "baseline_preds_v2.jsonl")
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def select_bayes_band(dataset, mode, v2_baseline, mass_fraction=1.0):
    """Return band as a set of test indices.

    mode == 'rate': band = {i : err_i > E_bayes}
       Per-sample uncertainty exceeds the dataset's average Bayes
       error rate. Fully parameter-free, per-dataset adaptive.

    mode == 'mass': band = top-K samples by err_i descending such
       that cumulative sum >= mass_fraction * N * E_bayes. Covers
       a fraction of the predicted total error budget.
    """
    info = v2_baseline[dataset]
    fit_path = info["fit_path"]
    threshold = float(info["threshold"])

    fit_scores = np.array(list(load_scores_file(fit_path).values()), dtype=float)
    gmm, hi_idx = fit_gmm(fit_scores)
    e_bayes = gmm_bayes_error(gmm)

    base_rows = load_baseline_preds(dataset)
    test_scores = np.array([r["score"] for r in base_rows], dtype=float)
    test_z = to_logit(test_scores).reshape(-1, 1)

    posteriors = gmm.predict_proba(test_z)[:, hi_idx]
    err_i = np.minimum(posteriors, 1.0 - posteriors)

    skip = SKIP_VIDEOS.get(dataset, set())
    eligible_idx = np.array(
        [i for i, r in enumerate(base_rows) if r["video_id"] not in skip],
        dtype=int,
    )
    n_eligible = len(eligible_idx)

    if mode == "rate":
        # Per-sample threshold: err_i > E_bayes
        in_band_eligible = err_i[eligible_idx] > e_bayes
        selected_set = set(int(i) for i in eligible_idx[in_band_eligible])
        sorted_err_for_diag = np.sort(err_i[eligible_idx][in_band_eligible])
        k = int(in_band_eligible.sum())
    elif mode == "mass":
        target_mass = mass_fraction * n_eligible * e_bayes
        order_in_eligible = np.argsort(-err_i[eligible_idx])
        sorted_err = err_i[eligible_idx][order_in_eligible]
        cum = np.cumsum(sorted_err)
        if target_mass <= 0:
            k = 0
        else:
            k = int(np.searchsorted(cum, target_mass, side="left") + 1)
            k = min(k, n_eligible)
        selected_eligible_idx = eligible_idx[order_in_eligible[:k]]
        selected_set = set(int(i) for i in selected_eligible_idx)
        sorted_err_for_diag = sorted_err[:k]
    else:
        raise ValueError(f"unknown mode {mode}")

    candidates = []
    for i, r in enumerate(base_rows):
        if i not in selected_set:
            continue
        side = "above" if int(r["pred_baseline"]) == 1 else "below"
        candidates.append(
            {
                "video_id": r["video_id"],
                "score": float(r["score"]),
                "posterior_hi": float(posteriors[i]),
                "err_contribution": float(err_i[i]),
                "threshold": threshold,
                "pred_baseline": int(r["pred_baseline"]),
                "side": side,
                "fit_source": info["fit_source"],
            }
        )

    if mode == "rate":
        suffix = "rate"
    else:
        suffix = f"mass{mass_fraction:.2f}"
    out_path = os.path.join(
        OUT_ROOT, dataset, f"candidates_bayes_band_{suffix}.jsonl"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Diagnostic with labels (clearly marked)
    ann = load_annotations(dataset)
    in_band_errors = 0
    total_errors = 0
    for i, r in enumerate(base_rows):
        if r["video_id"] in skip or r["video_id"] not in ann:
            continue
        gt = collapse_label(dataset, ann[r["video_id"]]["label"])
        is_err = int(r["pred_baseline"]) != gt
        if is_err:
            total_errors += 1
            if i in selected_set:
                in_band_errors += 1

    return {
        "dataset": dataset,
        "fit_source": info["fit_source"],
        "n_fit": len(fit_scores),
        "threshold": threshold,
        "gmm_means_logit": gmm.means_.flatten().tolist(),
        "gmm_stds_logit": np.sqrt(gmm.covariances_.flatten()).tolist(),
        "gmm_weights": gmm.weights_.flatten().tolist(),
        "bayes_error_rate": e_bayes,
        "expected_total_errors": float(n_eligible * e_bayes),
        "n_eligible": n_eligible,
        "n_band": int(k),
        "n_band_below": sum(1 for c in candidates if c["side"] == "below"),
        "n_band_above": sum(1 for c in candidates if c["side"] == "above"),
        "min_err_in_band": float(sorted_err_for_diag[0]) if k > 0 else None,
        "max_err_in_band": float(sorted_err_for_diag[-1]) if k > 0 else None,
        "out_path": out_path,
        "band_error_ceiling": in_band_errors,
        "total_errors": total_errors,
        "band_recall": in_band_errors / total_errors if total_errors else 0,
        "band_precision": in_band_errors / k if k > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["rate", "mass"],
        default="rate",
        help="rate: per-sample err > E_bayes (parameter-free); "
             "mass: cumulative top-K (--mass fraction)",
    )
    parser.add_argument("--mass", type=float, default=1.0)
    args = parser.parse_args()

    v2_baseline = load_v2_baseline()
    diags = []
    print(f"Bayes-band selection, mode={args.mode}"
          + (f" mass={args.mass:.2f}" if args.mode == "mass" else "") + "\n")
    print(
        f"{'dataset':<10} {'fit_n':>5} {'E_bayes':>8} {'expE_err':>9}  "
        f"{'N':>4} {'|band|':>7} {'below':>5} {'above':>5} "
        f"{'min_err':>9}  {'#err_band':>9} {'#err_tot':>9} "
        f"{'recall':>7} {'prec':>6}"
    )
    print("-" * 130)
    for ds in ALL_DATASETS:
        d = select_bayes_band(ds, args.mode, v2_baseline, args.mass)
        diags.append(d)
        print(
            f"{ds:<10} {d['n_fit']:>5d} "
            f"{d['bayes_error_rate']:>8.4f} {d['expected_total_errors']:>9.2f}  "
            f"{d['n_eligible']:>4d} {d['n_band']:>7d} "
            f"{d['n_band_below']:>5d} {d['n_band_above']:>5d} "
            f"{(d['min_err_in_band'] if d['min_err_in_band'] is not None else 0):>9.4f}  "
            f"{d['band_error_ceiling']:>9d} {d['total_errors']:>9d} "
            f"{d['band_recall']:>7.3f} {d['band_precision']:>6.3f}"
        )
    print()
    print("(diagnostics use labels, not part of decision pipeline)")
    suffix = "rate" if args.mode == "rate" else f"mass{args.mass:.2f}"
    diag_path = os.path.join(OUT_ROOT, f"bayes_band_diag_{suffix}.json")
    with open(diag_path, "w") as f:
        json.dump(diags, f, indent=2)
    print(f"\nDiagnostic dump → {diag_path}")


if __name__ == "__main__":
    main()
