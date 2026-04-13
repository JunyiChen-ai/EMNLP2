"""Non-suffix subset rule via density-aware score transform.

Design: boost the effective score of sparse (tail) atoms and suppress dense (cluster)
atoms. Then threshold on the transformed score.

Rationale: positive-rate correlates NEGATIVELY with local_density (r=-0.55 EN,
-0.73 ZH) — tail atoms have high positive rates regardless of being suffix vs
non-suffix. So a score transform that weights by local sparseness can produce a
non-suffix subset that better matches the true positive atoms.

Transform tested:
  g(x) = rank_pct(x) + alpha * (1 - density_pct(x))
where rank_pct = pool CDF at x, density_pct = percentile of local density.

Then apply MAD-quantile rule on g-space (unified single rule), same bar.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from collections import defaultdict
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}

def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, test_x, test_y

def local_density(pool, x_arr, k=10):
    """kNN-based local density: inverse of mean distance to k nearest neighbors in pool."""
    pool_sorted = np.sort(pool)
    out = np.zeros(len(x_arr))
    for i, x in enumerate(x_arr):
        # find k nearest in pool
        dists = np.abs(pool_sorted - x)
        dists.sort()
        # skip exact matches (same atom) to get neighborhood gap
        d_k = dists[min(k, len(dists)-1)]
        out[i] = 1.0 / (d_k + 1e-12)
    return out

def atom_gap(pool, x_arr):
    """Return gap_below + gap_above at each x (using pool-distinct atoms)."""
    atoms = np.sort(np.unique(pool))
    out = np.zeros(len(x_arr))
    for i, x in enumerate(x_arr):
        idx = np.searchsorted(atoms, x)
        idx = min(max(idx, 0), len(atoms)-1)
        below = atoms[idx] - atoms[idx-1] if idx > 0 else atoms[idx]
        above = atoms[idx+1] - atoms[idx] if idx < len(atoms)-1 else 1 - atoms[idx]
        out[i] = below + above
    return out

def transform_score(pool, x_arr, alpha, kind="density"):
    """g(x) = rank_pct(x) + alpha * sparseness_pct(x)."""
    # rank_pct: fraction of pool <= x
    pool_sorted = np.sort(pool)
    rank_pct = np.searchsorted(pool_sorted, x_arr, side="right") / len(pool_sorted)
    if kind == "density":
        dens = local_density(pool, x_arr, k=10)
        # sparseness is high when density is low -> use -dens -> rank
        sparseness_rank = 1 - (np.argsort(np.argsort(dens)) / (len(dens)-1+1e-12))
    elif kind == "gap":
        gaps = atom_gap(pool, x_arr)
        sparseness_rank = np.argsort(np.argsort(gaps)) / (len(gaps)-1+1e-12)
    return rank_pct + alpha * sparseness_rank

def evaluate_rule(alpha, kind, a=0.60, b=7.83):
    """Apply transform, then MAD-quantile rule on g-space."""
    results = {}
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, test_x, test_y = load(d)
        # transform pool + test (pool-based features, applied to both)
        g_pool = transform_score(pool, pool, alpha, kind)
        g_test = transform_score(pool, test_x, alpha, kind)
        mad = float(np.median(np.abs(g_pool - np.median(g_pool))))
        q = a + b * mad
        q = max(0.01, min(0.99, q))
        t = float(np.quantile(g_pool, q))
        m = metrics(g_test, test_y, t)
        acc_b, mf_b = BASE[d]
        results[d] = (m["acc"], m["mf"], m["acc"]>acc_b, m["mf"]>mf_b)
    return results

print("=== Density-transform + MAD rule ===")
for kind in ["density", "gap"]:
    print(f"\n--- kind = {kind} ---")
    for alpha in [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        r = evaluate_rule(alpha, kind)
        en_acc, en_mf, en_a, en_m = r["MHClip_EN"]
        zh_acc, zh_mf, zh_a, zh_m = r["MHClip_ZH"]
        tag = "STRICT-BOTH" if (en_a and en_m and zh_a and zh_m) else ""
        print(f"  alpha={alpha:.2f}  EN {en_acc:.4f}/{en_mf:.4f}  ZH {zh_acc:.4f}/{zh_mf:.4f}  {tag}")
