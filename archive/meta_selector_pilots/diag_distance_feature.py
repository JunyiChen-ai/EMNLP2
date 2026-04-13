"""Check if distance-to-kth-nearest-train-sample provides label-free
discrimination that, combined with score, gives non-suffix power.

For each test sample, compute:
  - score
  - d_k: distance to k-th nearest TRAIN sample (k in standard choices)
  - also: rank in combined pool (monotone in score)
  - also: local KDE estimate at sample (standard Silverman bandwidth)

Then check if (score, d_k) jointly permit a rule like "flag iff score > median
OR d_k > some_pool_stat" to improve on baseline.

Key test: does this feature combo give sign-consistent advantage on both
EN and ZH? If yes, a parameter-free rule is possible.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    return train_arr, test_x, test_y


def d_knn(x, train_arr, k):
    """Distance from x to its k-th nearest train sample (1D)."""
    diffs = np.abs(train_arr - x)
    return np.partition(diffs, k-1)[k-1]


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    n = len(train)
    # Standard k choices
    k_sqrt = max(1, int(np.sqrt(n)))
    print(f"\n=== {d} ===  |train|={n}  k_sqrt={k_sqrt}  n_test={len(test_x)}  pos_rate={test_y.mean():.3f}")

    # For each test sample, compute score and d_k for several k
    ks = [1, 3, 5, 10, k_sqrt, n//10]
    ks = sorted(set(ks))

    scores_pos = test_x[test_y == 1]
    scores_neg = test_x[test_y == 0]
    print(f"  score means:  pos={scores_pos.mean():.4f}  neg={scores_neg.mean():.4f}")

    for k in ks:
        d_pos = np.array([d_knn(x, train, k) for x in scores_pos])
        d_neg = np.array([d_knn(x, train, k) for x in scores_neg])
        print(f"  k={k:3d}: d_knn(pos)_mean={d_pos.mean():.5f}  d_knn(neg)_mean={d_neg.mean():.5f}"
              f"  d_knn(pos)/d_knn(neg)={d_pos.mean()/d_neg.mean():.3f}")

    # Also: correlation of d_k with label
    print(f"\n  Label-correlation of d_k (pos=1):")
    for k in ks:
        d_all = np.array([d_knn(x, train, k) for x in test_x])
        corr = np.corrcoef(d_all, test_y.astype(float))[0, 1]
        print(f"    k={k:3d}: corr(d_k, y)={corr:+.4f}")

    # KDE density at each test sample using combined pool
    from scipy.stats import gaussian_kde
    pool = np.concatenate([train, test_x])
    # Silverman bandwidth (standard)
    kde = gaussian_kde(pool, bw_method='silverman')
    dens_all = kde(test_x)
    corr_dens = np.corrcoef(dens_all, test_y.astype(float))[0, 1]
    print(f"  KDE Silverman density at sample -- corr(density, y)={corr_dens:+.4f}")

    # Combined rule test: score alone vs score + log(1/density)
    from quick_eval_all import otsu_threshold
    # Pure score
    t_s = otsu_threshold(test_x)
    m_s = metrics(test_x, test_y, t_s)
    # Transform: isolation_score = score * (1 + alpha * log(1/density_quantile))
    # No alpha: just density-inverse rank
    rank_score = np.array([(test_x <= x).mean() for x in test_x])
    rank_dens = np.array([(dens_all <= x).mean() for x in dens_all])
    # Higher rank_score + lower rank_dens (more isolated) => more likely positive
    combined = rank_score + (1.0 - rank_dens)
    t_c = otsu_threshold(combined)
    m_c = metrics(combined, test_y, t_c)
    print(f"  Otsu on score only:                  acc={m_s['acc']:.4f}  mf={m_s['mf']:.4f}")
    print(f"  Otsu on rank_score + (1-rank_dens):  acc={m_c['acc']:.4f}  mf={m_c['mf']:.4f}")

    # Another combo: just the rank_score (should match Otsu on score)
    t_r = otsu_threshold(rank_score)
    m_r = metrics(rank_score, test_y, t_r)
    # Equal-weight average: (rank_score + (1-rank_dens)) / 2
    avg = (rank_score + (1.0 - rank_dens)) / 2
    t_a = otsu_threshold(avg)
    m_a = metrics(avg, test_y, t_a)
    print(f"  Otsu on rank_score only:             acc={m_r['acc']:.4f}  mf={m_r['mf']:.4f}")
    print(f"  Otsu on avg(rank_score, 1-rank_dens):acc={m_a['acc']:.4f}  mf={m_a['mf']:.4f}")
