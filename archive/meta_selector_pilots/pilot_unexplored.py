"""Pilot the 5 unexplored directions from director's feedback:

1. Train-pool-as-reference calibration: apply train CDF to test scores, then
   apply threshold method. This is NOT PU self-label — it's shape-reference.

2. Multi-stage pipeline: apply one rule to partition atoms into confident and
   uncertain, apply another rule to uncertain.

3. Calibration-then-threshold: probability matching (map test CDF to uniform,
   then Otsu/GMM on uniform scores).

4. Information-theoretic thresholding beyond histogram entropy families:
   - Minimum Error Thresholding (Kittler-Illingworth 1986) - published standard
   - Cross-entropy minimization (Li-Lee 1993 actually; or Brink 1996)

5. Non-threshold prediction rules:
   - Flag by local-context property (density-peak detection on test density,
     without any parameters)
   - Mode-anchored fence rule
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def kittler_illingworth(scores, n_bins=256):
    """Minimum Error Thresholding (Kittler & Illingworth 1986).
    Published standard, n_bins=256 is the image-processing default.
    Returns the threshold minimizing the error criterion.
    """
    s = np.asarray(scores, dtype=float)
    hist, edges = np.histogram(s, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return float(np.median(s))

    best_j = 0
    best_cost = float('inf')
    for j in range(1, n_bins):
        p0 = hist[:j].sum() / total
        p1 = hist[j:].sum() / total
        if p0 < 1e-6 or p1 < 1e-6:
            continue
        mu0 = (hist[:j] * centers[:j]).sum() / (p0 * total)
        mu1 = (hist[j:] * centers[j:]).sum() / (p1 * total)
        var0 = (hist[:j] * (centers[:j] - mu0) ** 2).sum() / (p0 * total)
        var1 = (hist[j:] * (centers[j:] - mu1) ** 2).sum() / (p1 * total)
        if var0 <= 0 or var1 <= 0:
            continue
        sigma0 = np.sqrt(var0)
        sigma1 = np.sqrt(var1)
        cost = 1 + 2 * (p0 * np.log(sigma0) + p1 * np.log(sigma1)) - 2 * (p0 * np.log(p0) + p1 * np.log(p1))
        if cost < best_cost:
            best_cost = cost
            best_j = j
    return float(centers[best_j])


def train_cdf_transform(test_x, train):
    """For each test value, return fraction of train samples ≤ value."""
    return np.array([(train <= x).mean() for x in test_x])


def test_kde_peaks(scores):
    """Find density peaks in the test density using Silverman KDE.
    Returns (peak_locations, density_at_peak).
    """
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
    kde = gaussian_kde(scores, bw_method='silverman')
    x_grid = np.linspace(scores.min(), scores.max(), 500)
    dens = kde(x_grid)
    peaks, _ = find_peaks(dens)
    return x_grid[peaks], dens[peaks]


def multistage_confident_uncertain(test_x, pool):
    """Stage 1: Tukey inner fence (Q3+1.5*IQR on pool) → confident positives.
    Stage 2: on uncertain set (scores below fence), apply Otsu.
    Return the UNION as prediction.
    """
    q1, q3 = np.quantile(pool, [0.25, 0.75])
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    confident = test_x >= fence
    uncertain_mask = ~confident
    uncertain_x = test_x[uncertain_mask]
    if len(uncertain_x) < 4:
        return confident.astype(float)
    t_un = otsu_threshold(uncertain_x)
    uncertain_pos = uncertain_x >= t_un
    pred = np.zeros_like(test_x, dtype=bool)
    pred[confident] = True
    idx_unc = np.where(uncertain_mask)[0]
    pred[idx_unc[uncertain_pos]] = True
    return pred.astype(float)


def run_all():
    for d in ["MHClip_EN", "MHClip_ZH"]:
        train, test_x, test_y = load(d)
        pool = np.concatenate([train, test_x])
        acc_b, mf_b = BASE[d]
        print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

        # 1. Train CDF transform + Otsu
        test_cdf_train = train_cdf_transform(test_x, train)
        t_cdf = otsu_threshold(test_cdf_train)
        m = metrics(test_cdf_train, test_y, t_cdf)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"  [1] Train-CDF + Otsu       acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")

        # 1b. Train CDF transform + GMM
        t_cdf_g = gmm_threshold(test_cdf_train)
        m = metrics(test_cdf_train, test_y, t_cdf_g)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"  [1b] Train-CDF + GMM       acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")

        # 2. Pool-CDF transform + Otsu (shape calibration)
        pool_cdf = np.array([(pool <= x).mean() for x in test_x])
        t_p = otsu_threshold(pool_cdf)
        m = metrics(pool_cdf, test_y, t_p)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"  [2] Pool-CDF + Otsu        acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")

        # 3. Kittler-Illingworth on test
        t_ki = kittler_illingworth(test_x, n_bins=256)
        m = metrics(test_x, test_y, t_ki)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"  [3] Kittler-Illingworth    acc={m['acc']:.4f}  mf={m['mf']:.4f}  t={t_ki:.4f}  {tag}")

        # 3b. Kittler-Illingworth on pool
        t_ki2 = kittler_illingworth(pool, n_bins=256)
        m = metrics(test_x, test_y, t_ki2)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"  [3b] KI on pool            acc={m['acc']:.4f}  mf={m['mf']:.4f}  t={t_ki2:.4f}  {tag}")

        # 4. Multi-stage: Tukey confident + Otsu uncertain (UNION)
        pred = multistage_confident_uncertain(test_x, pool)
        m = metrics(pred, test_y, 0.5)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"  [4] Multi-stage            acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")

        # 5. Test-KDE peak rule: flag any sample at or above the highest peak
        peaks, dens_peaks = test_kde_peaks(test_x)
        if len(peaks) > 0:
            highest_peak_loc = peaks[np.argmin(peaks)]  # leftmost peak (the main mode - neg core)
            # Flag samples beyond the rightmost peak
            rightmost_peak_loc = peaks[-1]
            # A more principled rule: flag samples beyond main-mode valley
            # Find the valley (min density) between the leftmost and rightmost peaks
            from scipy.stats import gaussian_kde
            from scipy.signal import find_peaks
            kde = gaussian_kde(test_x, bw_method='silverman')
            x_grid = np.linspace(test_x.min(), test_x.max(), 500)
            dens = kde(x_grid)
            valleys, _ = find_peaks(-dens)
            valley_locs = x_grid[valleys]
            if len(valley_locs) > 0:
                t_v = float(valley_locs[0])  # first valley after leftmost peak
                m = metrics(test_x, test_y, t_v)
                tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
                print(f"  [5] KDE first valley       acc={m['acc']:.4f}  mf={m['mf']:.4f}  t={t_v:.4f}  {tag}")
            else:
                print(f"  [5] KDE first valley       (no valley found)")
        else:
            print(f"  [5] KDE first valley       (no peaks found)")

        # 6. Otsu on -log-density (density-inverted) - this collapses dense
        # regions so Otsu can find clusters in sparse mid-range
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(pool, bw_method='silverman')
        dens_test = kde(test_x)
        neg_log = -np.log(dens_test + 1e-12)
        t_nl = otsu_threshold(neg_log)
        m = metrics(neg_log, test_y, t_nl)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"  [6] Otsu on -log(pool-KDE) acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")


run_all()
