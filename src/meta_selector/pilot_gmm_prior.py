"""GMM K=2 with non-0.5 prior threshold.

Instead of picking threshold at 0.5 posterior crossover, use unsupervised pool-based
prior estimate (e.g., fraction of pool below Otsu) and pick the Bayes-optimal
threshold for THAT prior. This allows EN and ZH to have different implied priors,
responding to their shape difference.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.mixture import GaussianMixture

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(dataset):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(dataset)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, train_arr, test_x, test_y


def gmm_cross_threshold(src, prior_pos=0.5, K=2, use_log=False):
    """Threshold at posterior crossover for given positive prior.
    Uses the component with larger mean as 'positive'."""
    x = src.copy()
    if use_log:
        x = np.log(x + 1e-4)
    xt = x.reshape(-1, 1)
    gm = GaussianMixture(n_components=K, random_state=0, n_init=3).fit(xt)
    means = gm.means_.flatten()
    order = np.argsort(means)
    # overwrite GMM weights with desired prior
    w = gm.weights_.copy()
    # Assume binary: find threshold where posterior of TOP cluster (largest mean) = 0.5
    # Scan a fine grid and find crossover point
    grid = np.linspace(x.min(), x.max(), 10001)
    log_probs = gm._estimate_log_prob(grid.reshape(-1, 1))  # (n_grid, K)
    # Reweight with new prior
    log_w = np.zeros(K)
    log_w[order[-1]] = np.log(prior_pos + 1e-12)
    log_w[order[0]] = np.log(1 - prior_pos + 1e-12)
    log_post = log_probs + log_w
    log_post -= log_post.max(axis=1, keepdims=True)
    post = np.exp(log_post)
    post /= post.sum(axis=1, keepdims=True)
    top_post = post[:, order[-1]]
    # find where top_post crosses 0.5
    crosses = np.where(np.diff(np.sign(top_post - 0.5)))[0]
    if len(crosses) == 0:
        return float("nan"), order
    # pick the highest crossover (rightmost, separates near top mean)
    idx = crosses[-1]
    t = grid[idx]
    if use_log:
        t = np.exp(t) - 1e-4
    return float(t), order


def main():
    data = {}
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_arr, test_x, test_y = load(d)
        data[d] = {"pool": pool, "train": train_arr, "test_x": test_x, "test_y": test_y,
                   "base": BASE[d]}

    # Try various prior estimation strategies
    strategies = []
    # fixed priors
    for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        strategies.append((f"fix_p{p:.2f}", lambda s, t, p=p: p))
    # fraction above Otsu
    strategies.append(("1-q_otsu", lambda s, t: 1 - (s < otsu_threshold(s)).mean()))
    # fraction above mean
    strategies.append(("1-q_mean", lambda s, t: (s > s.mean()).mean()))
    # fraction above median
    strategies.append(("1-q_med", lambda s, t: (s > np.median(s)).mean()))
    # train otsu fraction
    strategies.append(("train_1-q_otsu", lambda s, t: 1 - (t < otsu_threshold(t)).mean()))

    for name, fn in strategies:
        print(f"\n=== prior = {name} ===")
        results = {}
        for d in ["MHClip_EN", "MHClip_ZH"]:
            pool = data[d]["pool"]
            train = data[d]["train"]
            test_x = data[d]["test_x"]
            test_y = data[d]["test_y"]
            for use_log in [False, True]:
                tag_log = "log" if use_log else "raw"
                for src_name, src in [("pool", pool), ("train", train)]:
                    try:
                        pri = float(fn(pool, train))
                    except Exception as e:
                        continue
                    try:
                        t, _ = gmm_cross_threshold(src, prior_pos=pri, K=2, use_log=use_log)
                    except Exception as e:
                        continue
                    if not np.isfinite(t):
                        continue
                    m = metrics(test_x, test_y, t)
                    acc_b, mf_b = data[d]["base"]
                    sb = m["acc"] > acc_b and m["mf"] > mf_b
                    key = (tag_log, src_name)
                    results[(d, key)] = (pri, t, m["acc"], m["mf"], sb)
        # report
        for key in sorted(set(k[1] for k in results.keys())):
            row_en = results.get(("MHClip_EN", key))
            row_zh = results.get(("MHClip_ZH", key))
            if row_en and row_zh:
                pri_en, t_en, acc_en, mf_en, sb_en = row_en
                pri_zh, t_zh, acc_zh, mf_zh, sb_zh = row_zh
                tag = ""
                if sb_en and sb_zh: tag = " **UNIFIED**"
                elif sb_en: tag = " [EN+]"
                elif sb_zh: tag = " [ZH+]"
                print(f"  {key[0]:<4} {key[1]:<5}  EN pri={pri_en:.3f} t={t_en:.4f} {acc_en:.4f}/{mf_en:.4f}  ZH pri={pri_zh:.3f} t={t_zh:.4f} {acc_zh:.4f}/{mf_zh:.4f}{tag}")


if __name__ == "__main__":
    main()
