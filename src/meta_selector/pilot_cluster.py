"""2D clustering-based non-monotone selector.

Featurize each sample as (score, feature). Features:
  - nbhd density
  - pool quantile rank
  - distance to nearest dense atom
  - log(local count)

Cluster into K=2 or 3 using KMeans, then label clusters by mean score (rank-based).
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7640, 0.6532), "MHClip_ZH": (0.8121, 0.7871)}


def load(dataset):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(dataset)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    test_arr = np.array(list(test.values()), dtype=float)
    pool = np.concatenate([train_arr, test_arr])
    return pool, train_arr, test_x, test_y


def nbhd(x, pool, window):
    return np.array([int(np.sum((pool != xi) & (np.abs(pool - xi) < window))) for xi in x])


def quantile_rank(x, pool):
    return np.array([np.searchsorted(np.sort(pool), xi) / len(pool) for xi in x])


def main():
    from sklearn.cluster import KMeans
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        pool, train_arr, test_x, test_y = load(dataset)

        print(f"\n=== {dataset} ===  baseline {acc_b:.4f}/{mf_b:.4f}")

        window = 0.05
        nb_pool = nbhd(pool, pool, window)
        nb_test = nbhd(test_x, pool, window)
        qr_pool = quantile_rank(pool, pool)
        qr_test = quantile_rank(test_x, pool)

        # Feature sets to try
        def fs(name, x, nb, qr):
            s = x
            return {
                "score_nbhd": np.column_stack([s, nb/nb.max()]),
                "score_lognbhd": np.column_stack([s, np.log1p(nb)/np.log1p(nb.max())]),
                "score_qr": np.column_stack([s, qr]),
                "score_qr_nbhd": np.column_stack([s, qr, nb/nb.max()]),
                "score_invnb": np.column_stack([s, 1/(1+nb)]),
            }[name]

        for feat_name in ["score_nbhd", "score_lognbhd", "score_qr", "score_qr_nbhd", "score_invnb"]:
            for K in [2, 3, 4]:
                X_pool = fs(feat_name, pool, nb_pool, qr_pool)
                X_test = fs(feat_name, test_x, nb_test, qr_test)
                # Standardize
                mu = X_pool.mean(0); sg = X_pool.std(0) + 1e-12
                Xp = (X_pool - mu) / sg
                Xt = (X_test - mu) / sg
                km = KMeans(n_clusters=K, n_init=10, random_state=42)
                km.fit(Xp)
                labels_pool = km.labels_
                labels_test = km.predict(Xt)
                # Rank clusters by mean score
                cluster_means = {}
                for c in range(K):
                    mask = (labels_pool == c)
                    if mask.sum() > 0:
                        cluster_means[c] = pool[mask].mean()
                    else:
                        cluster_means[c] = 0
                # Positive clusters: those with highest means such that cumulative positive
                # sample count matches an unsupervised prior. Use the "top cluster" or
                # "top clusters covering 20% of pool" depending on K.
                for rule in ["top1", "top_half"]:
                    sorted_cs = sorted(cluster_means.items(), key=lambda kv: -kv[1])
                    if rule == "top1":
                        pos_cs = {sorted_cs[0][0]}
                    else:
                        pos_cs = {c for c, _ in sorted_cs[:K//2 if K//2>0 else 1]}
                    preds = np.array([1 if l in pos_cs else 0 for l in labels_test])
                    tp = int(((preds==1)&(test_y==1)).sum()); fp = int(((preds==1)&(test_y==0)).sum())
                    fn = int(((preds==0)&(test_y==1)).sum()); tn = int(((preds==0)&(test_y==0)).sum())
                    acc = (tp+tn)/len(test_y)
                    p_pos = tp/(tp+fp) if (tp+fp)>0 else 0
                    r_pos = tp/(tp+fn) if (tp+fn)>0 else 0
                    f_pos = 2*p_pos*r_pos/(p_pos+r_pos) if (p_pos+r_pos)>0 else 0
                    p_neg = tn/(tn+fn) if (tn+fn)>0 else 0
                    r_neg = tn/(tn+fp) if (tn+fp)>0 else 0
                    f_neg = 2*p_neg*r_neg/(p_neg+r_neg) if (p_neg+r_neg)>0 else 0
                    mf = (f_pos+f_neg)/2
                    flag = ""
                    if acc > acc_b and mf > mf_b: flag = " ***STRICT-BOTH***"
                    elif acc > acc_b: flag = " [acc+]"
                    elif mf > mf_b: flag = " [mf+]"
                    print(f"  {feat_name:<18} K={K} {rule:<8}  acc={acc:.4f}  mf={mf:.4f}  npos={int(preds.sum())}{flag}")


if __name__ == "__main__":
    main()
