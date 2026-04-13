"""GMM with K >= 2 components. Threshold via posterior-sum over the top-K-by-mean clusters.

For each K, fit GMM, sort components by mean. Try declaring the TOP k clusters as
positive, for k in 1..K-1. See if any combination strict-beats baseline.
"""
import os, sys, json, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.mixture import GaussianMixture

BASE = {"MHClip_EN": (0.7640, 0.6532), "MHClip_ZH": (0.8121, 0.7871)}


def load(dataset):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(dataset)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, train_arr, test_x, test_y


def run(dataset, fit_src_name, fit_src, test_x, test_y, acc_b, mf_b):
    for K in [2, 3, 4, 5, 6, 8, 10]:
        try:
            g = GaussianMixture(n_components=K, random_state=42, max_iter=500)
            g.fit(fit_src.reshape(-1, 1))
        except Exception as e:
            continue
        means = g.means_.flatten()
        order = np.argsort(means)  # ascending by mean
        # Test posteriors
        post_test = g.predict_proba(test_x.reshape(-1, 1))  # (n_test, K)
        for top_k in range(1, K):
            pos_clusters = set(order[-top_k:].tolist())
            # sum posterior for pos clusters
            pos_prob = post_test[:, list(pos_clusters)].sum(axis=1)
            preds = (pos_prob >= 0.5).astype(int)
            tp = int(((preds==1)&(test_y==1)).sum()); fp = int(((preds==1)&(test_y==0)).sum())
            fn = int(((preds==0)&(test_y==1)).sum()); tn = int(((preds==0)&(test_y==0)).sum())
            n = len(test_y); acc = (tp+tn)/n
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
            if flag:
                print(f"  fit={fit_src_name} K={K} top_k={top_k}  acc={acc:.4f}  mf={mf:.4f}  npos={int(preds.sum())}{flag}")


def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        pool, train_arr, test_x, test_y = load(dataset)
        print(f"\n=== {dataset} === baseline {acc_b:.4f}/{mf_b:.4f}")
        for name, src in [("pool", pool), ("train", train_arr), ("test", test_x)]:
            run(dataset, name, src, test_x, test_y, acc_b, mf_b)


if __name__ == "__main__":
    main()
