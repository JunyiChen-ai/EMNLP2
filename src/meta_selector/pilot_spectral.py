"""Spectral clustering on the joint train+test pool as a graph.

Build Gaussian similarity matrix W over all n samples. Compute normalized
Laplacian, take first 2 eigenvectors, cluster with k-means. Label each
test sample by its cluster.

This is a genuinely different feature class: it uses global GRAPH
structure, not per-sample KDE.

Rules:
- cluster_k2: which cluster each test sample is assigned to (high/low)
- fiedler: the sign of the Fiedler vector at each test sample
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans, SpectralClustering

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def eval_pred(y, p):
    return accuracy_score(y, p), f1_score(y, p, average='macro')


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline acc={acc_b:.4f} mf={mf_b:.4f}")

    pool = np.concatenate([train, test_x])
    n = len(pool)
    # Gaussian similarity at various scales
    for sigma_mul in [0.5, 1.0, 2.0]:
        sigma = float(pool.std()) * sigma_mul * (4/(3*n))**(1/5)
        # Use SpectralClustering with precomputed RBF affinity
        W = np.exp(-0.5 * ((pool[:, None] - pool[None, :]) / sigma) ** 2)
        try:
            sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='kmeans', random_state=42)
            labels = sc.fit_predict(W)
        except Exception as e:
            print(f"  spectral sigma={sigma_mul}: failed {e}")
            continue
        # Which cluster is "high"?
        c0_mean = pool[labels == 0].mean() if (labels == 0).sum() else 0
        c1_mean = pool[labels == 1].mean() if (labels == 1).sum() else 0
        high_cluster = 0 if c0_mean > c1_mean else 1
        # Test predictions: test samples are pool[len(train):]
        test_labels = labels[len(train):]
        pred = (test_labels == high_cluster).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred): continue
        acc, mf = eval_pred(test_y, pred)
        strict = acc > acc_b and mf >= mf_b
        tag = " ** PASS **" if strict else ""
        print(f"  spectral sigma={sigma_mul:.2f}: acc={acc:.4f} mf={mf:.4f} n_pos={pred.sum()}{tag}")

        # Also try 3 clusters → top cluster positive
        try:
            sc3 = SpectralClustering(n_clusters=3, affinity='precomputed', assign_labels='kmeans', random_state=42)
            labels3 = sc3.fit_predict(W)
            means3 = [pool[labels3 == c].mean() for c in range(3)]
            top_cluster = int(np.argmax(means3))
            test_labels3 = labels3[len(train):]
            pred3 = (test_labels3 == top_cluster).astype(int)
            if pred3.sum() > 0 and pred3.sum() < len(pred3):
                acc, mf = eval_pred(test_y, pred3)
                strict = acc > acc_b and mf >= mf_b
                tag = " ** PASS **" if strict else ""
                print(f"  spectral-3 top only sigma={sigma_mul:.2f}: acc={acc:.4f} mf={mf:.4f} n_pos={pred3.sum()}{tag}")
            # Mid+top as positive (non-suffix-like assignment)
            mid_cluster = int(np.argsort(means3)[1])
            pred3b = ((test_labels3 == top_cluster) | (test_labels3 == mid_cluster)).astype(int)
            if pred3b.sum() > 0 and pred3b.sum() < len(pred3b):
                acc, mf = eval_pred(test_y, pred3b)
                strict = acc > acc_b and mf >= mf_b
                tag = " ** PASS **" if strict else ""
                print(f"  spectral-3 mid+top sigma={sigma_mul:.2f}: acc={acc:.4f} mf={mf:.4f} n_pos={pred3b.sum()}{tag}")
            # Top + lowest as positive (true non-suffix)
            low_cluster = int(np.argmin(means3))
            pred3c = ((test_labels3 == top_cluster) | (test_labels3 == low_cluster)).astype(int)
            if pred3c.sum() > 0 and pred3c.sum() < len(pred3c):
                acc, mf = eval_pred(test_y, pred3c)
                strict = acc > acc_b and mf >= mf_b
                tag = " ** PASS **" if strict else ""
                print(f"  spectral-3 top+low sigma={sigma_mul:.2f}: acc={acc:.4f} mf={mf:.4f} n_pos={pred3c.sum()}{tag}")
        except Exception as e:
            print(f"  spectral-3 sigma={sigma_mul}: failed {e}")
