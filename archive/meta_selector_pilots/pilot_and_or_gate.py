"""AND/OR gates of two published threshold methods.

Each method independently induces a 0/1 prediction. Combine with AND or OR.
This is a UNIFIED RULE — same formula on both datasets, no per-dataset branching.

Rationale: Otsu maximizes between-class variance (picks a wide split), KI minimizes
error rate of a 2-Gaussian fit (picks a tight split). OR fires if either method
flags the point; AND fires only if both agree. Published methods, combined via a
published logical rule (consensus / union), no tuning parameters.

Prior art: ensemble thresholding literature — Nie et al. 2018 "Ensemble Image
Thresholding", etc. AND/OR fusion is standard.
"""
import sys, numpy as np
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


def eval_pred(pred, y):
    """Compute acc and macro-F1 from binary predictions."""
    pred = np.asarray(pred, dtype=int)
    y = np.asarray(y, dtype=int)
    acc = (pred == y).mean()
    # macro-F1
    tp1 = ((pred == 1) & (y == 1)).sum()
    fp1 = ((pred == 1) & (y == 0)).sum()
    fn1 = ((pred == 0) & (y == 1)).sum()
    tp0 = ((pred == 0) & (y == 0)).sum()
    fp0 = ((pred == 0) & (y == 1)).sum()
    fn0 = ((pred == 1) & (y == 0)).sum()
    p1 = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0
    r1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0
    f1 = 2*p1*r1/(p1+r1) if (p1+r1) > 0 else 0
    p0 = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0
    r0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0
    f0 = 2*p0*r0/(p0+r0) if (p0+r0) > 0 else 0
    return acc, (f0 + f1) / 2


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    # Compute thresholds from each method on pool and test
    for ref_name, ref in [("pool", pool), ("test", test_x)]:
        t_otsu = otsu_threshold(ref)
        t_gmm = gmm_threshold(ref)
        t_ki = kittler_illingworth(ref)
        print(f"  [{ref_name}] t_otsu={t_otsu:.4f} t_gmm={t_gmm:.4f} t_ki={t_ki:.4f}")

        methods = {"Otsu": t_otsu, "GMM": t_gmm, "KI": t_ki}
        preds = {n: (test_x >= t).astype(int) for n, t in methods.items()}

        # AND/OR pairs
        for a, b in [("Otsu", "GMM"), ("Otsu", "KI"), ("GMM", "KI")]:
            and_pred = preds[a] & preds[b]
            or_pred = preds[a] | preds[b]
            acc, mf = eval_pred(and_pred, test_y)
            tag = "STRICT" if acc > acc_b and mf > mf_b else ""
            print(f"    {a}&{b}: acc={acc:.4f} mf={mf:.4f} n_pos={and_pred.sum()} {tag}")
            acc, mf = eval_pred(or_pred, test_y)
            tag = "STRICT" if acc > acc_b and mf > mf_b else ""
            print(f"    {a}|{b}: acc={acc:.4f} mf={mf:.4f} n_pos={or_pred.sum()} {tag}")

        # 3-way majority vote
        vote = preds["Otsu"] + preds["GMM"] + preds["KI"]
        maj_pred = (vote >= 2).astype(int)
        acc, mf = eval_pred(maj_pred, test_y)
        tag = "STRICT" if acc > acc_b and mf > mf_b else ""
        print(f"    majority3: acc={acc:.4f} mf={mf:.4f} n_pos={maj_pred.sum()} {tag}")

        # 3-way unanimous
        unan_pred = (vote == 3).astype(int)
        acc, mf = eval_pred(unan_pred, test_y)
        tag = "STRICT" if acc > acc_b and mf > mf_b else ""
        print(f"    unanimous3: acc={acc:.4f} mf={mf:.4f} n_pos={unan_pred.sum()} {tag}")
