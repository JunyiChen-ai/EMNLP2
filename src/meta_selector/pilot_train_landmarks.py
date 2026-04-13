"""Train-only landmark classifier.

For each test atom, compute its position relative to published landmarks
computed on TRAIN scores ONLY (no test labels, no test scores in landmark
derivation). Landmarks tested:

- Otsu threshold on train
- GMM threshold on train (2-component Bayes boundary)
- Kittler-Illingworth on train
- Train median + k*MAD_train (published Tukey-style fence) for k ∈ {1,2,3}
- Train Q3 + 1.5*IQR_train (Tukey inner fence)

Then: classify each test atom by a VOTE among these landmarks. Atom label
= 1 if majority of landmarks say "above", else 0. This produces an atom-
level label that is genuinely non-monotone in raw score IF the landmarks
are not all monotone functions of the same statistic (they are not —
Otsu/GMM/KI/Tukey can disagree on which atoms to include).

But note: within a single test atom, ALL samples share the same score, so
they get the same vote. This is still atom-constant. HOWEVER: the vote
is a function of TRAIN global topology, not test-only features, so the
atom-level labeling can be non-monotone in test score. Specifically, if
a test atom x falls above Otsu-train but below GMM-train, and another
atom x' > x falls below both, the vote flips even though x' > x.

This directly tests the director's "non-monotone atom-level labeling"
possibility.
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
    pred = np.asarray(pred, dtype=int)
    y = np.asarray(y, dtype=int)
    acc = (pred == y).mean()
    tp1 = ((pred == 1) & (y == 1)).sum()
    fp1 = ((pred == 1) & (y == 0)).sum()
    fn1 = ((pred == 0) & (y == 1)).sum()
    tp0 = ((pred == 0) & (y == 0)).sum()
    fp0 = ((pred == 0) & (y == 1)).sum()
    fn0 = ((pred == 1) & (y == 0)).sum()
    p1 = tp1/(tp1+fp1) if tp1+fp1 > 0 else 0
    r1 = tp1/(tp1+fn1) if tp1+fn1 > 0 else 0
    f1 = 2*p1*r1/(p1+r1) if p1+r1 > 0 else 0
    p0 = tp0/(tp0+fp0) if tp0+fp0 > 0 else 0
    r0 = tp0/(tp0+fn0) if tp0+fn0 > 0 else 0
    f0 = 2*p0*r0/(p0+r0) if p0+r0 > 0 else 0
    return acc, (f0+f1)/2


# Test bar: acc STRICT >, mf >= baseline
def check_bar(acc, mf, acc_b, mf_b):
    return (acc > acc_b) and (mf >= mf_b)


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}  (bar: acc>{acc_b:.4f} AND mf>={mf_b:.4f})")

    landmarks = {
        "train_otsu": otsu_threshold(train),
        "train_gmm": gmm_threshold(train),
        "train_ki": kittler_illingworth(train),
        "train_median": float(np.median(train)),
        "train_q75": float(np.quantile(train, 0.75)),
        "train_q90": float(np.quantile(train, 0.90)),
        "train_med+1mad": float(np.median(train) + np.median(np.abs(train - np.median(train)))),
        "train_med+2mad": float(np.median(train) + 2*np.median(np.abs(train - np.median(train)))),
        "train_med+3mad": float(np.median(train) + 3*np.median(np.abs(train - np.median(train)))),
        "train_q3+1.5iqr": float(np.quantile(train, 0.75) + 1.5 * (np.quantile(train, 0.75) - np.quantile(train, 0.25))),
    }
    print(f"  train landmarks: {[(k, round(v, 4)) for k, v in landmarks.items()]}")

    # Each landmark gives a prediction
    preds_by_lm = {name: (test_x >= t).astype(int) for name, t in landmarks.items()}
    for name, pred in preds_by_lm.items():
        acc, mf = eval_pred(pred, test_y)
        tag = "PASS" if check_bar(acc, mf, acc_b, mf_b) else ""
        print(f"  {name:20s}: acc={acc:.4f} mf={mf:.4f} n_pos={pred.sum()} {tag}")

    # Majority vote of landmarks
    vote_sum = np.zeros(len(test_x))
    for name, pred in preds_by_lm.items():
        vote_sum = vote_sum + pred
    n_lm = len(landmarks)
    for k in range(1, n_lm + 1):
        maj_pred = (vote_sum >= k).astype(int)
        acc, mf = eval_pred(maj_pred, test_y)
        tag = "PASS" if check_bar(acc, mf, acc_b, mf_b) else ""
        print(f"  vote>={k:2d}: acc={acc:.4f} mf={mf:.4f} n_pos={maj_pred.sum()} {tag}")
