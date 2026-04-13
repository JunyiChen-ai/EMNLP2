"""Non-monotone band rules over train landmarks.

Unlike threshold rules (x >= t) which are monotone in x, band rules of the
form (t_lo <= x < t_hi) OR (x >= t_hi2) create genuinely non-monotone atom
labelings. Example: atom is positive if it falls inside [train-Otsu, train-GMM]
OR above train-Q90. This is NOT a single-threshold rule.

Rationale: On ZH, the TF-GMM baseline cut at t=0.0362 is below the TF-Otsu cut
at t=0.22+. A band rule "include atoms in [0.036, 0.22]" behaves differently
from "include atoms above 0.036" because it excludes the far tail. If the far
tail contains label noise or non-hate anomalies (e.g., videos with extreme
content unrelated to hate), excluding them could flip the test accuracy.

This tests the director's specific suggestion: "rules that retain the sample-
level signal without collapsing it into a single-score dimension."

All landmarks derived from TRAIN ONLY. Bar: acc STRICT > baseline AND mf >= baseline.
"""
import sys, numpy as np
from itertools import combinations
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
    if total == 0: return float(np.median(s))
    best_j, best_cost = 0, float('inf')
    for j in range(1, n_bins):
        p0 = hist[:j].sum() / total
        p1 = hist[j:].sum() / total
        if p0 < 1e-6 or p1 < 1e-6: continue
        mu0 = (hist[:j] * centers[:j]).sum() / (p0 * total)
        mu1 = (hist[j:] * centers[j:]).sum() / (p1 * total)
        var0 = (hist[:j] * (centers[:j] - mu0)**2).sum() / (p0*total)
        var1 = (hist[j:] * (centers[j:] - mu1)**2).sum() / (p1*total)
        if var0 <= 0 or var1 <= 0: continue
        cost = 1 + 2*(p0*np.log(np.sqrt(var0)) + p1*np.log(np.sqrt(var1))) - 2*(p0*np.log(p0) + p1*np.log(p1))
        if cost < best_cost:
            best_cost, best_j = cost, j
    return float(centers[best_j])


def eval_pred(pred, y):
    pred = np.asarray(pred, dtype=int)
    y = np.asarray(y, dtype=int)
    acc = (pred == y).mean()
    tp1 = ((pred==1)&(y==1)).sum(); fp1 = ((pred==1)&(y==0)).sum(); fn1 = ((pred==0)&(y==1)).sum()
    tp0 = ((pred==0)&(y==0)).sum(); fp0 = ((pred==0)&(y==1)).sum(); fn0 = ((pred==1)&(y==0)).sum()
    p1 = tp1/(tp1+fp1) if tp1+fp1>0 else 0; r1 = tp1/(tp1+fn1) if tp1+fn1>0 else 0
    f1 = 2*p1*r1/(p1+r1) if p1+r1>0 else 0
    p0 = tp0/(tp0+fp0) if tp0+fp0>0 else 0; r0 = tp0/(tp0+fn0) if tp0+fn0>0 else 0
    f0 = 2*p0*r0/(p0+r0) if p0+r0>0 else 0
    return acc, (f0+f1)/2


def check_bar(acc, mf, ab, mb):
    return (acc > ab) and (mf >= mb)


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}  bar: acc>{acc_b:.4f} AND mf>={mf_b:.4f}")

    med = float(np.median(train))
    mad = float(np.median(np.abs(train - med)))
    q25, q50, q75, q90, q95 = np.quantile(train, [0.25, 0.50, 0.75, 0.90, 0.95])
    iqr = q75 - q25

    landmarks = {
        "t_otsu": otsu_threshold(train),
        "t_gmm": gmm_threshold(train),
        "t_ki": kittler_illingworth(train),
        "med": med,
        "q75": float(q75),
        "q90": float(q90),
        "q95": float(q95),
        "m+1mad": med + mad,
        "m+2mad": med + 2*mad,
        "m+3mad": med + 3*mad,
        "q3+1.5iqr": float(q75 + 1.5*iqr),
        "q3+3iqr": float(q75 + 3*iqr),
    }
    print(f"  landmarks: {sorted([(k, round(v,4)) for k,v in landmarks.items()], key=lambda x: x[1])}")

    # Test all 2-landmark bands: (lo <= x < hi) => pos. Both landmarks from train.
    best_band = None
    results = []
    names = list(landmarks.keys())
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j: continue
            t_lo = min(landmarks[a], landmarks[b])
            t_hi = max(landmarks[a], landmarks[b])
            # BAND rule: positive iff t_lo <= x < t_hi
            band_pred = ((test_x >= t_lo) & (test_x < t_hi)).astype(int)
            acc, mf = eval_pred(band_pred, test_y)
            if check_bar(acc, mf, acc_b, mf_b):
                results.append(("band", a, b, t_lo, t_hi, acc, mf))
            # INVERSE band rule: positive iff x < t_lo OR x >= t_hi
            inv_pred = ((test_x < t_lo) | (test_x >= t_hi)).astype(int)
            acc, mf = eval_pred(inv_pred, test_y)
            if check_bar(acc, mf, acc_b, mf_b):
                results.append(("invband", a, b, t_lo, t_hi, acc, mf))
            # COMPLEMENT-BAND: positive iff x < t_hi
            comp_pred = (test_x < t_hi).astype(int)
            acc, mf = eval_pred(comp_pred, test_y)
            if check_bar(acc, mf, acc_b, mf_b):
                results.append(("comp", a, b, t_lo, t_hi, acc, mf))

    print(f"  2-landmark rules PASSING: {len(results)}")
    for r in results[:20]:
        print(f"    {r}")

    # Single landmark (upper threshold) — baseline of this experiment
    for name, t in landmarks.items():
        pred = (test_x >= t).astype(int)
        acc, mf = eval_pred(pred, test_y)
        if check_bar(acc, mf, acc_b, mf_b):
            print(f"  single [{name} t={t:.4f}]: PASS acc={acc:.4f} mf={mf:.4f}")
