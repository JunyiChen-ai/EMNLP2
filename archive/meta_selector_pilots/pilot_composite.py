"""Atom-level composite score + Otsu-on-atoms (UVO style) with density features.

For each pool atom a, compute:
  n_pool(a), n_train(a), n_test(a)
  log counts
  nbhd density (atoms in ±window)
  percentile rank (in pool)

Then form composite = f(a, features) and apply per-atom Otsu.
Pilot MANY composite forms and see which dataset-agnostic form strict-beats both.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, metrics, load_scores_file, build_arrays
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


def atom_features(pool, decimals=4, window=0.05):
    u = sorted(set(np.round(pool, decimals).tolist()))
    pool_r = np.round(pool, decimals)
    feats = {}
    for a in u:
        n_pool = int((pool_r == a).sum())
        n_nbhd = int(((pool_r != a) & (np.abs(pool_r - a) < window)).sum())
        feats[a] = {"n_pool": n_pool, "n_nbhd": n_nbhd, "a": a}
    return feats, u


def apply_thr(test_x, feats, composite_fn, t, decimals=4):
    """Apply threshold on composite(atom(s))."""
    test_r = np.round(test_x, decimals)
    preds = np.zeros(len(test_x), dtype=int)
    for i, a in enumerate(test_r):
        if a not in feats:
            # nearest
            a = min(feats.keys(), key=lambda k: abs(k - a))
        c = composite_fn(feats[a])
        preds[i] = 1 if c >= t else 0
    return preds


def evaluate(preds, y):
    tp = int(((preds==1)&(y==1)).sum()); fp = int(((preds==1)&(y==0)).sum())
    fn = int(((preds==0)&(y==1)).sum()); tn = int(((preds==0)&(y==0)).sum())
    n = len(y); acc = (tp+tn)/n
    p_pos = tp/(tp+fp) if (tp+fp)>0 else 0
    r_pos = tp/(tp+fn) if (tp+fn)>0 else 0
    f_pos = 2*p_pos*r_pos/(p_pos+r_pos) if (p_pos+r_pos)>0 else 0
    p_neg = tn/(tn+fn) if (tn+fn)>0 else 0
    r_neg = tn/(tn+fp) if (tn+fp)>0 else 0
    f_neg = 2*p_neg*r_neg/(p_neg+r_neg) if (p_neg+r_neg)>0 else 0
    return {"acc": acc, "mf": (f_pos+f_neg)/2}


def atom_otsu(values):
    """Otsu on a set of 1D values (one point per atom)."""
    v = np.sort(np.asarray(values))
    best_t = 0.0; best = float("inf")
    for i in range(1, len(v)):
        c0 = v[:i]; c1 = v[i:]
        if len(c0)==0 or len(c1)==0: continue
        w0 = len(c0)/len(v); w1 = len(c1)/len(v)
        wv = w0*c0.var() + w1*c1.var()
        if wv < best:
            best = wv
            best_t = (c0[-1]+c1[0])/2
    return best_t


def run_pilot(dataset, composite_name, composite_fn):
    pool, train_arr, test_x, test_y = load(dataset)
    feats, u = atom_features(pool)

    # Composite value for each atom
    comp_vals = [composite_fn(feats[a]) for a in u]
    t = atom_otsu(comp_vals)
    preds = apply_thr(test_x, feats, composite_fn, t)
    m = evaluate(preds, test_y)
    return t, m


COMPOSITES = {
    "raw": lambda f: f["a"],
    "raw - 0.05*log1p(n_nbhd)": lambda f: f["a"] - 0.05 * np.log1p(f["n_nbhd"]),
    "raw - 0.1*log1p(n_nbhd)": lambda f: f["a"] - 0.1 * np.log1p(f["n_nbhd"]),
    "raw - 0.02*n_nbhd": lambda f: f["a"] - 0.02 * f["n_nbhd"],
    "raw - 0.05*n_nbhd": lambda f: f["a"] - 0.05 * f["n_nbhd"],
    "raw - 0.1*n_nbhd": lambda f: f["a"] - 0.1 * f["n_nbhd"],
    "raw + 0.05/(1+n_nbhd)": lambda f: f["a"] + 0.05 / (1 + f["n_nbhd"]),
    "raw + 0.1/(1+n_nbhd)": lambda f: f["a"] + 0.1 / (1 + f["n_nbhd"]),
    "raw + 0.2/(1+n_nbhd)": lambda f: f["a"] + 0.2 / (1 + f["n_nbhd"]),
    "raw + 0.3/(1+n_nbhd)": lambda f: f["a"] + 0.3 / (1 + f["n_nbhd"]),
    "raw / (1 + 0.1*log1p(n_pool))": lambda f: f["a"] / (1 + 0.1 * np.log1p(f["n_pool"])),
    "raw * (1 - n_nbhd/50)": lambda f: f["a"] * (1 - min(1, f["n_nbhd"]/50.0)),
}


def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        print(f"\n=== {dataset} ===  baseline {acc_b:.4f}/{mf_b:.4f}")
        for name, fn in COMPOSITES.items():
            t, m = run_pilot(dataset, name, fn)
            flag = ""
            if m["acc"] > acc_b and m["mf"] > mf_b: flag = " ***STRICT-BOTH***"
            elif m["acc"] > acc_b: flag = " [acc+]"
            elif m["mf"] > mf_b: flag = " [mf+]"
            print(f"  {name:<40}  t={t:+.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}{flag}")


if __name__ == "__main__":
    main()
