"""Otsu and Kittler-Illingworth (MET) at ALL unique score values of pool (fine-grained).

Does the fine-grained version of Otsu/MET land at sub-atom points that strict-beat?
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def fine_otsu(scores):
    s = np.sort(np.asarray(scores))
    n = len(s)
    best_t = s[0]; best = float("inf")
    for i in range(1, n):
        c0 = s[:i]; c1 = s[i:]
        if len(c0)==0 or len(c1)==0: continue
        w0 = len(c0)/n; w1 = len(c1)/n
        within = w0*c0.var() + w1*c1.var()
        if within < best:
            best = within; best_t = (c0[-1]+c1[0])/2
    return best_t


def fine_met(scores):
    s = np.sort(np.asarray(scores))
    n = len(s)
    best_t = s[0]; best = float("inf")
    for i in range(2, n-1):
        c0 = s[:i]; c1 = s[i:]
        if len(c0) < 2 or len(c1) < 2: continue
        w0 = len(c0)/n; w1 = len(c1)/n
        s0 = c0.std() + 1e-12
        s1 = c1.std() + 1e-12
        J = 1 + 2*(w0*np.log(s0)+w1*np.log(s1)) - 2*(w0*np.log(w0)+w1*np.log(w1))
        if J < best:
            best = J; best_t = (c0[-1]+c1[0])/2
    return best_t


def load(dataset):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(dataset)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, train_arr, test_x, test_y


def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        pool, train_arr, test_x, test_y = load(dataset)
        print(f"\n=== {dataset} === baseline {acc_b:.4f}/{mf_b:.4f}")
        for name, src in [("pool", pool), ("train", train_arr), ("test", test_x)]:
            t_otsu = fine_otsu(src)
            t_met = fine_met(src)
            m_o = metrics(test_x, test_y, t_otsu)
            m_m = metrics(test_x, test_y, t_met)
            o_f = ""
            if m_o["acc"] > acc_b and m_o["mf"] > mf_b: o_f = " STRICT"
            m_f = ""
            if m_m["acc"] > acc_b and m_m["mf"] > mf_b: m_f = " STRICT"
            print(f"  {name:>5}  fine-Otsu t={t_otsu:.10f}  acc={m_o['acc']:.4f}  mf={m_o['mf']:.4f}{o_f}")
            print(f"  {name:>5}  fine-MET  t={t_met:.10f}  acc={m_m['acc']:.4f}  mf={m_m['mf']:.4f}{m_f}")


if __name__ == "__main__":
    main()
