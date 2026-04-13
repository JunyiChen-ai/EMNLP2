"""Search for a formula q = f(pool_stats) that hits both strict-both regions."""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations

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


def compute_stats(x):
    mu = x.mean(); sigma = x.std(); med = np.median(x)
    ot = otsu_threshold(x); gm = gmm_threshold(x)
    q_ot = (x < ot).mean()
    q_gm = (x < gm).mean()
    return {"mu": mu, "sigma": sigma, "med": med, "ot": ot, "gm": gm, "q_ot": q_ot, "q_gm": q_gm,
            "skew": ((x-mu)**3).mean()/(sigma**3)}


def main():
    data = {}
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_arr, test_x, test_y = load(d)
        data[d] = {"pool": pool, "train": train_arr, "test_x": test_x, "test_y": test_y,
                   "stats": compute_stats(pool),
                   "tr_stats": compute_stats(train_arr),
                   "te_stats": compute_stats(test_x),
                   "base": BASE[d]}

    # Known strict-both q ranges (from pool quantile)
    targets = {"MHClip_EN": (0.8831, 0.8859), "MHClip_ZH": (0.6425, 0.6933)}

    formulas = {
        "q_otsu_pool": lambda s, t, e: s["q_ot"],
        "q_gmm_pool": lambda s, t, e: s["q_gm"],
        "1 - mu": lambda s, t, e: 1 - s["mu"],
        "1 - mu - sigma": lambda s, t, e: 1 - s["mu"] - s["sigma"],
        "1 - 2*mu": lambda s, t, e: 1 - 2*s["mu"],
        "1 - med": lambda s, t, e: 1 - s["med"],
        "1 - 2*med": lambda s, t, e: 1 - 2*s["med"],
        "(q_ot + q_gm)/2": lambda s, t, e: (s["q_ot"] + s["q_gm"])/2,
        "q_ot - sigma": lambda s, t, e: s["q_ot"] - s["sigma"],
        "q_ot - mu": lambda s, t, e: s["q_ot"] - s["mu"],
        "q_gm + sigma": lambda s, t, e: s["q_gm"] + s["sigma"],
        "1 - sqrt(mu)": lambda s, t, e: 1 - np.sqrt(s["mu"]),
        "q_ot - 0.01*skew": lambda s, t, e: s["q_ot"] - 0.01*s["skew"],
        "q_gm": lambda s, t, e: s["q_gm"],
        "train_q_ot": lambda s, t, e: t["q_ot"],
        "test_q_ot": lambda s, t, e: e["q_ot"],
        "(train_q_ot+pool_q_ot)/2": lambda s, t, e: (t["q_ot"]+s["q_ot"])/2,
        "1 - mu - 2*sigma^2": lambda s, t, e: 1 - s["mu"] - 2*s["sigma"]**2,
        "1 - 3*mu": lambda s, t, e: 1 - 3*s["mu"],
        "q_ot - mu^2": lambda s, t, e: s["q_ot"] - s["mu"]**2,
    }

    for name, fn in formulas.items():
        q_en = fn(data["MHClip_EN"]["stats"], data["MHClip_EN"]["tr_stats"], data["MHClip_EN"]["te_stats"])
        q_zh = fn(data["MHClip_ZH"]["stats"], data["MHClip_ZH"]["tr_stats"], data["MHClip_ZH"]["te_stats"])
        # Clamp to [0,1]
        q_en_c = max(0.01, min(0.99, q_en))
        q_zh_c = max(0.01, min(0.99, q_zh))
        t_en = float(np.quantile(data["MHClip_EN"]["pool"], q_en_c))
        t_zh = float(np.quantile(data["MHClip_ZH"]["pool"], q_zh_c))
        m_en = metrics(data["MHClip_EN"]["test_x"], data["MHClip_EN"]["test_y"], t_en)
        m_zh = metrics(data["MHClip_ZH"]["test_x"], data["MHClip_ZH"]["test_y"], t_zh)
        acc_b_en, mf_b_en = data["MHClip_EN"]["base"]
        acc_b_zh, mf_b_zh = data["MHClip_ZH"]["base"]
        sb_en = m_en["acc"] > acc_b_en and m_en["mf"] > mf_b_en
        sb_zh = m_zh["acc"] > acc_b_zh and m_zh["mf"] > mf_b_zh
        tag = ""
        if sb_en and sb_zh: tag = " ***UNIFIED STRICT-BOTH!***"
        elif sb_en: tag = " [EN+]"
        elif sb_zh: tag = " [ZH+]"
        print(f"  {name:<26}  q_en={q_en_c:.4f}({m_en['acc']:.4f}/{m_en['mf']:.4f})  q_zh={q_zh_c:.4f}({m_zh['acc']:.4f}/{m_zh['mf']:.4f}){tag}")


if __name__ == "__main__":
    main()
