"""Transform scores (log/power/yeo-johnson) then apply Otsu/GMM/MET.

Hypothesis: EN and ZH have different shape (kurt 8 vs 22, skew 2.3 vs 4.0).
A Gaussianizing transform collapses the shape difference so a single unsupervised
threshold rule (Otsu, GMM, or a fixed quantile) works on BOTH.
"""
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


def yeo_johnson(x, lam):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    if abs(lam) < 1e-12:
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1, lam) - 1) / lam
    if abs(lam - 2) < 1e-12:
        out[~pos] = -np.log1p(-x[~pos])
    else:
        out[~pos] = -(np.power(-x[~pos] + 1, 2 - lam) - 1) / (2 - lam)
    return out


def fit_yj_lam(x):
    """Maximum likelihood lambda for Yeo-Johnson (grid search)."""
    best_lam = 0.0
    best_ll = -np.inf
    for lam in np.arange(-2, 2.01, 0.05):
        xt = yeo_johnson(x, lam)
        mu = xt.mean()
        var = xt.var()
        if var <= 0:
            continue
        n = len(x)
        ll = -0.5 * n * np.log(var)
        jac = (lam - 1) * np.sum(np.sign(x) * np.log(np.abs(x) + 1))
        ll += jac
        if ll > best_ll:
            best_ll = ll
            best_lam = lam
    return best_lam


def main():
    data = {}
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_arr, test_x, test_y = load(d)
        data[d] = {"pool": pool, "train": train_arr, "test_x": test_x, "test_y": test_y,
                   "base": BASE[d]}

    EPS = 1e-4

    transforms = {
        "identity": lambda x: x,
        "log(x+1e-6)": lambda x: np.log(x + 1e-6),
        "log(x+1e-4)": lambda x: np.log(x + EPS),
        "log(x+1e-2)": lambda x: np.log(x + 1e-2),
        "log1p(x)": lambda x: np.log1p(x),
        "sqrt(x)": lambda x: np.sqrt(np.clip(x, 0, None)),
        "x^0.25": lambda x: np.power(np.clip(x, 1e-12, None), 0.25),
        "x^(1/3)": lambda x: np.cbrt(x),
        "logit(clip)": lambda x: np.log((np.clip(x, 1e-4, 1 - 1e-4)) / (1 - np.clip(x, 1e-4, 1 - 1e-4))),
        "asinh(x)": lambda x: np.arcsinh(x),
        "asinh(10x)": lambda x: np.arcsinh(10 * x),
        "log(x+0.1)": lambda x: np.log(x + 0.1),
    }

    for tname, T in transforms.items():
        print(f"\n=== transform: {tname} ===")
        for src_name in ["pool", "train", "test"]:
            pts_hits = []
            rows = []
            for d in ["MHClip_EN", "MHClip_ZH"]:
                pool = data[d]["pool"]
                train_arr = data[d]["train"]
                test_x = data[d]["test_x"]
                test_y = data[d]["test_y"]
                src = {"pool": pool, "train": train_arr, "test": test_x}[src_name]
                try:
                    src_t = T(src)
                    test_t = T(test_x)
                except Exception as e:
                    rows.append((d, "ERR", np.nan, np.nan))
                    continue
                # Otsu on transformed src
                try:
                    ot_t = otsu_threshold(src_t)
                    gm_t = gmm_threshold(src_t)
                except Exception as e:
                    rows.append((d, "ERR2", np.nan, np.nan))
                    continue
                m_o = metrics(test_t, test_y, ot_t)
                m_g = metrics(test_t, test_y, gm_t)
                acc_b, mf_b = data[d]["base"]
                sb_o = m_o["acc"] > acc_b and m_o["mf"] > mf_b
                sb_g = m_g["acc"] > acc_b and m_g["mf"] > mf_b
                tag_o = " ***" if sb_o else ""
                tag_g = " ***" if sb_g else ""
                rows.append((d, "Otsu", m_o["acc"], m_o["mf"], tag_o, ot_t))
                rows.append((d, "GMM", m_g["acc"], m_g["mf"], tag_g, gm_t))
            # Report
            for row in rows:
                d, method = row[0], row[1]
                if method == "ERR" or method == "ERR2":
                    print(f"  {src_name:<6} {d:<10} {method}")
                else:
                    _, _, acc, mf, tag, tt = row
                    print(f"  {src_name:<6} {d:<10} {method:<5}  t={tt:.6f}  acc={acc:.4f}  mf={mf:.4f}{tag}")
            # Check unified strict-both
            en_o = rows[0]; en_g = rows[1]; zh_o = rows[2]; zh_g = rows[3]
            for me_en, me_zh, mname in [(en_o, zh_o, "Otsu"), (en_g, zh_g, "GMM")]:
                if me_en[1] == "ERR" or me_zh[1] == "ERR":
                    continue
                _, _, acc_en, mf_en, tag_en, _ = me_en
                _, _, acc_zh, mf_zh, tag_zh, _ = me_zh
                acc_b_en, mf_b_en = BASE["MHClip_EN"]
                acc_b_zh, mf_b_zh = BASE["MHClip_ZH"]
                if acc_en > acc_b_en and mf_en > mf_b_en and acc_zh > acc_b_zh and mf_zh > mf_b_zh:
                    print(f"    ** UNIFIED STRICT-BOTH via {tname}+{mname}+{src_name} **")


if __name__ == "__main__":
    main()
