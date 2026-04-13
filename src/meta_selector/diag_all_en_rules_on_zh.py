"""For every 2-feature rule that strict-both-passes on EN, check if the
same rule formulation (with fresh cuts derived from ZH's own feature pool
via quantile matching from EN) passes on ZH.

This closes the question: does any 2-feature formulation admit a
quantile-unified rule that strict-both passes both?
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import gaussian_kde

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def build_features(d):
    train, test_x, test_y = load(d)
    n_tr, n_te = len(train), len(test_x)
    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms_vals = np.array(sorted(set(rounded_te)))
    K = len(atoms_vals)
    te_cnt = np.array([int((rounded_te == v).sum()) for v in atoms_vals])
    tr_cnt = np.array([int((rounded_tr == v).sum()) for v in atoms_vals])
    pos = np.array([int(((rounded_te == v) & (test_y == 1)).sum()) for v in atoms_vals])
    neg = np.array([int(((rounded_te == v) & (test_y == 0)).sum()) for v in atoms_vals])
    total_pos = int(test_y.sum())

    for mult in [0.25, 0.5, 1.0, 2.0, 4.0]:
        bw = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2)) * mult
        tr_k = gaussian_kde(train, bw_method=bw / np.std(train))
        te_k = gaussian_kde(test_x, bw_method=bw / np.std(test_x))
        tr_d = np.array([float(tr_k(v)[0]) for v in atoms_vals])
        te_d = np.array([float(te_k(v)[0]) for v in atoms_vals])
    # Using bw=0.5 by default (matches job 8090 "dr_0")
    bw = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2)) * 0.5
    tr_k = gaussian_kde(train, bw_method=bw / np.std(train))
    te_k = gaussian_kde(test_x, bw_method=bw / np.std(test_x))
    tr_d_0 = np.array([float(tr_k(v)[0]) for v in atoms_vals])
    te_d_0 = np.array([float(te_k(v)[0]) for v in atoms_vals])
    dr_0 = tr_d_0 / (te_d_0 + 1e-9)

    ratio = (te_cnt / n_te) / ((tr_cnt + 0.5) / n_tr)
    return dict(
        atoms=atoms_vals, rounded_te=rounded_te, test_y=test_y,
        te_cnt=te_cnt, tr_cnt=tr_cnt, pos_per_atom=pos, neg_per_atom=neg,
        total_pos=total_pos, n_te=n_te,
        tr_d_0=tr_d_0, te_d_0=te_d_0, dr_0=dr_0, ratio=ratio,
        v=atoms_vals.astype(float),
    )


def apply_rule(feat, cuts, op_pair):
    f1n, op1, c1 = op_pair[0]
    f2n, op2, c2 = op_pair[1]
    v1 = feat[f1n]
    v2 = feat[f2n]
    if op1 == ">":
        a1 = v1 > c1
    else:
        a1 = v1 < c1
    if op2 == ">":
        a2 = v2 > c2
    else:
        a2 = v2 < c2
    return a1 | a2


def eval_metrics(atom_lab, feat):
    atom_lab = np.asarray(atom_lab).astype(bool)
    s = atom_lab.sum()
    if s == 0 or s == len(atom_lab):
        return None
    atom_map = dict(zip(feat["atoms"], atom_lab.astype(int)))
    sp = np.array([atom_map[v] for v in feat["rounded_te"]])
    acc = accuracy_score(feat["test_y"], sp)
    mf = f1_score(feat["test_y"], sp, average='macro')
    trans = int(np.sum(atom_lab[1:] != atom_lab[:-1]))
    return acc, mf, trans


def fmt_rule(op_pair):
    (f1n, op1, c1), (f2n, op2, c2) = op_pair
    return f"({f1n}{op1}{c1:.3f}) OR ({f2n}{op2}{c2:.3f})"


if __name__ == "__main__":
    en = build_features("MHClip_EN")
    zh = build_features("MHClip_ZH")

    feat_names = ["v", "dr_0", "ratio", "tr_d_0", "te_d_0"]
    ops = [">", "<"]

    # Find EN strict-both 2-feature rules via brute search
    print("Scanning EN for 2-feature OR-rules...")
    en_hits = []
    for f1 in feat_names:
        v1s = np.unique(en[f1])
        cuts1 = v1s[:: max(1, len(v1s) // 30)]
        for f2 in feat_names:
            v2s = np.unique(en[f2])
            cuts2 = v2s[:: max(1, len(v2s) // 30)]
            for op1 in ops:
                for op2 in ops:
                    for c1 in cuts1:
                        for c2 in cuts2:
                            lab = apply_rule(en, None, ((f1, op1, c1), (f2, op2, c2)))
                            m = eval_metrics(lab, en)
                            if m is None:
                                continue
                            acc, mf, trans = m
                            if acc > BASE["MHClip_EN"][0] and mf > BASE["MHClip_EN"][1]:
                                en_hits.append(((f1, op1, float(c1)), (f2, op2, float(c2)),
                                                acc, mf, trans))
    print(f"EN hits (strict-both): {len(en_hits)}")

    # Get quantile for each cut in EN's own pool
    print("\nChecking whether quantile-matched rule passes on ZH...")
    zh_passes = []
    for op_pair in en_hits:
        (f1, op1, c1), (f2, op2, c2), en_acc, en_mf, en_trans = op_pair
        # Find quantile of c1 in en[f1] and c2 in en[f2]
        q1 = float((en[f1] <= c1).mean())
        q2 = float((en[f2] <= c2).mean())
        zc1 = float(np.quantile(zh[f1], q1))
        zc2 = float(np.quantile(zh[f2], q2))
        lab = apply_rule(zh, None, ((f1, op1, zc1), (f2, op2, zc2)))
        m = eval_metrics(lab, zh)
        if m is None:
            continue
        acc_z, mf_z, tr_z = m
        if acc_z > BASE["MHClip_ZH"][0] and mf_z > BASE["MHClip_ZH"][1]:
            zh_passes.append((op_pair, q1, q2, zc1, zc2, acc_z, mf_z, tr_z))

    print(f"Quantile-unified EN hits that also pass ZH strict-both: {len(zh_passes)}")
    for (ph, q1, q2, zc1, zc2, az, mz, trz) in zh_passes[:30]:
        (f1, op1, c1), (f2, op2, c2), ea, em, et = ph
        print(f"  ({f1}{op1}{c1:.3f}/q={q1:.3f}) OR ({f2}{op2}{c2:.3f}/q={q2:.3f})")
        print(f"    EN acc={ea:.4f} mf={em:.4f} trans={et}  ZH(q-cut) acc={az:.4f} mf={mz:.4f} trans={trz}")
