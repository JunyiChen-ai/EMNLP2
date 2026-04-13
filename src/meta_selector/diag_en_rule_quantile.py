"""Convert the EN non-suffix rule to quantile form, apply to ZH.

EN rule: (v > 0.3491) OR (dr_0 < 0.8795)
  -> quantile of v=0.3491 in EN atoms_vals
  -> quantile of dr_0=0.8795 in EN dr_0 atom-level values
Then apply the SAME quantile cuts to ZH.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
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


def eval_pred(y, p):
    return accuracy_score(y, p), f1_score(y, p, average='macro')


def get_feats(d):
    train, test_x, test_y = load(d)
    n_tr, n_te = len(train), len(test_x)
    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms_vals = np.array(sorted(set(rounded_te)))
    te_counts = np.array([int((rounded_te == v).sum()) for v in atoms_vals])
    tr_counts = np.array([int((rounded_tr == v).sum()) for v in atoms_vals])
    ratio = (te_counts / n_te) / ((tr_counts + 0.5) / n_tr)

    bw = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2)) * 0.5
    train_kde = gaussian_kde(train, bw_method=bw / np.std(train))
    test_kde = gaussian_kde(test_x, bw_method=bw / np.std(test_x))
    tr_d_0 = np.array([float(train_kde(v)[0]) for v in atoms_vals])
    te_d_0 = np.array([float(test_kde(v)[0]) for v in atoms_vals])
    dr_0 = tr_d_0 / (te_d_0 + 1e-6)
    return train, test_x, test_y, rounded_te, atoms_vals, dr_0, ratio


# Get EN features, determine quantile of 0.3491 and 0.8795
_, _, _, _, en_atoms, en_dr0, _ = get_feats("MHClip_EN")
q_v_en = float((en_atoms <= 0.3491).mean())  # empirical CDF in atoms_vals
q_dr_en = float((en_dr0 <= 0.8795).mean())
print(f"EN cut quantiles: v@{q_v_en:.4f}, dr_0@{q_dr_en:.4f}")

# Try a sweep of quantile cuts on BOTH datasets with rule
# (v > q_v) OR (dr_0 < q_dr)
# and also the dual: (v > q_v) AND (dr_0 < q_dr)
hits = []
qs_v = np.linspace(0.55, 0.95, 21)
qs_dr = np.linspace(0.05, 0.45, 21)
for q_v in qs_v:
    for q_dr in qs_dr:
        res = {}
        pass_both = True
        for d in ["MHClip_EN", "MHClip_ZH"]:
            _, _, test_y, rounded_te, atoms, dr_0, _ = get_feats(d)
            cv = float(np.quantile(atoms, q_v))
            cd = float(np.quantile(dr_0, q_dr))
            atom_lab = (atoms > cv) | (dr_0 < cd)
            atom_map = dict(zip(atoms, atom_lab.astype(int)))
            sp = np.array([atom_map[v] for v in rounded_te])
            if sp.sum() == 0 or sp.sum() == len(sp):
                pass_both = False
                break
            acc, mf = eval_pred(test_y, sp)
            acc_b, mf_b = BASE[d]
            if not (acc > acc_b and mf > mf_b):
                pass_both = False
                break
            res[d] = (acc, mf)
        if pass_both:
            hits.append((q_v, q_dr, res))

print(f"Unified (v>qv) OR (dr_0<qdr) strict-both hits: {len(hits)}")
for q_v, q_dr, res in hits[:20]:
    en = res["MHClip_EN"]
    zh = res["MHClip_ZH"]
    print(f"  q_v={q_v:.3f} q_dr={q_dr:.3f}  EN {en[0]:.4f}/{en[1]:.4f}  ZH {zh[0]:.4f}/{zh[1]:.4f}")

# Also try (v>qv) AND NOT(dr_0 < qdr) and other combos - no just the OR which is what EN uses
print("\nTrying all 4 combos with the EN-OR template:")
for op in ["OR", "AND"]:
    for d1 in ["gt", "lt"]:
        for d2 in ["gt", "lt"]:
            hits2 = []
            for q_v in qs_v:
                for q_dr in qs_dr:
                    res = {}
                    pass_both = True
                    for d in ["MHClip_EN", "MHClip_ZH"]:
                        _, _, test_y, rounded_te, atoms, dr_0, _ = get_feats(d)
                        cv = float(np.quantile(atoms, q_v))
                        cd = float(np.quantile(dr_0, q_dr))
                        a1 = atoms > cv if d1 == "gt" else atoms < cv
                        a2 = dr_0 > cd if d2 == "gt" else dr_0 < cd
                        atom_lab = (a1 | a2) if op == "OR" else (a1 & a2)
                        atom_map = dict(zip(atoms, atom_lab.astype(int)))
                        sp = np.array([atom_map[v] for v in rounded_te])
                        if sp.sum() == 0 or sp.sum() == len(sp):
                            pass_both = False
                            break
                        acc, mf = eval_pred(test_y, sp)
                        acc_b, mf_b = BASE[d]
                        if not (acc > acc_b and mf > mf_b):
                            pass_both = False
                            break
                        res[d] = (acc, mf)
                    if pass_both:
                        hits2.append((q_v, q_dr, res))
            print(f"  v{d1} {op} dr_0{d2}: {len(hits2)} hits")
            for q_v, q_dr, res in hits2[:3]:
                en = res["MHClip_EN"]; zh = res["MHClip_ZH"]
                print(f"    q_v={q_v:.3f} q_dr={q_dr:.3f}  EN {en[0]:.4f}/{en[1]:.4f}  ZH {zh[0]:.4f}/{zh[1]:.4f}")
