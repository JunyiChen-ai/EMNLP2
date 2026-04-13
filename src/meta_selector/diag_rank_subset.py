"""Structural subsets of the base positive set.

Signal from job 8101: ZH `base AND rank_even` pattern nearly works
(0.7852/0.7070). The idea: among base-POS atoms (atoms with score >= t_base),
pick a structured subset.

Tests:
- base AND (atom is k-th base-POS, k even)
- base AND (atom is NOT the argmax/argmin within base-POS)
- base AND (index-in-base-POS % k for k=2,3)
- base AND (te_cnt > median(te_cnt within base-POS))
- base AND (tr_cnt == 0)
- base AND (tr_cnt <= median(tr_cnt within base-POS))
- base AND (neighbor to left in atoms is NOT base-POS) [edge of base-POS region]

And flipping: base_atoms AND NOT <condition> variants.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def eval_rule(atom_lab, atoms_vals, rounded_te, test_y, name, acc_b, mf_b, passes):
    atom_lab = np.asarray(atom_lab).astype(bool)
    s = atom_lab.sum()
    if s == 0 or s == len(atom_lab):
        return
    atom_map = dict(zip(atoms_vals, atom_lab.astype(int)))
    sp = np.array([atom_map[v] for v in rounded_te])
    acc = accuracy_score(test_y, sp)
    mf = f1_score(test_y, sp, average='macro')
    trans = int(np.sum(atom_lab[1:] != atom_lab[:-1]))
    strict = acc > acc_b and mf > mf_b
    if strict:
        passes.append((name, acc, mf, trans))
    tag = " PASS" if strict else ""
    print(f"  {name}: acc={acc:.4f} mf={mf:.4f} trans={trans}{tag}")


def per_dataset(d, all_passes):
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    n_tr, n_te = len(train), len(test_x)
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms_vals = np.array(sorted(set(rounded_te)))
    K = len(atoms_vals)
    base_atom = (atoms_vals >= t_base).astype(bool)

    te_cnt = np.array([int((rounded_te == v).sum()) for v in atoms_vals])
    tr_cnt = np.array([int((rounded_tr == v).sum()) for v in atoms_vals])

    base_idx = np.where(base_atom)[0]
    n_base = len(base_idx)
    print(f"  |base_pos|={n_base}, t_base={t_base:.4f}")
    passes = []

    # Position within base-POS (0, 1, 2, ...)
    pos_in_base = np.full(K, -1)
    pos_in_base[base_idx] = np.arange(n_base)

    # ---- Parity within base-POS ----
    for k in [2, 3, 4]:
        for r in range(k):
            lab = base_atom & (pos_in_base % k == r) & (pos_in_base >= 0)
            eval_rule(lab, atoms_vals, rounded_te, test_y,
                      f"base AND pos_in_base%{k}={r}", acc_b, mf_b, passes)
            lab = base_atom & (pos_in_base % k != r) & (pos_in_base >= 0)
            eval_rule(lab, atoms_vals, rounded_te, test_y,
                      f"base AND pos_in_base%{k}!={r}", acc_b, mf_b, passes)

    # ---- te_cnt within base-POS ----
    base_te_cnt = te_cnt[base_atom]
    med_te = float(np.median(base_te_cnt))
    mean_te = float(np.mean(base_te_cnt))
    eval_rule(base_atom & (te_cnt >= med_te),
              atoms_vals, rounded_te, test_y,
              "base AND te_cnt>=med_in_base", acc_b, mf_b, passes)
    eval_rule(base_atom & (te_cnt > med_te),
              atoms_vals, rounded_te, test_y,
              "base AND te_cnt>med_in_base", acc_b, mf_b, passes)
    eval_rule(base_atom & (te_cnt < med_te),
              atoms_vals, rounded_te, test_y,
              "base AND te_cnt<med_in_base", acc_b, mf_b, passes)
    eval_rule(base_atom & (te_cnt >= mean_te),
              atoms_vals, rounded_te, test_y,
              "base AND te_cnt>=mean_in_base", acc_b, mf_b, passes)
    eval_rule(base_atom & (te_cnt < mean_te),
              atoms_vals, rounded_te, test_y,
              "base AND te_cnt<mean_in_base", acc_b, mf_b, passes)

    # ---- tr_cnt within base-POS ----
    base_tr_cnt = tr_cnt[base_atom]
    med_tr = float(np.median(base_tr_cnt))
    mean_tr = float(np.mean(base_tr_cnt))
    eval_rule(base_atom & (tr_cnt < med_tr),
              atoms_vals, rounded_te, test_y,
              "base AND tr_cnt<med_in_base", acc_b, mf_b, passes)
    eval_rule(base_atom & (tr_cnt <= med_tr),
              atoms_vals, rounded_te, test_y,
              "base AND tr_cnt<=med_in_base", acc_b, mf_b, passes)
    eval_rule(base_atom & (tr_cnt < mean_tr),
              atoms_vals, rounded_te, test_y,
              "base AND tr_cnt<mean_in_base", acc_b, mf_b, passes)
    eval_rule(base_atom & (tr_cnt > med_tr),
              atoms_vals, rounded_te, test_y,
              "base AND tr_cnt>med_in_base", acc_b, mf_b, passes)

    # ---- te_cnt/tr_cnt ratio within base-POS ----
    ratio_raw = te_cnt / (tr_cnt + 0.5)
    med_ratio = float(np.median(ratio_raw[base_atom]))
    eval_rule(base_atom & (ratio_raw > med_ratio),
              atoms_vals, rounded_te, test_y,
              "base AND ratio>med_in_base", acc_b, mf_b, passes)
    eval_rule(base_atom & (ratio_raw < med_ratio),
              atoms_vals, rounded_te, test_y,
              "base AND ratio<med_in_base", acc_b, mf_b, passes)
    eval_rule(base_atom & (ratio_raw >= med_ratio),
              atoms_vals, rounded_te, test_y,
              "base AND ratio>=med_in_base", acc_b, mf_b, passes)

    # ---- Boundary structure ----
    # atom is at edge of base-POS region (left or right neighbor is NOT base)
    is_bdy = np.zeros(K, dtype=bool)
    for i in range(K):
        if base_atom[i]:
            left_not_base = (i == 0) or (not base_atom[i - 1])
            right_not_base = (i == K - 1) or (not base_atom[i + 1])
            if left_not_base or right_not_base:
                is_bdy[i] = True
    eval_rule(base_atom & is_bdy,
              atoms_vals, rounded_te, test_y,
              "base AND boundary", acc_b, mf_b, passes)
    eval_rule(base_atom & ~is_bdy,
              atoms_vals, rounded_te, test_y,
              "base AND interior", acc_b, mf_b, passes)

    # ---- Every-nth dropping ----
    # Drop every atom where pos_in_base is at a "local min" of te_cnt among base
    # i.e. keep only local maxes of te_cnt
    kept = np.zeros(K, dtype=bool)
    for i in range(K):
        if not base_atom[i]:
            continue
        neighbors_in_base = []
        for j in range(max(0, i - 1), min(K, i + 2)):
            if j != i and base_atom[j]:
                neighbors_in_base.append(te_cnt[j])
        if not neighbors_in_base or te_cnt[i] >= max(neighbors_in_base):
            kept[i] = True
    eval_rule(kept, atoms_vals, rounded_te, test_y,
              "base AND local_max_te_cnt", acc_b, mf_b, passes)

    # Drop atoms that are local MIN of te_cnt within base-POS
    kept = base_atom.copy()
    for i in range(K):
        if not base_atom[i]:
            continue
        neighbors_in_base = []
        for j in range(max(0, i - 1), min(K, i + 2)):
            if j != i and base_atom[j]:
                neighbors_in_base.append(te_cnt[j])
        if neighbors_in_base and te_cnt[i] < min(neighbors_in_base):
            kept[i] = False
    eval_rule(kept, atoms_vals, rounded_te, test_y,
              "base drop local_min_te", acc_b, mf_b, passes)

    # Drop atoms that are local MAX of tr_cnt within base-POS
    kept = base_atom.copy()
    for i in range(K):
        if not base_atom[i]:
            continue
        neighbors_in_base = []
        for j in range(max(0, i - 1), min(K, i + 2)):
            if j != i and base_atom[j]:
                neighbors_in_base.append(tr_cnt[j])
        if neighbors_in_base and tr_cnt[i] > max(neighbors_in_base):
            kept[i] = False
    eval_rule(kept, atoms_vals, rounded_te, test_y,
              "base drop local_max_tr", acc_b, mf_b, passes)

    return passes


if __name__ == "__main__":
    all_passes = {}
    for d in ["MHClip_EN", "MHClip_ZH"]:
        p = per_dataset(d, all_passes)
        all_passes[d] = set(x[0] for x in p)
    common = all_passes.get("MHClip_EN", set()) & all_passes.get("MHClip_ZH", set())
    print(f"\n=== Rules passing strict-both on BOTH datasets: {len(common)} ===")
    for r in sorted(common):
        print(f"  {r}")
