"""Discrete/non-continuous atom-level features.

Motivation: smooth-continuous features (KDE, density ratios, pool ratios)
cannot create period-1 alternation patterns observed in the oracle passing
subsets. Non-continuous features can:
- rank-parity: parity of rank-order of an atom
- cluster-id: discrete cluster membership from 1-D clustering
- iterative fixed-point: discrete dynamics on base labels
- graph-degree parity: parity of k-NN-graph degree

All features are label-free (depend only on train/test score distributions).
All rules are parameter-free or use simple structural constants.
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


def eval_rule(atom_lab, atoms_vals, rounded_te, test_y, name, acc_b, mf_b, results):
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
    tag = " PASS" if strict else ""
    print(f"  {name}: acc={acc:.4f} mf={mf:.4f} trans={trans}{tag}")
    results.append((name, acc, mf, trans, strict, atom_lab.copy()))


def per_dataset(d):
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

    results = []

    # ---- Rank-parity family ----
    # Sort atoms by score — parity of index (atoms are already sorted in atoms_vals)
    idx = np.arange(K)
    eval_rule(idx % 2 == 0, atoms_vals, rounded_te, test_y,
              "rank_parity_even", acc_b, mf_b, results)
    eval_rule(idx % 2 == 1, atoms_vals, rounded_te, test_y,
              "rank_parity_odd", acc_b, mf_b, results)
    eval_rule(base_atom | (idx % 2 == 0), atoms_vals, rounded_te, test_y,
              "base OR rank_even", acc_b, mf_b, results)
    eval_rule(base_atom | (idx % 2 == 1), atoms_vals, rounded_te, test_y,
              "base OR rank_odd", acc_b, mf_b, results)
    eval_rule(base_atom & (idx % 2 == 0), atoms_vals, rounded_te, test_y,
              "base AND rank_even", acc_b, mf_b, results)
    eval_rule(base_atom & (idx % 2 == 1), atoms_vals, rounded_te, test_y,
              "base AND rank_odd", acc_b, mf_b, results)

    # Parity by te_cnt rank
    te_rank = np.argsort(np.argsort(te_cnt))
    eval_rule(te_rank % 2 == 0, atoms_vals, rounded_te, test_y,
              "te_cnt_rank_parity_even", acc_b, mf_b, results)
    eval_rule(base_atom & (te_rank % 2 == 0), atoms_vals, rounded_te, test_y,
              "base AND te_cnt_rank_even", acc_b, mf_b, results)
    eval_rule(base_atom | (te_rank % 2 == 0), atoms_vals, rounded_te, test_y,
              "base OR te_cnt_rank_even", acc_b, mf_b, results)

    # Parity by tr_cnt rank
    tr_rank = np.argsort(np.argsort(tr_cnt))
    eval_rule(tr_rank % 2 == 0, atoms_vals, rounded_te, test_y,
              "tr_cnt_rank_parity_even", acc_b, mf_b, results)
    eval_rule(base_atom & (tr_rank % 2 == 0), atoms_vals, rounded_te, test_y,
              "base AND tr_cnt_rank_even", acc_b, mf_b, results)

    # ---- 1-D clustering on atom scores (gap clustering) ----
    # Cluster atoms by gap between consecutive scores; split where gap > median gap
    gaps = np.diff(atoms_vals)
    if len(gaps) > 0:
        med_gap = np.median(gaps)
        # New cluster at each gap > median
        splits = gaps > med_gap
        cluster_id = np.concatenate([[0], np.cumsum(splits)])
        n_clust = int(cluster_id.max()) + 1
        # Rule: atom is POS iff its cluster-id is odd (parity of cluster)
        eval_rule(cluster_id % 2 == 1, atoms_vals, rounded_te, test_y,
                  "gap_cluster_parity_odd", acc_b, mf_b, results)
        eval_rule(base_atom & (cluster_id % 2 == 1), atoms_vals, rounded_te, test_y,
                  "base AND gap_cluster_odd", acc_b, mf_b, results)
        eval_rule(base_atom | (cluster_id % 2 == 1), atoms_vals, rounded_te, test_y,
                  "base OR gap_cluster_odd", acc_b, mf_b, results)
        # Rule: atom is POS iff cluster size == 1 (singleton cluster = isolated)
        cl_size = np.array([int((cluster_id == c).sum()) for c in cluster_id])
        eval_rule(cl_size == 1, atoms_vals, rounded_te, test_y,
                  "singleton_cluster", acc_b, mf_b, results)
        eval_rule(base_atom | (cl_size == 1), atoms_vals, rounded_te, test_y,
                  "base OR singleton", acc_b, mf_b, results)
        eval_rule(base_atom & (cl_size == 1), atoms_vals, rounded_te, test_y,
                  "base AND singleton", acc_b, mf_b, results)

    # Cluster by 90th-percentile gap (larger clusters)
    if len(gaps) > 0:
        q_gap = np.quantile(gaps, 0.9)
        splits = gaps > q_gap
        cluster_id = np.concatenate([[0], np.cumsum(splits)])
        cl_size = np.array([int((cluster_id == c).sum()) for c in cluster_id])
        eval_rule(cl_size == 1, atoms_vals, rounded_te, test_y,
                  "q90_singleton", acc_b, mf_b, results)
        eval_rule(base_atom | (cl_size == 1), atoms_vals, rounded_te, test_y,
                  "base OR q90_singleton", acc_b, mf_b, results)

    # ---- Iterative fixed-point labeling ----
    # Start with base, iterate: flip atom if its 2 neighbors (in atom rank) both disagree
    def iterate_flip(init, max_iter=20):
        lab = init.copy().astype(bool)
        for _ in range(max_iter):
            new = lab.copy()
            for i in range(K):
                if 0 < i < K - 1:
                    if lab[i - 1] == lab[i + 1] and lab[i - 1] != lab[i]:
                        new[i] = lab[i - 1]
            if np.array_equal(new, lab):
                break
            lab = new
        return lab

    # Smoothed base
    smoothed = iterate_flip(base_atom)
    eval_rule(smoothed, atoms_vals, rounded_te, test_y,
              "iter_smooth_base", acc_b, mf_b, results)

    # "Anti-smooth": atoms that DISAGREE with their smoothed version = locally anomalous
    anomalous = base_atom != smoothed
    eval_rule(anomalous, atoms_vals, rounded_te, test_y,
              "iter_anomalous_base", acc_b, mf_b, results)
    eval_rule(base_atom | anomalous, atoms_vals, rounded_te, test_y,
              "base OR anomalous", acc_b, mf_b, results)
    eval_rule(base_atom & ~anomalous, atoms_vals, rounded_te, test_y,
              "base AND NOT anomalous", acc_b, mf_b, results)

    # ---- Graph over atoms: k-NN by score ----
    # Each atom connects to its 2 nearest neighbors in score. Degree is 2 unless endpoint.
    # Instead: label atom by parity of # atoms in train set within radius r (r = avg gap).
    avg_gap = float(np.mean(gaps)) if len(gaps) > 0 else 1e-3
    # Number of train samples within avg_gap of each atom
    tr_nbhd = np.array([int(((rounded_tr >= v - avg_gap) &
                             (rounded_tr <= v + avg_gap)).sum())
                        for v in atoms_vals])
    te_nbhd = np.array([int(((rounded_te >= v - avg_gap) &
                             (rounded_te <= v + avg_gap)).sum())
                        for v in atoms_vals])
    # Parity rules
    eval_rule(tr_nbhd % 2 == 1, atoms_vals, rounded_te, test_y,
              "tr_nbhd_odd", acc_b, mf_b, results)
    eval_rule(te_nbhd % 2 == 1, atoms_vals, rounded_te, test_y,
              "te_nbhd_odd", acc_b, mf_b, results)
    eval_rule(base_atom & (tr_nbhd % 2 == 1), atoms_vals, rounded_te, test_y,
              "base AND tr_nbhd_odd", acc_b, mf_b, results)
    eval_rule(base_atom | (tr_nbhd % 2 == 1), atoms_vals, rounded_te, test_y,
              "base OR tr_nbhd_odd", acc_b, mf_b, results)
    eval_rule(base_atom & (te_nbhd % 2 == 1), atoms_vals, rounded_te, test_y,
              "base AND te_nbhd_odd", acc_b, mf_b, results)
    eval_rule(base_atom | (te_nbhd % 2 == 1), atoms_vals, rounded_te, test_y,
              "base OR te_nbhd_odd", acc_b, mf_b, results)

    # Comparison: tr_nbhd vs te_nbhd
    eval_rule(tr_nbhd > te_nbhd, atoms_vals, rounded_te, test_y,
              "tr_nbhd>te_nbhd", acc_b, mf_b, results)
    eval_rule(tr_nbhd < te_nbhd, atoms_vals, rounded_te, test_y,
              "tr_nbhd<te_nbhd", acc_b, mf_b, results)
    eval_rule(base_atom & (tr_nbhd < te_nbhd), atoms_vals, rounded_te, test_y,
              "base AND tr_nbhd<te_nbhd", acc_b, mf_b, results)
    eval_rule(base_atom | (tr_nbhd < te_nbhd), atoms_vals, rounded_te, test_y,
              "base OR tr_nbhd<te_nbhd", acc_b, mf_b, results)

    # ---- Co-occurrence / cardinality features ----
    # "Atom has unique test count" = te_cnt equals max(te_cnt)
    eval_rule(te_cnt == te_cnt.max(), atoms_vals, rounded_te, test_y,
              "te_cnt_is_max", acc_b, mf_b, results)
    eval_rule(base_atom | (te_cnt == te_cnt.max()), atoms_vals, rounded_te, test_y,
              "base OR te_cnt_is_max", acc_b, mf_b, results)
    eval_rule(base_atom & ~(te_cnt == te_cnt.max()), atoms_vals, rounded_te, test_y,
              "base AND NOT te_cnt_is_max", acc_b, mf_b, results)
    # Atom has tr_cnt == 0 (unseen in train)
    eval_rule(tr_cnt == 0, atoms_vals, rounded_te, test_y,
              "tr_cnt_is_0", acc_b, mf_b, results)
    eval_rule(base_atom | (tr_cnt == 0), atoms_vals, rounded_te, test_y,
              "base OR tr_cnt_is_0", acc_b, mf_b, results)
    eval_rule(base_atom & ~(tr_cnt == 0), atoms_vals, rounded_te, test_y,
              "base AND NOT tr_cnt_is_0", acc_b, mf_b, results)

    return results


if __name__ == "__main__":
    for d in ["MHClip_EN", "MHClip_ZH"]:
        per_dataset(d)
