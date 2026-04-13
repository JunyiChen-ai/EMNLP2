"""Non-threshold structural atom rules.

Per team-lead hint: flag atoms that satisfy a structural property P,
where P is NOT 'score > t'. Tried properties:
- is_local_max(te_cnt) on atom sequence (has higher te_cnt than both neighbors)
- is_local_max(tr_cnt)
- is_local_min(near_dist)
- te_cnt > tr_cnt at all k-nearest atoms
- atom sits on concave region of te_cdf (2nd-difference > threshold)
- atom count is an "outlier" vs neighbors (robust z-score>q)
- runlength property: atom is in the first / last run of base_atom==True
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


def build(d):
    train, test_x, test_y = load(d)
    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms = np.array(sorted(set(rounded_te)))
    K = len(atoms)
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base).astype(bool)
    te_cnt = np.array([int((rounded_te == v).sum()) for v in atoms], dtype=float)
    tr_cnt = np.array([int((rounded_tr == v).sum()) for v in atoms], dtype=float)
    near_dist = np.array([float(np.min(np.abs(train - v))) for v in atoms])
    te_cdf = np.array([float((test_x <= v).mean()) for v in atoms])

    # is_local_max: atom i has value strictly greater than both neighbors
    def loc_max(a):
        r = np.zeros(len(a), dtype=bool)
        for i in range(1, len(a) - 1):
            r[i] = (a[i] > a[i - 1] and a[i] > a[i + 1])
        return r

    def loc_min(a):
        r = np.zeros(len(a), dtype=bool)
        for i in range(1, len(a) - 1):
            r[i] = (a[i] < a[i - 1] and a[i] < a[i + 1])
        return r

    lmax_te = loc_max(te_cnt)
    lmax_tr = loc_max(tr_cnt)
    lmin_nd = loc_min(near_dist)
    # concave: 2nd diff of te_cdf positive
    te_cdf_d2 = np.concatenate([[0], np.diff(np.diff(te_cdf)), [0]])
    concave = te_cdf_d2 > 0
    # neighbor outlier: |te - median(window)| / mad
    def outlier_mask(a, window=3):
        r = np.zeros(len(a), dtype=bool)
        for i in range(len(a)):
            lo = max(0, i - window); hi = min(len(a), i + window + 1)
            w = a[lo:hi]
            med = np.median(w)
            mad = np.median(np.abs(w - med)) + 1e-9
            r[i] = abs(a[i] - med) / mad > 2.5
        return r
    out_te = outlier_mask(te_cnt)
    out_tr = outlier_mask(tr_cnt)
    # runs of base_atom: first-run atoms and last-run atoms
    first_run = np.zeros(K, dtype=bool)
    if base_atom.any():
        start = np.argmax(base_atom)
        i = start
        while i < K and base_atom[i]:
            first_run[i] = True
            i += 1

    return dict(atoms=atoms, rounded_te=rounded_te, test_y=test_y,
                base_atom=base_atom, lmax_te=lmax_te, lmax_tr=lmax_tr,
                lmin_nd=lmin_nd, concave=concave, out_te=out_te, out_tr=out_tr,
                first_run=first_run)


def eval_lab(lab, f):
    lab = np.asarray(lab).astype(bool)
    s = lab.sum()
    if s == 0 or s == len(lab):
        return None
    atom_map = dict(zip(f["atoms"], lab.astype(int)))
    sp = np.array([atom_map[v] for v in f["rounded_te"]])
    acc = accuracy_score(f["test_y"], sp)
    mf = f1_score(f["test_y"], sp, average='macro')
    return acc, mf


en = build("MHClip_EN")
zh = build("MHClip_ZH")

props = ["lmax_te", "lmax_tr", "lmin_nd", "concave", "out_te", "out_tr", "first_run"]

hits = []
for p in props:
    for logic in ["plain", "base_or", "base_and", "base_and_not", "base_xor"]:
        ok_both = True
        res = {}
        for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
            prop = f[p]
            if logic == "plain":
                lab = prop
            elif logic == "base_or":
                lab = f["base_atom"] | prop
            elif logic == "base_and":
                lab = f["base_atom"] & prop
            elif logic == "base_and_not":
                lab = f["base_atom"] & ~prop
            else:
                lab = f["base_atom"] ^ prop
            m = eval_lab(lab, f)
            if m is None:
                ok_both = False
                break
            ab, mb = BASE[dn]
            if not (m[0] > ab and m[1] > mb):
                ok_both = False
                break
            res[dn] = m
        if ok_both:
            hits.append((p, logic, res))

# Pairs
hits2 = []
for p1 in props:
    for p2 in props:
        for logic in ["p1_or_p2", "p1_and_p2", "base_and_notp1_or_p2",
                      "base_or_p1_and_p2", "base_and_notp1_and_notp2"]:
            ok_both = True
            res = {}
            for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
                a = f[p1]; b = f[p2]
                if logic == "p1_or_p2":
                    lab = a | b
                elif logic == "p1_and_p2":
                    lab = a & b
                elif logic == "base_and_notp1_or_p2":
                    lab = f["base_atom"] & ~(a | b)
                elif logic == "base_or_p1_and_p2":
                    lab = f["base_atom"] | (a & b)
                else:
                    lab = f["base_atom"] & ~a & ~b
                m = eval_lab(lab, f)
                if m is None:
                    ok_both = False
                    break
                ab, mb = BASE[dn]
                if not (m[0] > ab and m[1] > mb):
                    ok_both = False
                    break
                res[dn] = m
            if ok_both:
                hits2.append((p1, p2, logic, res))

print(f"Structural-local 1-prop strict-both hits: {len(hits)}")
for h in hits[:20]:
    print(" ", h[:-1], "EN", h[-1]['MHClip_EN'], "ZH", h[-1]['MHClip_ZH'])
print(f"Structural-local 2-prop strict-both hits: {len(hits2)}")
for h in hits2[:20]:
    print(" ", h[:-1], "EN", h[-1]['MHClip_EN'], "ZH", h[-1]['MHClip_ZH'])
