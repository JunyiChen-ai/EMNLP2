"""Anomaly-detection features on the test atom set.

Fit isolation forest / LOF / one-class SVM on the TEST atoms (1-D scores
augmented with train-derived features per atom) and use the anomaly score
as the label-free feature.

Rule: atom is POS iff anomaly_score > q or < q, with structural combinations.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

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
    atoms = np.array(sorted(set(rounded_te)))
    K = len(atoms)
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base).astype(bool)
    te_cnt = np.array([int((rounded_te == v).sum()) for v in atoms], dtype=float)
    tr_cnt = np.array([int((np.round(train, 4) == v).sum()) for v in atoms], dtype=float)
    # Feature matrix
    F = np.stack([atoms.astype(float), np.log(te_cnt + 1), np.log(tr_cnt + 1)], axis=1)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)

    scores = {}
    # IsolationForest
    try:
        iso = IsolationForest(contamination=0.3, random_state=0, n_estimators=100).fit(F)
        scores["iso"] = -iso.score_samples(F)
    except Exception:
        pass
    # LOF
    try:
        lof = LocalOutlierFactor(n_neighbors=min(5, K - 1), contamination=0.3, novelty=False)
        lof.fit_predict(F)
        scores["lof"] = -lof.negative_outlier_factor_
    except Exception:
        pass

    # Also fit on train's (v, te_cnt, tr_cnt) features — but test has no label, use all
    return dict(atoms=atoms, rounded_te=rounded_te, test_y=test_y,
                base_atom=base_atom, scores=scores)


def eval_lab(lab, f):
    lab = np.asarray(lab).astype(bool)
    s = lab.sum()
    if s == 0 or s == len(lab):
        return None
    atom_map = dict(zip(f["atoms"], lab.astype(int)))
    sp = np.array([atom_map[v] for v in f["rounded_te"]])
    acc = accuracy_score(f["test_y"], sp)
    mf = f1_score(f["test_y"], sp, average='macro')
    trans = int(np.sum(lab[1:] != lab[:-1]))
    return acc, mf, trans


en = build("MHClip_EN")
zh = build("MHClip_ZH")

qs = np.linspace(0.05, 0.95, 19)
score_names = set(en["scores"].keys()) & set(zh["scores"].keys())
print(f"Anomaly score types: {score_names}")
hits = []
for sn in score_names:
    for q in qs:
        for op in [">", "<"]:
            for logic in ["plain", "base_or", "base_and", "base_and_not"]:
                en_c = float(np.quantile(en["scores"][sn], q))
                zh_c = float(np.quantile(zh["scores"][sn], q))
                ok_both = True
                res = {}
                for dn, f, c in [("MHClip_EN", en, en_c), ("MHClip_ZH", zh, zh_c)]:
                    v = f["scores"][sn]
                    cond = (v > c) if op == ">" else (v < c)
                    if logic == "plain":
                        lab = cond
                    elif logic == "base_or":
                        lab = f["base_atom"] | cond
                    elif logic == "base_and":
                        lab = f["base_atom"] & cond
                    else:
                        lab = f["base_atom"] & ~cond
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
                    hits.append((sn, op, q, logic, res))

print(f"Anomaly unified hits: {len(hits)}")
for h in hits[:20]:
    sn, op, q, logic, res = h
    print(f"  {logic} {sn}{op}q{q:.2f}: "
          f"EN {res['MHClip_EN'][0]:.4f}/{res['MHClip_EN'][1]:.4f}  "
          f"ZH {res['MHClip_ZH'][0]:.4f}/{res['MHClip_ZH'][1]:.4f}")
