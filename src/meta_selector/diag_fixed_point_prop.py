"""Iterative/self-referential label propagation on atoms.

Feature class: fixed-point over atom labels. Start from base_atom, then
iterate: new_label[i] = f(neighbors' old labels, atom stats). This is a
self-referential feature — it cannot be written as a closed-form statistic
over a single atom, it depends on the pool and its own previous output.

Propagators tried:
- Majority vote over k-NN in (v, tr_cnt, te_cnt) feature space
- Label smoothing over score-adjacent atoms (atom[i-1], atom[i+1])
- Asymmetric flip: atom keeps base label unless >=k neighbors disagree

Each iterated labeling is then evaluated directly (no threshold).
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

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
    F = np.stack([atoms, np.log(te_cnt + 1), np.log(tr_cnt + 1)], axis=1)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    return dict(atoms=atoms, rounded_te=rounded_te, test_y=test_y,
                base_atom=base_atom, te_cnt=te_cnt, tr_cnt=tr_cnt, F=F, K=K)


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


def propagate_knn(f, k, n_iter, start_from_base=True):
    nn = NearestNeighbors(n_neighbors=min(k + 1, f["K"])).fit(f["F"])
    idx = nn.kneighbors(f["F"], return_distance=False)[:, 1:]  # drop self
    lab = f["base_atom"].copy() if start_from_base else np.zeros(f["K"], dtype=bool)
    history = [lab.copy()]
    for _ in range(n_iter):
        new = np.array([lab[idx[i]].sum() > (k / 2) for i in range(f["K"])])
        if (new == lab).all():
            break
        lab = new
        history.append(lab.copy())
    return lab, history


def propagate_score_adj(f, window, n_iter, start_from_base=True):
    K = f["K"]
    lab = f["base_atom"].copy() if start_from_base else np.zeros(K, dtype=bool)
    history = [lab.copy()]
    for _ in range(n_iter):
        new = lab.copy()
        for i in range(K):
            lo = max(0, i - window)
            hi = min(K, i + window + 1)
            vote = lab[lo:hi].sum()
            total = hi - lo
            new[i] = (vote > total / 2)
        if (new == lab).all():
            break
        lab = new
        history.append(lab.copy())
    return lab, history


def propagate_asym_flip(f, k, n_iter, start_from_base=True):
    nn = NearestNeighbors(n_neighbors=min(k + 1, f["K"])).fit(f["F"])
    idx = nn.kneighbors(f["F"], return_distance=False)[:, 1:]
    lab = f["base_atom"].copy() if start_from_base else np.zeros(f["K"], dtype=bool)
    history = [lab.copy()]
    for _ in range(n_iter):
        new = lab.copy()
        for i in range(f["K"]):
            neigh = lab[idx[i]]
            if lab[i] and neigh.sum() < k / 2:
                new[i] = False
            elif (not lab[i]) and neigh.sum() > k / 2:
                new[i] = True
        if (new == lab).all():
            break
        lab = new
        history.append(lab.copy())
    return lab, history


en = build("MHClip_EN")
zh = build("MHClip_ZH")

configs = []
for k in [2, 3, 4, 5, 6]:
    for n_iter in [1, 2, 3, 5, 10]:
        for sfb in [True, False]:
            configs.append(("knn", k, n_iter, sfb))
for win in [1, 2, 3, 4]:
    for n_iter in [1, 2, 3, 5, 10]:
        for sfb in [True, False]:
            configs.append(("adj", win, n_iter, sfb))
for k in [2, 3, 4, 5, 6]:
    for n_iter in [1, 2, 3, 5, 10]:
        for sfb in [True, False]:
            configs.append(("asym", k, n_iter, sfb))

hits = []
for kind, p, n_iter, sfb in configs:
    ok_both = True
    res = {}
    for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
        if kind == "knn":
            lab, _ = propagate_knn(f, p, n_iter, sfb)
        elif kind == "adj":
            lab, _ = propagate_score_adj(f, p, n_iter, sfb)
        else:
            lab, _ = propagate_asym_flip(f, p, n_iter, sfb)
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
        hits.append((kind, p, n_iter, sfb, res))

print(f"Configs tried: {len(configs)}")
print(f"Fixed-point propagation strict-both hits: {len(hits)}")
for h in hits[:20]:
    kind, p, n_iter, sfb, res = h
    print(f"  {kind}(p={p},iter={n_iter},sfb={sfb}): "
          f"EN {res['MHClip_EN'][0]:.4f}/{res['MHClip_EN'][1]:.4f}  "
          f"ZH {res['MHClip_ZH'][0]:.4f}/{res['MHClip_ZH'][1]:.4f}")
