"""Label-free rule candidates on density-ratio features.

The rule must select the CUT via a published, data-independent method
applied to label-free data. No labels may be consulted.

Candidates:
1. cut = 0 (natural zero of log_ratio)
2. cut = median(log_ratio over test)
3. cut = mean(log_ratio over test)
4. cut = Otsu on log_ratio over test
5. cut = GMM k=2 on log_ratio over test, midpoint of 2 means
6. cut = Triangle method on log_ratio histogram

For each bandwidth k in {0.5, 1, 2, 4}, try each cut rule.
Report acc, mf, and strict-beat pass.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu, threshold_triangle

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def kde_at(x, pool, h):
    return np.sum(np.exp(-0.5 * ((pool - x) / h) ** 2)) / (len(pool) * h * np.sqrt(2*np.pi))


def silverman_h(pool):
    return float(pool.std()) * (4 / (3 * len(pool))) ** (1 / 5)


def log_ratio_per_sample(train, test_x, h):
    f_tr = np.array([kde_at(x, train, h) for x in test_x])
    f_te = np.array([kde_at(x, test_x, h) for x in test_x])
    return np.log((f_tr + 1e-12) / (f_te + 1e-12))


def eval_pred(y, p):
    return accuracy_score(y, p), f1_score(y, p, average='macro')


def cut_rules(feat):
    rules = {}
    rules["zero"] = 0.0
    rules["median"] = float(np.median(feat))
    rules["mean"] = float(np.mean(feat))
    try:
        rules["otsu"] = float(threshold_otsu(feat, nbins=256))
    except Exception:
        rules["otsu"] = None
    try:
        rules["triangle"] = float(threshold_triangle(feat, nbins=256))
    except Exception:
        rules["triangle"] = None
    try:
        g = GaussianMixture(n_components=2, random_state=0).fit(feat.reshape(-1, 1))
        mus = sorted(g.means_.flatten())
        rules["gmm_mid"] = float((mus[0] + mus[1]) / 2)
    except Exception:
        rules["gmm_mid"] = None
    return rules


print(f"{'dataset':8s} {'k':>5s} {'rule':>10s} {'cut':>10s} {'acc':>8s} {'mf':>8s}  strict?")
print("-" * 70)
for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    hs = silverman_h(train)
    for k in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        h = hs * k
        feat = log_ratio_per_sample(train, test_x, h)
        rules = cut_rules(feat)
        for rname, cut in rules.items():
            if cut is None: continue
            for direction in ["lt", "gt"]:
                pred = (feat < cut if direction == "lt" else feat > cut).astype(int)
                if pred.sum() == 0 or pred.sum() == len(pred): continue
                acc, mf = eval_pred(test_y, pred)
                strict = (acc > acc_b) and (mf >= mf_b)
                tag = " ** PASS **" if strict else ""
                if strict or rname == "zero":
                    print(f"{d[-2:]:>8s} {k:>5.2f} {rname+'_'+direction:>10s} {cut:>10.4f} {acc:>8.4f} {mf:>8.4f} {tag}")
