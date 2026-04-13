"""Nearest-train-distance feature: atom's distance to nearest train atom.
ZH signal at base AND nearest_dist > q0.10 (job 8110). Sweep quantiles
to see if a unified rule across both datasets exists.
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
    atoms = np.array(sorted(set(rounded_te)))
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base).astype(bool)
    nearest_dist = np.array([float(np.min(np.abs(train - a))) for a in atoms])
    return dict(atoms=atoms, rounded_te=rounded_te, test_y=test_y,
                base_atom=base_atom, nearest_dist=nearest_dist)


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

print("Sweep quantile cuts for base AND (nearest_dist > q_cut)")
print(f"{'q':>6} | {'EN acc/mf':>15} | {'ZH acc/mf':>15} | both")
for q in np.linspace(0.0, 0.95, 20):
    en_c = float(np.quantile(en["nearest_dist"], q)) if q > 0 else -1
    zh_c = float(np.quantile(zh["nearest_dist"], q)) if q > 0 else -1
    en_lab = en["base_atom"] & (en["nearest_dist"] > en_c)
    zh_lab = zh["base_atom"] & (zh["nearest_dist"] > zh_c)
    en_m = eval_lab(en_lab, en)
    zh_m = eval_lab(zh_lab, zh)
    en_s = f"{en_m[0]:.4f}/{en_m[1]:.4f}" if en_m else "trivial"
    zh_s = f"{zh_m[0]:.4f}/{zh_m[1]:.4f}" if zh_m else "trivial"
    en_pass = en_m and en_m[0] > BASE["MHClip_EN"][0] and en_m[1] > BASE["MHClip_EN"][1]
    zh_pass = zh_m and zh_m[0] > BASE["MHClip_ZH"][0] and zh_m[1] > BASE["MHClip_ZH"][1]
    both = "YES" if (en_pass and zh_pass) else ("EN" if en_pass else ("ZH" if zh_pass else ""))
    print(f"{q:.2f} | {en_s:>15} | {zh_s:>15} | {both}")

print("\nSweep quantile cuts for (nearest_dist > q_cut) alone")
for q in np.linspace(0.0, 0.95, 20):
    en_c = float(np.quantile(en["nearest_dist"], q)) if q > 0 else -1
    zh_c = float(np.quantile(zh["nearest_dist"], q)) if q > 0 else -1
    en_lab = (en["nearest_dist"] > en_c)
    zh_lab = (zh["nearest_dist"] > zh_c)
    en_m = eval_lab(en_lab, en)
    zh_m = eval_lab(zh_lab, zh)
    en_s = f"{en_m[0]:.4f}/{en_m[1]:.4f}" if en_m else "trivial"
    zh_s = f"{zh_m[0]:.4f}/{zh_m[1]:.4f}" if zh_m else "trivial"
    en_pass = en_m and en_m[0] > BASE["MHClip_EN"][0] and en_m[1] > BASE["MHClip_EN"][1]
    zh_pass = zh_m and zh_m[0] > BASE["MHClip_ZH"][0] and zh_m[1] > BASE["MHClip_ZH"][1]
    both = "YES" if (en_pass and zh_pass) else ("EN" if en_pass else ("ZH" if zh_pass else ""))
    print(f"{q:.2f} | {en_s:>15} | {zh_s:>15} | {both}")

print("\nSweep (base OR nearest_dist > q_cut)")
for q in np.linspace(0.0, 0.95, 20):
    en_c = float(np.quantile(en["nearest_dist"], q)) if q > 0 else -1
    zh_c = float(np.quantile(zh["nearest_dist"], q)) if q > 0 else -1
    en_lab = en["base_atom"] | (en["nearest_dist"] > en_c)
    zh_lab = zh["base_atom"] | (zh["nearest_dist"] > zh_c)
    en_m = eval_lab(en_lab, en)
    zh_m = eval_lab(zh_lab, zh)
    en_s = f"{en_m[0]:.4f}/{en_m[1]:.4f}" if en_m else "trivial"
    zh_s = f"{zh_m[0]:.4f}/{zh_m[1]:.4f}" if zh_m else "trivial"
    en_pass = en_m and en_m[0] > BASE["MHClip_EN"][0] and en_m[1] > BASE["MHClip_EN"][1]
    zh_pass = zh_m and zh_m[0] > BASE["MHClip_ZH"][0] and zh_m[1] > BASE["MHClip_ZH"][1]
    both = "YES" if (en_pass and zh_pass) else ("EN" if en_pass else ("ZH" if zh_pass else ""))
    print(f"{q:.2f} | {en_s:>15} | {zh_s:>15} | {both}")
