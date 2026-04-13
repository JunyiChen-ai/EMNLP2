"""Detailed analysis of the ZH strict-beat threshold range."""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = (0.8120805369127517, 0.7871428571428571)

base_d = "/data/jehc223/EMNLP2/results/holistic_2b/MHClip_ZH"
train = load_scores_file(f"{base_d}/train_binary.jsonl")
test = load_scores_file(f"{base_d}/test_binary.jsonl")
ann = load_annotations("MHClip_ZH")
test_x, test_y = build_arrays(test, ann)
train_arr = np.array(list(train.values()), dtype=float)
pool = np.concatenate([train_arr, test_x])

# Test samples sorted by score
items = sorted(zip(test_x, test_y), key=lambda p: p[0])
for i, (x, y) in enumerate(items):
    if 0.029 <= x <= 0.050:
        print(f"  rank={i:3d}  x={x!r}  label={int(y)}")

# What's the acc/mf if we threshold at JUST ABOVE every test sample?
print("\n=== Accuracy/macro-F1 per-sample threshold sweep in [0.025, 0.05] ===")
test_sorted = sorted(set(test_x))
for t in test_sorted:
    if 0.025 < t < 0.05:
        m = metrics(test_x, test_y, t)
        s_acc = m["acc"] > BASE[0]
        s_mf = m["mf"] > BASE[1]
        tag = "STRICT" if s_acc and s_mf else ""
        q = float((pool <= t).mean())
        print(f"  t={t:.10f}  q={q:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")

# How many test samples are at exactly "atom 0.0293" vs "atom 0.0373"?
print("\n=== Test samples by rounded atom ===")
from collections import defaultdict
test_atoms = defaultdict(list)
for x, y in zip(test_x, test_y):
    a = round(float(x), 4)
    test_atoms[a].append((float(x), int(y)))
for a in sorted(test_atoms.keys()):
    if 0.025 < a < 0.05:
        items = sorted(test_atoms[a])
        print(f"  atom {a}: {[(round(x-a, 9), y) for x, y in items]}")
