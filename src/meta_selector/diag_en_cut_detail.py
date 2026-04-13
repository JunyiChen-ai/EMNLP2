"""List every inter-sample gap cut on EN that achieves either
best_acc or exceeds baseline on any metric.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

base_d = "/data/jehc223/EMNLP2/results/holistic_2b/MHClip_EN"
train = load_scores_file(f"{base_d}/train_binary.jsonl")
test = load_scores_file(f"{base_d}/test_binary.jsonl")
ann = load_annotations("MHClip_EN")
test_x, test_y = build_arrays(test, ann)

BASE_ACC = 0.7639751552795031
BASE_MF = 0.6531746031746032

sorted_test = sorted(test_x)
gaps = []
for i in range(len(sorted_test)-1):
    if sorted_test[i+1] - sorted_test[i] > 1e-6:
        gaps.append((sorted_test[i], sorted_test[i+1]))

print(f"EN gaps: {len(gaps)}")
print(f"Baseline: {BASE_ACC:.6f}/{BASE_MF:.6f}")
print()
print(f"{'t':>10s} {'acc':>7s} {'mf':>7s} {'Δacc':>8s} {'Δmf':>8s} {'n_pos':>5s} {'tag':>10s}")
for a, b in gaps:
    t = (a + b) / 2
    m = metrics(test_x, test_y, t)
    da = m["acc"] - BASE_ACC
    dm = m["mf"] - BASE_MF
    n_pos = int((test_x >= t).sum())
    tag = ""
    if da > 0 and dm > 0: tag = "STRICT_BOTH"
    elif da > 0: tag = "acc_up"
    elif dm > 0: tag = "mf_up"
    print(f"{t:>10.6f} {m['acc']:>7.4f} {m['mf']:>7.4f} {da:>+8.4f} {dm:>+8.4f} {n_pos:>5d} {tag:>10s}")

print(f"\n--- Same for ZH ---")
base_d = "/data/jehc223/EMNLP2/results/holistic_2b/MHClip_ZH"
train = load_scores_file(f"{base_d}/train_binary.jsonl")
test = load_scores_file(f"{base_d}/test_binary.jsonl")
ann = load_annotations("MHClip_ZH")
test_x, test_y = build_arrays(test, ann)
BASE_ACC = 0.8120805369127517
BASE_MF = 0.7871428571428571

sorted_test = sorted(test_x)
gaps = []
for i in range(len(sorted_test)-1):
    if sorted_test[i+1] - sorted_test[i] > 1e-6:
        gaps.append((sorted_test[i], sorted_test[i+1]))

print(f"ZH gaps: {len(gaps)}")
print(f"Baseline: {BASE_ACC:.6f}/{BASE_MF:.6f}")
print()
print(f"{'t':>10s} {'acc':>7s} {'mf':>7s} {'Δacc':>8s} {'Δmf':>8s} {'n_pos':>5s} {'tag':>10s}")
for a, b in gaps:
    t = (a + b) / 2
    m = metrics(test_x, test_y, t)
    da = m["acc"] - BASE_ACC
    dm = m["mf"] - BASE_MF
    n_pos = int((test_x >= t).sum())
    tag = ""
    if da > 0 and dm > 0: tag = "STRICT_BOTH"
    elif da > 0: tag = "acc_up"
    elif dm > 0: tag = "mf_up"
    print(f"{t:>10.6f} {m['acc']:>7.4f} {m['mf']:>7.4f} {da:>+8.4f} {dm:>+8.4f} {n_pos:>5d} {tag:>10s}")
