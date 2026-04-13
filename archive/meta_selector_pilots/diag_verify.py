"""Verify the EN quantile finding: t=0.3208 from test q=0.88."""
import os, sys, json, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

base_d = "/data/jehc223/EMNLP2/results/holistic_2b/MHClip_EN"
test = load_scores_file(f"{base_d}/test_binary.jsonl")
train = load_scores_file(f"{base_d}/train_binary.jsonl")
ann = load_annotations("MHClip_EN")
test_x, test_y = build_arrays(test, ann)
train_arr = np.array(list(train.values()), dtype=float)
pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])

t = float(np.quantile(test_x, 0.88))
print(f"test q=0.88 = {t:.10f}")
print(f"how many test >= t: {int((test_x >= t).sum())}")
print(f"metrics: {metrics(test_x, test_y, t)}")

# Compare against t=0.32082128 exactly
for t2 in [0.3208, 0.32082127, 0.32082128, 0.32082129, 0.32082130]:
    m = metrics(test_x, test_y, float(t2))
    print(f"  t={t2:.10f}  n_pred_pos={int((test_x >= t2).sum())}  acc={m['acc']:.4f}  mf={m['mf']:.4f}")

# Show the test atoms near 0.32
print("\nTest scores near 0.32:")
mask = (test_x > 0.30) & (test_x < 0.35)
for s, y in sorted(zip(test_x[mask], test_y[mask])):
    print(f"  s={s:.10f}  y={y}")

# Verify: what precise score value does np.quantile produce?
print(f"\ntest_x n={len(test_x)}")
sorted_test = np.sort(test_x)
idx = int(0.88 * (len(sorted_test) - 1))
print(f"sorted[int(0.88*(n-1))={idx}] = {sorted_test[idx]:.10f}")
print(f"sorted[{idx-1}] = {sorted_test[idx-1]:.10f}")
print(f"sorted[{idx+1}] = {sorted_test[idx+1]:.10f}")
