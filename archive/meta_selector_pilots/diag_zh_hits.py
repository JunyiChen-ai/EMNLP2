"""Drill into ZH strict-beat hits to see distinct raw threshold values and whether
they straddle atom boundaries or sit within clusters."""
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

unique_vals = sorted(set(list(pool) + list(test_x)))
print(f"Total unique values in pool+test: {len(unique_vals)}")

# Count values in the "atom 0.0293" and "atom 0.0373" ranges
for atom_v, hi in [(0.029, 0.030), (0.037, 0.038)]:
    vals = [v for v in unique_vals if atom_v <= v < hi]
    print(f"\nValues in [{atom_v}, {hi}): {len(vals)}")
    for v in vals:
        in_pool = (pool == v).sum()
        in_test = (test_x == v).sum()
        label = None
        for x, y in zip(test_x, test_y):
            if x == v:
                label = int(y)
                break
        print(f"  {v!r}  pool={in_pool}  test={in_test}  test_label={label}")

# Now find distinct threshold values that strict-beat
print("\n=== All strict-beat thresholds (distinct values) ===")
cand = list(unique_vals)
for i in range(len(unique_vals)-1):
    cand.append((unique_vals[i] + unique_vals[i+1]) / 2)
cand = sorted(set(cand))

hits = []
for t in cand:
    m = metrics(test_x, test_y, t)
    if m["acc"] > BASE[0] and m["mf"] > BASE[1]:
        hits.append((t, m["acc"], m["mf"]))

print(f"Total strict-beat thresholds: {len(hits)}")
# Group consecutive
if hits:
    prev_t = hits[0][0]
    print(f"  Range: [{hits[0][0]!r}, {hits[-1][0]!r}]")
    print(f"  Range rounded: [{hits[0][0]:.10f}, {hits[-1][0]:.10f}]")
    # Check if these are all distinct values or repeated
    ts = set(t for t, _, _ in hits)
    print(f"  Distinct threshold values in hits: {len(ts)}")
    print(f"  First 5 distinct:")
    for t in sorted(ts)[:5]:
        print(f"    {t!r}")
    print(f"  Last 5 distinct:")
    for t in sorted(ts)[-5:]:
        print(f"    {t!r}")
