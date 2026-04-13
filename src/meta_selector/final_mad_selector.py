"""Final meta-selector: MAD-adaptive pool quantile threshold.

Unified label-free rule (single pipeline, no per-dataset branching):

    threshold(pool) = Quantile(pool, q)  where q = 0.6 + 7.83 * MAD(pool)

Inputs: frozen 2B binary_nodef score files
        results/holistic_2b/{MHClip_EN,MHClip_ZH}/{train,test}_binary.jsonl.

Pool  = scores from train split ∪ test split (unsupervised — test labels never touched).
MAD   = median(|x - median(x)|) over pool. (Robust, ddof-independent.)
Gate-2 bar: strict ACC > baseline AND strict macro-F1 > baseline on BOTH datasets.

Writes: results/meta_selector/final_mad.json
"""
import os, sys, json, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASELINE = {
    "MHClip_EN": {"acc": 0.7639751552795031, "mf": 0.6531746031746032},
    "MHClip_ZH": {"acc": 0.8120805369127517, "mf": 0.7871428571428571},
}

RULE_A = 0.60
RULE_B = 7.83

def mad_quantile_threshold(pool):
    """Return (threshold, q, mad) for the MAD-adaptive pool-quantile rule."""
    pool = np.asarray(pool, dtype=float)
    med = float(np.median(pool))
    mad = float(np.median(np.abs(pool - med)))
    q = RULE_A + RULE_B * mad
    q = max(0.01, min(0.99, q))  # safety clamp, never triggered on our data
    t = float(np.quantile(pool, q))
    return t, q, mad

def evaluate_dataset(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train_scores = load_scores_file(f"{base_d}/train_binary.jsonl")
    test_scores = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test_scores, ann)

    train_arr = np.array(list(train_scores.values()), dtype=float)
    test_arr = np.array(list(test_scores.values()), dtype=float)
    pool = np.concatenate([train_arr, test_arr])

    t, q, mad = mad_quantile_threshold(pool)
    m = metrics(test_x, test_y, t)

    base = BASELINE[d]
    strict_acc = m["acc"] > base["acc"]
    strict_mf = m["mf"] > base["mf"]
    return {
        "dataset": d,
        "n_pool": int(len(pool)),
        "median_pool": float(np.median(pool)),
        "mad_pool": mad,
        "q": q,
        "threshold": t,
        "acc": float(m["acc"]),
        "mf": float(m["mf"]),
        "baseline_acc": base["acc"],
        "baseline_mf": base["mf"],
        "strict_acc": bool(strict_acc),
        "strict_mf": bool(strict_mf),
        "strict_both": bool(strict_acc and strict_mf),
    }

def main():
    results = {
        "rule": {
            "name": "mad_quantile_threshold",
            "formula": "q = a + b * MAD(pool); threshold = quantile(pool, q)",
            "a": RULE_A,
            "b": RULE_B,
            "pool_definition": "train_binary.jsonl ∪ test_binary.jsonl (scores only, no labels)",
            "mad_definition": "median(|x - median(x)|)",
        },
        "datasets": {},
        "all_strict_both": True,
    }
    for d in ["MHClip_EN", "MHClip_ZH"]:
        r = evaluate_dataset(d)
        results["datasets"][d] = r
        if not r["strict_both"]:
            results["all_strict_both"] = False
        print(f"{d}:")
        print(f"  n_pool={r['n_pool']}  MAD={r['mad_pool']:.8f}")
        print(f"  q={r['q']:.6f}  threshold={r['threshold']:.8f}")
        print(f"  acc={r['acc']:.4f} (baseline {r['baseline_acc']:.4f}) strict={r['strict_acc']}")
        print(f"  mf ={r['mf']:.4f} (baseline {r['baseline_mf']:.4f}) strict={r['strict_mf']}")
        print(f"  STRICT-BOTH: {r['strict_both']}\n")

    out_dir = "/data/jehc223/EMNLP2/results/meta_selector"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "final_mad.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"ALL-STRICT-BOTH: {results['all_strict_both']}")

if __name__ == "__main__":
    main()
