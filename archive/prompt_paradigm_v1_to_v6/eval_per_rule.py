"""
prompt_paradigm v5 — Per-Rule Disjunction Readout evaluator.

Reads results/prompt_paradigm/{dataset}/{train,test}_per_rule.jsonl which
contain per-rule probability vectors (p_rules: list of K floats) and the
`format_clean` flag, and reports all 8 binding Gate 2 clauses from
docs/proposals/prompt_paradigm_v5.md §5.

Binding clauses:
  1. Oracle-first: EN oracle > 0.7764 AND ZH oracle > 0.8121 (strict)
  2. Macro-F1 non-regression on the unified cell
  3. Ablation A load-bearing (baseline BINARY P(Yes) strict-less-than v5 max
     on BOTH datasets)
  4. Ablation C aggregator robustness (max strict-beats mean, nor, top2,
     weighted on BOTH datasets, else retire)
  5. Ablation D prior-art self-check (v5 max strict-beats baseline,
     v3 p_evidence, v4 nor on BOTH datasets)
  6. Ablation E structured-output compliance rate >= 0.80 on BOTH datasets
  7. Ablation F per-rule mean-variance > 0.01 on train split on BOTH datasets
  8. N_test reconciliation: EN N=161, ZH N=149

Aggregators computed (primary = max):
    max        : max_i p_i
    mean       : mean_i p_i
    nor        : 1 - prod_i (1 - p_i)
    top2_mean  : mean of the top-2 p_i
    weighted   : sum_i w_i * p_i, weights = per-rule train positive rate / sum

Writes:
  - results/prompt_paradigm/report_v5.json
  - appends a v5 section to results/analysis/prompt_paradigm_report.md

Usage (always via sbatch):
  sbatch --cpus-per-task=2 --mem=4G --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh \
    && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 \
    && python src/prompt_paradigm/eval_per_rule.py"
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_utils import DATASET_ROOTS, load_annotations
from quick_eval_all import gmm_threshold, metrics, otsu_threshold

PROJECT_ROOT = "/data/jehc223/EMNLP2"

BASELINE = {
    "MHClip_EN": {"acc": 0.7640, "mf": 0.6532, "oracle": 0.7764,
                  "source": "2B binary_nodef TF-Otsu"},
    "MHClip_ZH": {"acc": 0.8121, "mf": 0.7871, "oracle": 0.8121,
                  "source": "2B binary_nodef TF-GMM"},
}

RULE_COUNT = {"MHClip_EN": 9, "MHClip_ZH": 8}


def load_jsonl(p):
    rows = []
    if not os.path.exists(p):
        return rows
    with open(p) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def build_label_map(dataset):
    ann = load_annotations(dataset)
    out = {}
    for vid, a in ann.items():
        label = a.get("hate_label", a.get("label"))
        if label in ("Hateful", "Offensive", 1):
            out[vid] = 1
        elif label in ("Normal", 0):
            out[vid] = 0
    return out


def acc_sweep(scores, labels):
    best_acc = 0.0
    best_t = None
    uniq = sorted(set(scores))
    if not uniq:
        return {"acc": 0.0, "t": None}
    for t in uniq:
        preds = np.asarray([1 if s >= t else 0 for s in scores])
        labs = np.asarray(labels)
        acc = float((preds == labs).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return {"acc": best_acc, "t": best_t}


def full_metrics_block(test_scores, labels, train_scores):
    scores_np = np.asarray(test_scores, dtype=float)
    labels_np = np.asarray(labels)
    train_np = np.asarray(train_scores, dtype=float) if len(train_scores) else None

    tf_otsu_t = otsu_threshold(scores_np)
    tf_gmm_t = gmm_threshold(scores_np)
    tr_otsu_t = otsu_threshold(train_np) if train_np is not None else tf_otsu_t
    tr_gmm_t = gmm_threshold(train_np) if train_np is not None else tf_gmm_t

    def block(t):
        m = metrics(scores_np, labels_np, t)
        return {
            "acc": m["acc"], "mp": m["mp"], "mr": m["mr"], "mf": m["mf"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
            "t": float(t),
        }

    pos = [s for s, y in zip(test_scores, labels) if y == 1]
    neg = [s for s, y in zip(test_scores, labels) if y == 0]
    pos_mean = float(np.mean(pos)) if pos else 0.0
    neg_mean = float(np.mean(neg)) if neg else 0.0
    wins = sum(1 for ps in pos for ns in neg if ps > ns)
    ties = sum(1 for ps in pos for ns in neg if ps == ns)
    denom = len(pos) * len(neg)
    auc = (wins + 0.5 * ties) / denom if denom else 0.0

    return {
        "n_test": len(test_scores),
        "n_test_pos": int(np.sum(labels_np)),
        "oracle": acc_sweep(test_scores, labels),
        "tf_otsu": block(tf_otsu_t),
        "tf_gmm": block(tf_gmm_t),
        "tr_otsu": block(tr_otsu_t),
        "tr_gmm": block(tr_gmm_t),
        "auc": float(auc),
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
    }


def load_v5_pair(dataset):
    """Return (test_rows_with_labels, train_rows).

    test: list of (p_rules[K], label, format_clean)
    train: list of p_rules[K]
    """
    root = PROJECT_ROOT
    ann_to_label = build_label_map(dataset)
    K = RULE_COUNT[dataset]

    test_path = os.path.join(root, "results", "prompt_paradigm", dataset, "test_per_rule.jsonl")
    train_path = os.path.join(root, "results", "prompt_paradigm", dataset, "train_per_rule.jsonl")

    split_csv = os.path.join(DATASET_ROOTS[dataset], "splits", "test_clean.csv")
    with open(split_csv) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    test_raw = {r["video_id"]: r for r in load_jsonl(test_path)}
    test_triples = []
    test_raw_all_clean = []  # (vid_id, format_clean) for all test rows that appear
    for vid in test_ids:
        r = test_raw.get(vid)
        if r is None:
            continue
        p_rules = r.get("p_rules")
        fc = bool(r.get("format_clean", False))
        test_raw_all_clean.append((vid, fc))
        if p_rules is None:
            continue
        valid = [p for p in p_rules if p is not None]
        if len(valid) == 0:
            continue
        if vid not in ann_to_label:
            continue
        # Pad None with 0.0 for aggregation safety (max, mean, etc. all skip Nones
        # via the None-aware aggregators below)
        test_triples.append((p_rules, ann_to_label[vid], fc))

    train_rows = load_jsonl(train_path)
    train_vecs = []
    for r in train_rows:
        p_rules = r.get("p_rules")
        if p_rules is None:
            continue
        valid = [p for p in p_rules if p is not None]
        if len(valid) == 0:
            continue
        train_vecs.append(p_rules)

    return test_triples, train_vecs, test_raw_all_clean, K


def load_baseline_scores(dataset):
    """Load the 2B binary_nodef baseline scores (test split) for Ablation A/D
    and Ablation D prior-art self-check."""
    ann_to_label = build_label_map(dataset)
    path = os.path.join(PROJECT_ROOT, "results", "holistic_2b", dataset, "test_binary.jsonl")
    rows = load_jsonl(path)
    by = {r["video_id"]: r for r in rows}

    split_csv = os.path.join(DATASET_ROOTS[dataset], "splits", "test_clean.csv")
    with open(split_csv) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    train_path = os.path.join(PROJECT_ROOT, "results", "holistic_2b", dataset, "train_binary.jsonl")
    train_rows = load_jsonl(train_path)
    train_scores = [r["score"] for r in train_rows if r.get("score") is not None]

    out = []
    for vid in test_ids:
        r = by.get(vid)
        if r is None or r.get("score") is None or vid not in ann_to_label:
            continue
        out.append((r["score"], ann_to_label[vid]))
    return out, train_scores


def load_v3_evidence_scores(dataset):
    ann_to_label = build_label_map(dataset)
    test_path = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", dataset, "test_polarity.jsonl")
    train_path = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", dataset, "train_polarity.jsonl")
    rows = load_jsonl(test_path)
    by = {r["video_id"]: r for r in rows}

    split_csv = os.path.join(DATASET_ROOTS[dataset], "splits", "test_clean.csv")
    with open(split_csv) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    train_rows = load_jsonl(train_path)
    train_scores = [r["p_evidence"] for r in train_rows if r.get("p_evidence") is not None]

    out = []
    for vid in test_ids:
        r = by.get(vid)
        if r is None or r.get("p_evidence") is None or vid not in ann_to_label:
            continue
        out.append((r["p_evidence"], ann_to_label[vid]))
    return out, train_scores


def load_v4_nor_scores(dataset):
    """Recompute v4 rank-noisy-OR fused score so we can include it in the
    prior-art comparison without depending on report_v4.json."""
    import bisect
    ann_to_label = build_label_map(dataset)

    test_path = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", dataset, "test_modality.jsonl")
    train_path = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", dataset, "train_modality.jsonl")
    if not os.path.exists(test_path) or not os.path.exists(train_path):
        return [], []

    split_csv = os.path.join(DATASET_ROOTS[dataset], "splits", "test_clean.csv")
    with open(split_csv) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    test_rows = {r["video_id"]: r for r in load_jsonl(test_path)}
    train_rows = load_jsonl(train_path)
    train_pairs = [(r["p_visual"], r["p_text"]) for r in train_rows
                   if r.get("p_visual") is not None and r.get("p_text") is not None]
    if not train_pairs:
        return [], []
    tr_pv = sorted([pv for pv, _ in train_pairs])
    tr_pt = sorted([pt for _, pt in train_pairs])

    def rank_of(v, s):
        return bisect.bisect_right(s, v) / len(s) if s else 0.5

    out = []
    for vid in test_ids:
        r = test_rows.get(vid)
        if r is None or r.get("p_visual") is None or r.get("p_text") is None:
            continue
        if vid not in ann_to_label:
            continue
        r_v = rank_of(r["p_visual"], tr_pv)
        r_t = rank_of(r["p_text"], tr_pt)
        score = 1.0 - (1.0 - r_v) * (1.0 - r_t)
        out.append((score, ann_to_label[vid]))

    # train scores in the same nor aggregation
    train_out = []
    for pv, pt in train_pairs:
        r_v = rank_of(pv, tr_pv)
        r_t = rank_of(pt, tr_pt)
        train_out.append(1.0 - (1.0 - r_v) * (1.0 - r_t))
    return out, train_out


# ---------------------------------------------------------------------------
# Aggregators (None-aware)
# ---------------------------------------------------------------------------

def _valid(p_rules):
    return [p for p in p_rules if p is not None]


def agg_max(p_rules):
    v = _valid(p_rules)
    return max(v) if v else 0.0


def agg_mean(p_rules):
    v = _valid(p_rules)
    return float(np.mean(v)) if v else 0.0


def agg_nor(p_rules):
    v = _valid(p_rules)
    if not v:
        return 0.0
    prod = 1.0
    for p in v:
        prod *= (1.0 - p)
    return 1.0 - prod


def agg_top2_mean(p_rules):
    v = _valid(p_rules)
    if not v:
        return 0.0
    v_sorted = sorted(v, reverse=True)
    return float(np.mean(v_sorted[:2])) if len(v_sorted) >= 1 else 0.0


def _build_weights(train_vecs, K):
    if not train_vecs:
        return [1.0 / K] * K
    means = [0.0] * K
    counts = [0] * K
    for vec in train_vecs:
        for i, p in enumerate(vec):
            if p is None:
                continue
            means[i] += p
            counts[i] += 1
    for i in range(K):
        means[i] = means[i] / counts[i] if counts[i] > 0 else 0.0
    total = sum(means)
    if total <= 0:
        return [1.0 / K] * K
    return [m / total for m in means]


def agg_weighted(p_rules, weights):
    out = 0.0
    used_w = 0.0
    for i, p in enumerate(p_rules):
        if p is None or i >= len(weights):
            continue
        out += weights[i] * p
        used_w += weights[i]
    return out / used_w if used_w > 0 else 0.0


AGGREGATORS = {
    "max": agg_max,
    "mean": agg_mean,
    "nor": agg_nor,
    "top2_mean": agg_top2_mean,
}


def compute_scores(test_triples, aggregator_fn, weights=None):
    scores = []
    labels = []
    for p_rules, y, _fc in test_triples:
        if aggregator_fn is agg_weighted:
            s = agg_weighted(p_rules, weights)
        else:
            s = aggregator_fn(p_rules)
        scores.append(s)
        labels.append(y)
    return scores, labels


def compute_train_scores(train_vecs, aggregator_fn, weights=None):
    out = []
    for p_rules in train_vecs:
        if aggregator_fn is agg_weighted:
            out.append(agg_weighted(p_rules, weights))
        else:
            out.append(aggregator_fn(p_rules))
    return out


# ---------------------------------------------------------------------------
# Per-rule statistics (Ablation B and F)
# ---------------------------------------------------------------------------

def per_rule_correlation(test_triples, K):
    """Per-rule point-biserial correlation with binary label."""
    labels = np.asarray([y for _, y, _ in test_triples], dtype=float)
    corrs = []
    for i in range(K):
        col = np.asarray([
            (p_rules[i] if (i < len(p_rules) and p_rules[i] is not None) else np.nan)
            for p_rules, _, _ in test_triples
        ])
        mask = ~np.isnan(col)
        if mask.sum() < 3 or labels[mask].std() == 0 or col[mask].std() == 0:
            corrs.append(0.0)
            continue
        corrs.append(float(np.corrcoef(col[mask], labels[mask])[0, 1]))
    return corrs


def per_rule_mean_variance(train_vecs, K):
    """Return per-rule mean on train, and variance of those means across rules."""
    if not train_vecs:
        return [], 0.0
    means = []
    for i in range(K):
        col = [vec[i] for vec in train_vecs
               if i < len(vec) and vec[i] is not None]
        means.append(float(np.mean(col)) if col else 0.0)
    return means, float(np.var(means))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[v5-eval] starting in {PROJECT_ROOT}")
    out = {"baseline": BASELINE, "aggregators": {}, "per_rule_stats": {}, "format_compliance": {}}

    for dataset in ("MHClip_EN", "MHClip_ZH"):
        K = RULE_COUNT[dataset]
        print(f"\n[v5-eval] === {dataset} (K={K}) ===")
        test_triples, train_vecs, test_all_rows, _K = load_v5_pair(dataset)
        print(f"[v5-eval]   test N={len(test_triples)}, train N={len(train_vecs)}, "
              f"all_test_rows_seen={len(test_all_rows)}")
        if not test_triples:
            print(f"[v5-eval]   WARNING: no test data for {dataset}")
            continue

        # Format compliance rate (Ablation E)
        n_rows = len(test_all_rows)
        n_clean = sum(1 for _, fc in test_all_rows if fc)
        clean_rate = n_clean / n_rows if n_rows else 0.0
        out["format_compliance"][dataset] = {
            "n_rows": n_rows,
            "n_clean": n_clean,
            "clean_rate": clean_rate,
        }
        print(f"[v5-eval]   format_compliance: {n_clean}/{n_rows} = {clean_rate:.3f}")

        # Per-rule weights for the weighted aggregator (from train)
        weights = _build_weights(train_vecs, K)

        # Main aggregators: max (primary), mean, nor, top2_mean, weighted
        for agg_name, fn in AGGREGATORS.items():
            test_scores, labels = compute_scores(test_triples, fn)
            train_scores = compute_train_scores(train_vecs, fn)
            block = full_metrics_block(test_scores, labels, train_scores)
            out["aggregators"].setdefault(agg_name, {})[dataset] = block
            print(f"[v5-eval]   {agg_name:10s} oracle={block['oracle']['acc']:.4f} "
                  f"tf_otsu_acc={block['tf_otsu']['acc']:.4f} "
                  f"tf_otsu_mf={block['tf_otsu']['mf']:.4f} "
                  f"auc={block['auc']:.4f}")

        # Weighted
        test_scores, labels = compute_scores(test_triples, agg_weighted, weights)
        train_scores = compute_train_scores(train_vecs, agg_weighted, weights)
        block = full_metrics_block(test_scores, labels, train_scores)
        out["aggregators"].setdefault("weighted", {})[dataset] = block
        print(f"[v5-eval]   weighted   oracle={block['oracle']['acc']:.4f} "
              f"tf_otsu_acc={block['tf_otsu']['acc']:.4f} "
              f"tf_otsu_mf={block['tf_otsu']['mf']:.4f} "
              f"auc={block['auc']:.4f}")

        # Baseline (2B binary_nodef) loaded from holistic_2b/.../test_binary.jsonl
        base_pairs, base_train = load_baseline_scores(dataset)
        if base_pairs:
            scores = [p for p, _ in base_pairs]
            labels_b = [y for _, y in base_pairs]
            block = full_metrics_block(scores, labels_b, base_train)
            out["aggregators"].setdefault("baseline_pyes", {})[dataset] = block
            print(f"[v5-eval]   baseline   oracle={block['oracle']['acc']:.4f} "
                  f"auc={block['auc']:.4f}")

        # v3 p_evidence (prior art)
        v3_pairs, v3_train = load_v3_evidence_scores(dataset)
        if v3_pairs:
            scores = [p for p, _ in v3_pairs]
            labels_b = [y for _, y in v3_pairs]
            block = full_metrics_block(scores, labels_b, v3_train)
            out["aggregators"].setdefault("v3_evidence", {})[dataset] = block
            print(f"[v5-eval]   v3_evid    oracle={block['oracle']['acc']:.4f} "
                  f"auc={block['auc']:.4f}")

        # v4 nor (prior art)
        v4_pairs, v4_train = load_v4_nor_scores(dataset)
        if v4_pairs:
            scores = [p for p, _ in v4_pairs]
            labels_b = [y for _, y in v4_pairs]
            block = full_metrics_block(scores, labels_b, v4_train)
            out["aggregators"].setdefault("v4_nor", {})[dataset] = block
            print(f"[v5-eval]   v4_nor     oracle={block['oracle']['acc']:.4f} "
                  f"auc={block['auc']:.4f}")

        # Per-rule stats
        rule_corrs = per_rule_correlation(test_triples, K)
        rule_means, rule_mean_var = per_rule_mean_variance(train_vecs, K)
        out["per_rule_stats"][dataset] = {
            "rule_corrs_test": rule_corrs,
            "rule_means_train": rule_means,
            "rule_mean_variance_train": rule_mean_var,
            "weights_used": weights,
        }
        print(f"[v5-eval]   per-rule test corrs: "
              f"{[round(c, 3) for c in rule_corrs]}")
        print(f"[v5-eval]   per-rule train means: "
              f"{[round(m, 3) for m in rule_means]} "
              f"var={rule_mean_var:.4f}")

    # -------- Binding clauses --------
    en_max = out["aggregators"].get("max", {}).get("MHClip_EN")
    zh_max = out["aggregators"].get("max", {}).get("MHClip_ZH")
    en_base = out["aggregators"].get("baseline_pyes", {}).get("MHClip_EN")
    zh_base = out["aggregators"].get("baseline_pyes", {}).get("MHClip_ZH")
    en_v3 = out["aggregators"].get("v3_evidence", {}).get("MHClip_EN")
    zh_v3 = out["aggregators"].get("v3_evidence", {}).get("MHClip_ZH")
    en_v4 = out["aggregators"].get("v4_nor", {}).get("MHClip_EN")
    zh_v4 = out["aggregators"].get("v4_nor", {}).get("MHClip_ZH")

    clauses = {}

    # Clause 1: oracle-first
    en_o = en_max["oracle"]["acc"] if en_max else None
    zh_o = zh_max["oracle"]["acc"] if zh_max else None
    clauses["1_oracle_first"] = {
        "en_oracle": en_o, "en_baseline_oracle": BASELINE["MHClip_EN"]["oracle"],
        "zh_oracle": zh_o, "zh_baseline_oracle": BASELINE["MHClip_ZH"]["oracle"],
        "passed": (en_o is not None and zh_o is not None
                   and en_o > BASELINE["MHClip_EN"]["oracle"]
                   and zh_o > BASELINE["MHClip_ZH"]["oracle"]),
    }

    # Clause 2: unified winner (mF1 non-regression)
    unified_winner = None
    for tname in ("tf_otsu", "tf_gmm", "tr_otsu", "tr_gmm"):
        if en_max is None or zh_max is None:
            break
        en_c = en_max[tname]; zh_c = zh_max[tname]
        if (en_c["acc"] > BASELINE["MHClip_EN"]["acc"]
                and zh_c["acc"] > BASELINE["MHClip_ZH"]["acc"]
                and en_c["mf"] >= BASELINE["MHClip_EN"]["mf"]
                and zh_c["mf"] >= BASELINE["MHClip_ZH"]["mf"]):
            unified_winner = {
                "threshold_family": tname,
                "MHClip_EN": en_c,
                "MHClip_ZH": zh_c,
            }
            break
    clauses["2_mf1_non_regression_unified"] = {
        "unified_winner": unified_winner,
        "passed": unified_winner is not None,
    }

    # Clause 3: Ablation A (baseline strict-less-than v5 max)
    if en_max and zh_max and en_base and zh_base:
        en_base_o = en_base["oracle"]["acc"]
        zh_base_o = zh_base["oracle"]["acc"]
        en_max_o = en_max["oracle"]["acc"]
        zh_max_o = zh_max["oracle"]["acc"]
        clauses["3_ablation_A_baseline_load_bearing"] = {
            "en_baseline_oracle": en_base_o,
            "en_v5_max_oracle": en_max_o,
            "zh_baseline_oracle": zh_base_o,
            "zh_v5_max_oracle": zh_max_o,
            "leak": (en_base_o >= en_max_o or zh_base_o >= zh_max_o),
        }
    else:
        clauses["3_ablation_A_baseline_load_bearing"] = {"leak": None, "note": "missing data"}

    # Clause 4: Ablation C aggregator robustness
    if en_max and zh_max:
        en_max_o = en_max["oracle"]["acc"]
        zh_max_o = zh_max["oracle"]["acc"]
        details = {}
        violation = False
        for alt_name in ("mean", "nor", "top2_mean", "weighted"):
            en_alt = out["aggregators"].get(alt_name, {}).get("MHClip_EN")
            zh_alt = out["aggregators"].get(alt_name, {}).get("MHClip_ZH")
            if en_alt is None or zh_alt is None:
                continue
            en_alt_o = en_alt["oracle"]["acc"]
            zh_alt_o = zh_alt["oracle"]["acc"]
            details[alt_name] = {"en_oracle": en_alt_o, "zh_oracle": zh_alt_o}
            if en_alt_o >= en_max_o and zh_alt_o >= zh_max_o:
                violation = True
        clauses["4_ablation_C_aggregator_robustness"] = {
            "en_max_oracle": en_max_o,
            "zh_max_oracle": zh_max_o,
            "alternatives": details,
            "violation": violation,
        }
    else:
        clauses["4_ablation_C_aggregator_robustness"] = {"violation": None}

    # Clause 5: Ablation D prior-art strict-beat
    if en_max and zh_max:
        en_max_o = en_max["oracle"]["acc"]
        zh_max_o = zh_max["oracle"]["acc"]
        prior = {}
        leak = False
        for src_name, en_src, zh_src in (
            ("baseline", en_base, zh_base),
            ("v3_evidence", en_v3, zh_v3),
            ("v4_nor", en_v4, zh_v4),
        ):
            if en_src is None or zh_src is None:
                prior[src_name] = {"note": "missing"}
                continue
            en_src_o = en_src["oracle"]["acc"]
            zh_src_o = zh_src["oracle"]["acc"]
            prior[src_name] = {"en_oracle": en_src_o, "zh_oracle": zh_src_o}
            # leak = prior ≥ v5 max on EITHER dataset
            if en_src_o >= en_max_o or zh_src_o >= zh_max_o:
                leak = True
        clauses["5_ablation_D_prior_art"] = {
            "en_v5_max_oracle": en_max_o,
            "zh_v5_max_oracle": zh_max_o,
            "prior": prior,
            "leak": leak,
        }
    else:
        clauses["5_ablation_D_prior_art"] = {"leak": None}

    # Clause 6: format compliance
    en_fc = out["format_compliance"].get("MHClip_EN", {}).get("clean_rate")
    zh_fc = out["format_compliance"].get("MHClip_ZH", {}).get("clean_rate")
    clauses["6_format_compliance"] = {
        "en_clean_rate": en_fc,
        "zh_clean_rate": zh_fc,
        "passed": (en_fc is not None and zh_fc is not None
                   and en_fc >= 0.80 and zh_fc >= 0.80),
    }

    # Clause 7: per-rule mean variance > 0.01
    en_var = out["per_rule_stats"].get("MHClip_EN", {}).get("rule_mean_variance_train")
    zh_var = out["per_rule_stats"].get("MHClip_ZH", {}).get("rule_mean_variance_train")
    clauses["7_per_rule_variance"] = {
        "en_variance_train": en_var,
        "zh_variance_train": zh_var,
        "passed": (en_var is not None and zh_var is not None
                   and en_var > 0.01 and zh_var > 0.01),
    }

    # Clause 8: N_test reconciliation
    en_n = en_max["n_test"] if en_max else None
    zh_n = zh_max["n_test"] if zh_max else None
    clauses["8_n_test_reconciliation"] = {
        "en_n_test": en_n, "en_expected": 161,
        "zh_n_test": zh_n, "zh_expected": 149,
        "passed": (en_n == 161 and zh_n == 149),
    }

    out["clauses"] = clauses
    out["unified_winner"] = unified_winner
    out["all_clauses_pass"] = (
        clauses["1_oracle_first"].get("passed") is True
        and clauses["2_mf1_non_regression_unified"].get("passed") is True
        and clauses["3_ablation_A_baseline_load_bearing"].get("leak") is False
        and clauses["4_ablation_C_aggregator_robustness"].get("violation") is False
        and clauses["5_ablation_D_prior_art"].get("leak") is False
        and clauses["6_format_compliance"].get("passed") is True
        and clauses["7_per_rule_variance"].get("passed") is True
        and clauses["8_n_test_reconciliation"].get("passed") is True
    )

    report_path = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", "report_v5.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[v5-eval] wrote {report_path}")
    print(f"[v5-eval] all_clauses_pass={out['all_clauses_pass']}")

    # Terse MD append
    md_path = os.path.join(PROJECT_ROOT, "results", "analysis", "prompt_paradigm_report.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "a") as f:
        f.write("\n\n## v5 — Per-Rule Disjunction Readout (2026-04-13)\n\n")
        f.write(f"- Clause 1 (Oracle-first): EN max {en_o} vs {BASELINE['MHClip_EN']['oracle']} | "
                f"ZH max {zh_o} vs {BASELINE['MHClip_ZH']['oracle']} | "
                f"passed={clauses['1_oracle_first']['passed']}\n")
        f.write(f"- Clause 2 (Unified-winner): "
                f"{unified_winner['threshold_family'] if unified_winner else 'NONE'}\n")
        f.write(f"- Clause 3 (Ablation A baseline leak): {clauses['3_ablation_A_baseline_load_bearing'].get('leak')}\n")
        f.write(f"- Clause 4 (Ablation C aggregator violation): "
                f"{clauses['4_ablation_C_aggregator_robustness'].get('violation')}\n")
        f.write(f"- Clause 5 (Ablation D prior-art leak): {clauses['5_ablation_D_prior_art'].get('leak')}\n")
        f.write(f"- Clause 6 (format compliance ≥ 0.80): "
                f"EN {en_fc:.3f} ZH {zh_fc:.3f} passed={clauses['6_format_compliance']['passed']}\n")
        f.write(f"- Clause 7 (per-rule variance > 0.01): "
                f"EN {en_var} ZH {zh_var} passed={clauses['7_per_rule_variance']['passed']}\n")
        f.write(f"- Clause 8 (N_test): EN={en_n} ZH={zh_n} passed={clauses['8_n_test_reconciliation']['passed']}\n")
        f.write(f"- **All clauses pass: {out['all_clauses_pass']}**\n")


if __name__ == "__main__":
    main()
