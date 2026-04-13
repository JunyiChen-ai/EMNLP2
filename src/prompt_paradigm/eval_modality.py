"""
prompt_paradigm v4 — Modality-Split Evidence Probes evaluator.

Reads results/prompt_paradigm/{dataset}/{train,test}_modality.jsonl which
contain raw (p_visual, p_text) per video, and reports all 7 binding Gate 2
clauses from docs/proposals/prompt_paradigm_v4.md §5.

Binding clauses:
  1. Oracle-first: EN oracle > 0.7764 AND ZH oracle > 0.8121 (strict)
  2. macro-F1 non-regression on the unified cell
  3. Ablation A/B load-bearing: if max(visual-only, text-only) oracle >=
     fused oracle on either dataset, v4 retires
  4. Ablation C prior-art self-check: if v3 p_evidence oracle >= v4 fused
     oracle on either dataset, v4 retires
  5. Ablation D aggregator robustness: if any of {prob-avg, rank-avg,
     rank-max} oracle >= rank-noisy-OR oracle on BOTH datasets, v4 retires
  6. Ablation E rescue-rate phenomenon check: cross-rescue >= 30% on both
     datasets AND rank correlation < 0.7
  7. N_test reconciliation: EN N=161, ZH N=149

Fusion (main aggregator = rank-space noisy-OR):

    r_v = rank_train(p_visual)            # empirical CDF against train-split
    r_t = rank_train(p_text)              # empirical CDF against train-split
    score = 1 - (1 - r_v)*(1 - r_t)

Writes:
  - results/prompt_paradigm/report_v4.json
  - appends a v4 section to results/analysis/prompt_paradigm_report.md

Usage (always via sbatch):
  sbatch --cpus-per-task=2 --mem=4G --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh \
    && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 \
    && python src/prompt_paradigm/eval_modality.py"
"""

import bisect
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


def rank_of(v, sorted_ref):
    if not sorted_ref:
        return 0.5
    return bisect.bisect_right(sorted_ref, v) / len(sorted_ref)


def build_label_map(dataset):
    ann = load_annotations(dataset)
    # map: vid_id -> binary label (1 = Hateful/Offensive, 0 = Normal)
    out = {}
    for vid, a in ann.items():
        label = a.get("hate_label", a.get("label"))
        if label in ("Hateful", "Offensive", 1):
            out[vid] = 1
        elif label in ("Normal", 0):
            out[vid] = 0
    return out


def acc_sweep(scores, labels):
    """Oracle ACC by sweeping unique score values (train-free oracle ceiling)."""
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
    """Return oracle + tf_otsu/tf_gmm/tr_otsu/tr_gmm + distribution stats."""
    scores_np = np.asarray(test_scores)
    labels_np = np.asarray(labels)
    train_np = np.asarray(train_scores) if len(train_scores) else None

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

    # distribution stats (for phenomenon check)
    pos = [s for s, y in zip(test_scores, labels) if y == 1]
    neg = [s for s, y in zip(test_scores, labels) if y == 0]
    pos_mean = float(np.mean(pos)) if pos else 0.0
    neg_mean = float(np.mean(neg)) if neg else 0.0
    pos_std = float(np.std(pos)) if pos else 0.0
    neg_std = float(np.std(neg)) if neg else 0.0

    # Pairwise AUC
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
        "pos_mean": pos_mean, "neg_mean": neg_mean,
        "pos_std": pos_std, "neg_std": neg_std,
    }


def load_v4_pair(dataset, split):
    """Return (test_rows_with_labels, train_rows) as lists of (p_vis, p_txt[, label])."""
    root = PROJECT_ROOT
    ann_to_label = build_label_map(dataset)

    test_path = os.path.join(root, "results", "prompt_paradigm", dataset, "test_modality.jsonl")
    train_path = os.path.join(root, "results", "prompt_paradigm", dataset, "train_modality.jsonl")

    # clean-split test filter
    split_csv = os.path.join(DATASET_ROOTS[dataset], "splits", "test_clean.csv")
    with open(split_csv) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    test_raw = {r["video_id"]: r for r in load_jsonl(test_path)}
    test_triples = []
    for vid in test_ids:
        r = test_raw.get(vid)
        if r is None:
            continue
        if r.get("p_visual") is None or r.get("p_text") is None:
            continue
        if vid not in ann_to_label:
            continue
        test_triples.append((r["p_visual"], r["p_text"], ann_to_label[vid]))

    train_rows = load_jsonl(train_path)
    train_pairs = [(r["p_visual"], r["p_text"]) for r in train_rows
                   if r.get("p_visual") is not None and r.get("p_text") is not None]

    return test_triples, train_pairs


def load_v3_evidence(dataset):
    """Load v3 p_evidence for Ablation C prior-art self-check. Filtered by clean split and annotations."""
    ann_to_label = build_label_map(dataset)
    path = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", dataset, "test_polarity.jsonl")
    rows = load_jsonl(path)
    by = {r["video_id"]: r for r in rows}

    split_csv = os.path.join(DATASET_ROOTS[dataset], "splits", "test_clean.csv")
    with open(split_csv) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    out = []
    for vid in test_ids:
        r = by.get(vid)
        if r is None or r.get("p_evidence") is None or vid not in ann_to_label:
            continue
        out.append((r["p_evidence"], ann_to_label[vid]))
    return out


def compute_rank_scores(test_triples, train_pairs, aggregator):
    """Return (list_of_scores, labels). aggregator in {nor, avg, max, prob_avg}."""
    tr_pv = sorted([pv for pv, _ in train_pairs])
    tr_pt = sorted([pt for _, pt in train_pairs])

    scores = []
    labels = []
    for pv, pt, y in test_triples:
        r_v = rank_of(pv, tr_pv)
        r_t = rank_of(pt, tr_pt)
        if aggregator == "nor":
            s = 1.0 - (1.0 - r_v) * (1.0 - r_t)
        elif aggregator == "avg":
            s = (r_v + r_t) / 2.0
        elif aggregator == "max":
            s = max(r_v, r_t)
        elif aggregator == "prob_avg":
            # prob-space average on raw probabilities (not ranks)
            s = (pv + pt) / 2.0
        else:
            raise ValueError(aggregator)
        scores.append(s)
        labels.append(y)
    return scores, labels


def compute_train_rank_scores(train_pairs, aggregator):
    tr_pv = sorted([pv for pv, _ in train_pairs])
    tr_pt = sorted([pt for _, pt in train_pairs])
    out = []
    for pv, pt in train_pairs:
        r_v = rank_of(pv, tr_pv)
        r_t = rank_of(pt, tr_pt)
        if aggregator == "nor":
            out.append(1.0 - (1.0 - r_v) * (1.0 - r_t))
        elif aggregator == "avg":
            out.append((r_v + r_t) / 2.0)
        elif aggregator == "max":
            out.append(max(r_v, r_t))
        elif aggregator == "prob_avg":
            out.append((pv + pt) / 2.0)
    return out


def compute_rescue_stats(test_triples, train_pairs):
    """Ablation E: rank correlation and cross-rescue rates on the test set."""
    tr_pv = sorted([pv for pv, _ in train_pairs])
    tr_pt = sorted([pt for _, pt in train_pairs])
    r_vs = []
    r_ts = []
    labels = []
    for pv, pt, y in test_triples:
        r_vs.append(rank_of(pv, tr_pv))
        r_ts.append(rank_of(pt, tr_pt))
        labels.append(y)

    r_vs = np.array(r_vs); r_ts = np.array(r_ts); labels_np = np.array(labels)
    corr = float(np.corrcoef(r_vs, r_ts)[0, 1]) if len(r_vs) > 1 else 0.0

    # Rescue rates, computed on positives:
    # "text rescues visual" = positives in the bottom half by r_v that are in
    #   top half by r_t
    # "visual rescues text" = positives in the bottom half by r_t that are in
    #   top half by r_v
    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    if not pos_idx:
        return {"corr": corr, "text_rescues_visual": 0.0,
                "visual_rescues_text": 0.0, "n_pos": 0}

    pos_rv = r_vs[pos_idx]
    pos_rt = r_ts[pos_idx]
    # use median of positives as the "low" threshold
    med_rv = float(np.median(pos_rv))
    med_rt = float(np.median(pos_rt))

    bot_rv = pos_rv < med_rv  # positives Call 1 underestimates
    bot_rt = pos_rt < med_rt  # positives Call 2 underestimates

    # rescue target = paired rank above the population median of the OTHER probe
    other_rv_med = float(np.median(r_vs))
    other_rt_med = float(np.median(r_ts))

    if bot_rv.sum() > 0:
        text_rescues_visual = float(((pos_rt[bot_rv] > other_rt_med)).mean())
    else:
        text_rescues_visual = 0.0
    if bot_rt.sum() > 0:
        visual_rescues_text = float(((pos_rv[bot_rt] > other_rv_med)).mean())
    else:
        visual_rescues_text = 0.0

    return {
        "corr": corr,
        "text_rescues_visual": text_rescues_visual,
        "visual_rescues_text": visual_rescues_text,
        "n_pos": len(pos_idx),
    }


def main():
    print(f"[v4-eval] starting in {PROJECT_ROOT}")
    out = {"baseline": BASELINE, "aggregators": {}}

    for dataset in ("MHClip_EN", "MHClip_ZH"):
        print(f"\n[v4-eval] === {dataset} ===")
        test_triples, train_pairs = load_v4_pair(dataset, "test")
        print(f"[v4-eval]   test N={len(test_triples)}, train N={len(train_pairs)}")
        if not test_triples:
            print(f"[v4-eval]   WARNING: no test data for {dataset}")
            continue

        # Primary aggregator + 3 alternatives for Ablation D
        for agg in ("nor", "avg", "max", "prob_avg"):
            test_scores, labels = compute_rank_scores(test_triples, train_pairs, agg)
            train_scores = compute_train_rank_scores(train_pairs, agg)
            block = full_metrics_block(test_scores, labels, train_scores)
            out["aggregators"].setdefault(agg, {})[dataset] = block
            print(f"[v4-eval]   {agg:10s} oracle={block['oracle']['acc']:.4f} "
                  f"tf_otsu_acc={block['tf_otsu']['acc']:.4f} "
                  f"tf_otsu_mf={block['tf_otsu']['mf']:.4f} "
                  f"auc={block['auc']:.4f}")

        # Ablation A/B: visual-only, text-only (raw probabilities)
        for name, idx in (("visual_only", 0), ("text_only", 1)):
            scores = [t[idx] for t in test_triples]
            labels = [t[2] for t in test_triples]
            train_scores = [p[idx] for p in train_pairs]
            block = full_metrics_block(scores, labels, train_scores)
            out["aggregators"].setdefault(name, {})[dataset] = block
            print(f"[v4-eval]   {name:12s} oracle={block['oracle']['acc']:.4f} "
                  f"auc={block['auc']:.4f}")

        # Ablation C: v3 p_evidence as prior-art replica
        v3_pairs = load_v3_evidence(dataset)
        if v3_pairs:
            scores = [p for p, _ in v3_pairs]
            labels = [y for _, y in v3_pairs]
            block = full_metrics_block(scores, labels, [])
            out["aggregators"].setdefault("v3_evidence", {})[dataset] = block
            print(f"[v4-eval]   v3_evidence  oracle={block['oracle']['acc']:.4f} "
                  f"auc={block['auc']:.4f}")

        # Ablation E: rescue rates, rank correlation
        rescue = compute_rescue_stats(test_triples, train_pairs)
        out.setdefault("ablation_e", {})[dataset] = rescue
        print(f"[v4-eval]   ablation_e corr={rescue['corr']:.3f} "
              f"text_rescues_visual={rescue['text_rescues_visual']:.2%} "
              f"visual_rescues_text={rescue['visual_rescues_text']:.2%}")

    # --- Binding clause verdicts ---
    en_nor = out["aggregators"].get("nor", {}).get("MHClip_EN")
    zh_nor = out["aggregators"].get("nor", {}).get("MHClip_ZH")
    en_vis = out["aggregators"].get("visual_only", {}).get("MHClip_EN")
    zh_vis = out["aggregators"].get("visual_only", {}).get("MHClip_ZH")
    en_txt = out["aggregators"].get("text_only", {}).get("MHClip_EN")
    zh_txt = out["aggregators"].get("text_only", {}).get("MHClip_ZH")
    en_v3  = out["aggregators"].get("v3_evidence", {}).get("MHClip_EN")
    zh_v3  = out["aggregators"].get("v3_evidence", {}).get("MHClip_ZH")

    clauses = {}

    # Clause 1: oracle-first (strict beat on both)
    en_o = en_nor["oracle"]["acc"] if en_nor else None
    zh_o = zh_nor["oracle"]["acc"] if zh_nor else None
    clauses["1_oracle_first"] = {
        "en_oracle": en_o, "en_baseline_oracle": BASELINE["MHClip_EN"]["oracle"],
        "zh_oracle": zh_o, "zh_baseline_oracle": BASELINE["MHClip_ZH"]["oracle"],
        "passed": (en_o is not None and zh_o is not None
                   and en_o > BASELINE["MHClip_EN"]["oracle"]
                   and zh_o > BASELINE["MHClip_ZH"]["oracle"]),
    }

    # Clause 2: mF1 non-regression on unified cell (find best threshold family that beats
    # both ACCs, check mF1 non-regression on that family)
    unified_winner = None
    for tname in ("tf_otsu", "tf_gmm", "tr_otsu", "tr_gmm"):
        if en_nor is None or zh_nor is None:
            break
        en_c = en_nor[tname]; zh_c = zh_nor[tname]
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

    # Clause 3: Ablation A/B load-bearing
    if en_vis and zh_vis and en_txt and zh_txt and en_nor and zh_nor:
        en_max = max(en_vis["oracle"]["acc"], en_txt["oracle"]["acc"])
        zh_max = max(zh_vis["oracle"]["acc"], zh_txt["oracle"]["acc"])
        clauses["3_ablation_AB_load_bearing"] = {
            "en_max_single": en_max,
            "en_fused": en_nor["oracle"]["acc"],
            "zh_max_single": zh_max,
            "zh_fused": zh_nor["oracle"]["acc"],
            "leak": (en_max >= en_nor["oracle"]["acc"]
                     or zh_max >= zh_nor["oracle"]["acc"]),
        }
    else:
        clauses["3_ablation_AB_load_bearing"] = {"leak": None, "note": "missing data"}

    # Clause 4: Ablation C prior-art self-check (vs v3 p_evidence)
    if en_v3 and zh_v3 and en_nor and zh_nor:
        en_v3o = en_v3["oracle"]["acc"]
        zh_v3o = zh_v3["oracle"]["acc"]
        clauses["4_ablation_C_prior_art"] = {
            "en_v3_evidence_oracle": en_v3o,
            "en_v4_fused_oracle": en_nor["oracle"]["acc"],
            "zh_v3_evidence_oracle": zh_v3o,
            "zh_v4_fused_oracle": zh_nor["oracle"]["acc"],
            "leak": (en_v3o >= en_nor["oracle"]["acc"]
                     or zh_v3o >= zh_nor["oracle"]["acc"]),
        }
    else:
        clauses["4_ablation_C_prior_art"] = {"leak": None, "note": "missing data"}

    # Clause 5: Ablation D aggregator robustness
    en_avg = out["aggregators"].get("avg", {}).get("MHClip_EN")
    zh_avg = out["aggregators"].get("avg", {}).get("MHClip_ZH")
    en_mx  = out["aggregators"].get("max", {}).get("MHClip_EN")
    zh_mx  = out["aggregators"].get("max", {}).get("MHClip_ZH")
    en_pa  = out["aggregators"].get("prob_avg", {}).get("MHClip_EN")
    zh_pa  = out["aggregators"].get("prob_avg", {}).get("MHClip_ZH")
    if all(x is not None for x in (en_nor, zh_nor, en_avg, zh_avg,
                                    en_mx, zh_mx, en_pa, zh_pa)):
        violation = False
        details = {}
        for alt_name, en_alt, zh_alt in (("avg", en_avg, zh_avg),
                                          ("max", en_mx, zh_mx),
                                          ("prob_avg", en_pa, zh_pa)):
            en_alt_o = en_alt["oracle"]["acc"]
            zh_alt_o = zh_alt["oracle"]["acc"]
            details[alt_name] = {
                "en_oracle": en_alt_o,
                "zh_oracle": zh_alt_o,
            }
            if (en_alt_o >= en_nor["oracle"]["acc"]
                    and zh_alt_o >= zh_nor["oracle"]["acc"]):
                violation = True
        clauses["5_ablation_D_aggregator_robustness"] = {
            "en_nor_oracle": en_nor["oracle"]["acc"],
            "zh_nor_oracle": zh_nor["oracle"]["acc"],
            "alternatives": details,
            "violation": violation,
        }
    else:
        clauses["5_ablation_D_aggregator_robustness"] = {"violation": None}

    # Clause 6: Ablation E rescue-rate phenomenon check
    abl_e = out.get("ablation_e", {})
    if abl_e.get("MHClip_EN") and abl_e.get("MHClip_ZH"):
        en_e = abl_e["MHClip_EN"]; zh_e = abl_e["MHClip_ZH"]
        en_rescue_max = max(en_e["text_rescues_visual"], en_e["visual_rescues_text"])
        zh_rescue_max = max(zh_e["text_rescues_visual"], zh_e["visual_rescues_text"])
        en_ok = (en_rescue_max >= 0.30) and (en_e["corr"] < 0.7)
        zh_ok = (zh_rescue_max >= 0.30) and (zh_e["corr"] < 0.7)
        clauses["6_ablation_E_rescue_phenomenon"] = {
            "en": en_e, "zh": zh_e,
            "en_passed": en_ok, "zh_passed": zh_ok,
            "passed": en_ok and zh_ok,
        }
    else:
        clauses["6_ablation_E_rescue_phenomenon"] = {"passed": None}

    # Clause 7: N_test reconciliation
    en_n = en_nor["n_test"] if en_nor else None
    zh_n = zh_nor["n_test"] if zh_nor else None
    clauses["7_n_test_reconciliation"] = {
        "en_n_test": en_n, "en_expected": 161,
        "zh_n_test": zh_n, "zh_expected": 149,
        "passed": (en_n == 161 and zh_n == 149),
    }

    out["clauses"] = clauses
    out["unified_winner"] = unified_winner
    out["all_clauses_pass"] = all(
        (c.get("passed") is True) or (c.get("leak") is False)
        or (c.get("violation") is False)
        for c in clauses.values()
    )

    report_path = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", "report_v4.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[v4-eval] wrote {report_path}")

    # Terse MD summary
    md_path = os.path.join(PROJECT_ROOT, "results", "analysis", "prompt_paradigm_report.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "a") as f:
        f.write("\n\n## v4 — Modality-Split Evidence Probes (2026-04-13)\n\n")
        f.write(f"- Oracle-first: EN {en_o} vs {BASELINE['MHClip_EN']['oracle']} | "
                f"ZH {zh_o} vs {BASELINE['MHClip_ZH']['oracle']} | "
                f"passed={clauses['1_oracle_first']['passed']}\n")
        f.write(f"- Unified-winner: {unified_winner['threshold_family'] if unified_winner else 'NONE'}\n")
        f.write(f"- Ablation A/B leak: {clauses['3_ablation_AB_load_bearing'].get('leak')}\n")
        f.write(f"- Ablation C leak:  {clauses['4_ablation_C_prior_art'].get('leak')}\n")
        f.write(f"- Ablation D violation: {clauses['5_ablation_D_aggregator_robustness'].get('violation')}\n")
        f.write(f"- Ablation E passed: {clauses['6_ablation_E_rescue_phenomenon'].get('passed')}\n")
        f.write(f"- All clauses pass: {out['all_clauses_pass']}\n")


if __name__ == "__main__":
    main()
