"""
prompt_paradigm v3 evaluator — Polarity-Calibrated Probes.

Gate 2 verification for v3 with FOUR binding clauses from director's Gate 1 approval:

  1. Oracle-first pre-check. If fused-score test-oracle ACC <= baseline oracle on
     EITHER dataset (EN 0.7764 / ZH 0.8121), v3 is falsified at the rank level —
     report MISS, do NOT proceed to threshold-cell selection.

  2. AP1 self-binding. If Ablation B (probability-space average) matches or beats
     the logit-space fused score on oracle ACC on EITHER dataset, v3 is retired as
     an anti-pattern-1 violation — the log-odds framing is not load-bearing and the
     method collapses to a prompt-average ensemble.

  3. Ablation load-bearing. If Ablation A (Evidence-Probe alone = baseline) matches
     or beats the fused score, v3 collapses to baseline and is a null result.

  4. Ablation E sign check. The bias-cancellation story predicts
       sign(pos_mean_fused − pos_mean_evidence)  = POSITIVE (Call 1 is FN-biased,
                                                              fusion lifts positives)
       sign(neg_mean_fused − (1 − mean(p_compliance)_neg))  = NEGATIVE (Call 2 after
                                                              negation is FP-biased,
                                                              fusion lowers negatives)
     If the observed sign differs, the mechanism is empirically falsified on 2B and
     v3 is dead regardless of raw numbers.

Imports frozen back-half functions from src/quick_eval_all.py by reference only.

Reads:
  results/prompt_paradigm/{MHClip_EN,MHClip_ZH}/{test,train}_polarity.jsonl

Writes:
  results/prompt_paradigm/report_v3.json
  results/analysis/prompt_paradigm_report.md  (overwrites v2 report — permitted)
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_utils import load_annotations
from quick_eval_all import (
    otsu_threshold,
    gmm_threshold,
    metrics,
    build_arrays,
)

PROJECT_ROOT = "/data/jehc223/EMNLP2"

# Baselines from project_current_baseline.md + director's Gate-1-v3 ruling.
BASELINE = {
    "MHClip_EN": {
        "acc": 0.7640,
        "mf": 0.6532,
        "oracle": 0.7764,
        "source": "2B binary_nodef TF-Otsu",
    },
    "MHClip_ZH": {
        "acc": 0.8121,
        "mf": 0.7871,
        "oracle": 0.8121,
        "source": "2B binary_nodef TF-GMM",
    },
}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def fuse_logit(p_e, p_c):
    """v3 main: logit-space bias-cancellation."""
    if p_e is None or p_c is None:
        return None
    return sigmoid(0.5 * (logit(p_e) - logit(p_c)))


def fuse_prob(p_e, p_c):
    """Ablation B: probability-space average of Evidence and (1-Compliance)."""
    if p_e is None or p_c is None:
        return None
    return 0.5 * (p_e + (1.0 - p_c))


def compliance_negated(p_e, p_c):
    """Ablation D: Call 2 alone, hate-aligned (1 - p_compliance)."""
    if p_c is None:
        return None
    return 1.0 - p_c


def evidence_only(p_e, p_c):
    """Ablation A: Call 1 alone (baseline replica)."""
    return p_e


def load_polarity(path):
    """Load polarity JSONL. Returns dict[vid -> {p_evidence, p_compliance, score}]."""
    d = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vid = r.get("video_id")
            if vid is None:
                continue
            p_e = r.get("p_evidence")
            p_c = r.get("p_compliance")
            score = r.get("score")
            if p_e is None or p_c is None or score is None:
                continue
            d[vid] = {
                "p_evidence": float(p_e),
                "p_compliance": float(p_c),
                "score": float(score),
            }
    return d


def oracle_acc(xs, ys):
    best = 0.0
    best_t = 0.5
    for t in np.arange(0.0, 1.001, 0.01):
        m = metrics(xs, ys, float(t))
        if m["acc"] > best:
            best = m["acc"]
            best_t = float(t)
    return best, best_t


def four_cells(test_x, test_y, train_x):
    out = {}
    try:
        t = otsu_threshold(test_x)
        out["tf_otsu"] = {**metrics(test_x, test_y, t), "t": t}
    except Exception as e:
        out["tf_otsu"] = {"err": str(e)}
    try:
        t = gmm_threshold(test_x)
        out["tf_gmm"] = {**metrics(test_x, test_y, t), "t": t}
    except Exception as e:
        out["tf_gmm"] = {"err": str(e)}
    if train_x is not None and len(train_x) > 0:
        try:
            t = otsu_threshold(train_x)
            out["tr_otsu"] = {**metrics(test_x, test_y, t), "t": t}
        except Exception as e:
            out["tr_otsu"] = {"err": str(e)}
        try:
            t = gmm_threshold(train_x)
            out["tr_gmm"] = {**metrics(test_x, test_y, t), "t": t}
        except Exception as e:
            out["tr_gmm"] = {"err": str(e)}
    return out


def evaluate_aggregator(ds, d_test, d_train, fn, label):
    """Evaluate a single aggregation fn(p_e, p_c) on one dataset."""
    ann = load_annotations(ds)
    simple_test = {}
    for vid, rec in d_test.items():
        s = fn(rec["p_evidence"], rec["p_compliance"])
        if s is None:
            continue
        simple_test[vid] = s
    test_x, test_y = build_arrays(simple_test, ann)
    if len(test_x) == 0:
        return {"error": "no test overlap"}

    train_x = None
    if d_train:
        simple_train = {}
        for vid, rec in d_train.items():
            s = fn(rec["p_evidence"], rec["p_compliance"])
            if s is None:
                continue
            simple_train[vid] = s
        train_x = np.array(list(simple_train.values()))

    oracle, oracle_t = oracle_acc(test_x, test_y)
    res = {
        "label": label,
        "n_test": int(len(test_x)),
        "n_test_pos": int(test_y.sum()),
        "oracle": {"acc": oracle, "t": oracle_t},
        **four_cells(test_x, test_y, train_x),
    }
    pos = test_x[test_y == 1]
    neg = test_x[test_y == 0]
    res["pos_mean"] = float(pos.mean()) if len(pos) else None
    res["neg_mean"] = float(neg.mean()) if len(neg) else None
    res["pos_std"] = float(pos.std()) if len(pos) else None
    res["neg_std"] = float(neg.std()) if len(neg) else None
    # Bhattacharyya overlap (Gaussian approx) between pos and neg distributions
    if len(pos) > 1 and len(neg) > 1:
        mu1 = pos.mean(); mu2 = neg.mean()
        v1 = pos.var() + 1e-9; v2 = neg.var() + 1e-9
        bc = 0.25 * math.log(0.25 * (v1 / v2 + v2 / v1 + 2)) \
             + 0.25 * ((mu1 - mu2) ** 2) / (v1 + v2)
        res["bhattacharyya_dist"] = float(bc)
        res["bhattacharyya_overlap"] = float(math.exp(-bc))
    return res


def find_unified(en_row, zh_row):
    """Return (cell, en_m, zh_m) for the unified cell that strict-beats baseline
    on BOTH datasets (ACC strict >, mF1 >=), or None."""
    cells = ["tf_otsu", "tf_gmm", "tr_otsu", "tr_gmm"]
    passing = []
    for c in cells:
        e = en_row.get(c)
        z = zh_row.get(c)
        if not e or not z or "err" in e or "err" in z:
            continue
        en_ok = (e["acc"] > BASELINE["MHClip_EN"]["acc"]
                 and e["mf"] >= BASELINE["MHClip_EN"]["mf"])
        zh_ok = (z["acc"] > BASELINE["MHClip_ZH"]["acc"]
                 and z["mf"] >= BASELINE["MHClip_ZH"]["mf"])
        if en_ok and zh_ok:
            passing.append((c, e, z))
    if not passing:
        return None
    passing.sort(
        key=lambda x: min(
            x[1]["acc"] - BASELINE["MHClip_EN"]["acc"],
            x[2]["acc"] - BASELINE["MHClip_ZH"]["acc"],
        ),
        reverse=True,
    )
    return passing[0]


def fmt(m):
    if not m or "err" in m:
        return "ERR"
    if "acc" not in m:
        return "—"
    if "mf" in m:
        return f"{m['acc']:.4f}/{m['mf']:.3f}"
    return f"{m['acc']:.4f}"


def sign_check_ablation_e(per_agg_rows):
    """Check per-class-mean drift signs predicted by §4 Ablation E.

    Prediction (from v3 proposal):
      (a) fused pos_mean > evidence_only pos_mean (Call 1 FN-bias lifted)
      (b) fused neg_mean < compliance_negated neg_mean (Call 2-negated FP-bias lowered)

    Returns per-dataset dict with observed signs and PASS/FAIL.
    """
    out = {}
    for ds in ["MHClip_EN", "MHClip_ZH"]:
        fused = per_agg_rows.get("logit_fused", {}).get(ds, {})
        evid = per_agg_rows.get("evidence_only", {}).get(ds, {})
        cneg = per_agg_rows.get("compliance_negated", {}).get(ds, {})
        if any("error" in r for r in [fused, evid, cneg]):
            out[ds] = {"status": "err"}
            continue
        pos_fused = fused.get("pos_mean")
        pos_evid = evid.get("pos_mean")
        neg_fused = fused.get("neg_mean")
        neg_cneg = cneg.get("neg_mean")
        if None in (pos_fused, pos_evid, neg_fused, neg_cneg):
            out[ds] = {"status": "missing_stats"}
            continue
        delta_pos = pos_fused - pos_evid
        delta_neg = neg_fused - neg_cneg
        out[ds] = {
            "pos_mean_fused": pos_fused,
            "pos_mean_evidence": pos_evid,
            "delta_pos": delta_pos,
            "pos_sign_expected": "+",
            "pos_sign_observed": "+" if delta_pos > 0 else "-" if delta_pos < 0 else "0",
            "neg_mean_fused": neg_fused,
            "neg_mean_compl_neg": neg_cneg,
            "delta_neg": delta_neg,
            "neg_sign_expected": "-",
            "neg_sign_observed": "-" if delta_neg < 0 else "+" if delta_neg > 0 else "0",
            "passed": (delta_pos > 0 and delta_neg < 0),
        }
    return out


def main():
    out_dir = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm")
    analysis_dir = os.path.join(PROJECT_ROOT, "results", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    data = {}
    for ds in ["MHClip_EN", "MHClip_ZH"]:
        base = os.path.join(out_dir, ds)
        test_path = os.path.join(base, "test_polarity.jsonl")
        train_path = os.path.join(base, "train_polarity.jsonl")
        if not os.path.isfile(test_path):
            data[ds] = {"error": f"missing {test_path}"}
            continue
        d_test = load_polarity(test_path)
        d_train = load_polarity(train_path) if os.path.isfile(train_path) else {}
        data[ds] = {"test": d_test, "train": d_train}

    aggregators = [
        ("logit_fused", fuse_logit),                 # v3 main claim
        ("prob_average", fuse_prob),                 # Ablation B (AP1 self-binding test)
        ("evidence_only", evidence_only),            # Ablation A (baseline replica)
        ("compliance_negated", compliance_negated),  # Ablation D (deflected alone)
    ]

    per_agg_rows = {}
    for agg_name, fn in aggregators:
        per_agg_rows[agg_name] = {}
        for ds in ["MHClip_EN", "MHClip_ZH"]:
            if "error" in data[ds]:
                per_agg_rows[agg_name][ds] = {"error": data[ds]["error"]}
                continue
            row = evaluate_aggregator(
                ds, data[ds]["test"], data[ds]["train"], fn, agg_name
            )
            per_agg_rows[agg_name][ds] = row

    report = {"baseline": BASELINE, "aggregators": per_agg_rows}

    # ----- Clause 1: Oracle-first pre-check on logit_fused -----
    fused_en = per_agg_rows["logit_fused"].get("MHClip_EN", {})
    fused_zh = per_agg_rows["logit_fused"].get("MHClip_ZH", {})
    en_oracle = fused_en.get("oracle", {}).get("acc")
    zh_oracle = fused_zh.get("oracle", {}).get("acc")
    oracle_pass = (
        en_oracle is not None and zh_oracle is not None
        and en_oracle > BASELINE["MHClip_EN"]["oracle"]
        and zh_oracle > BASELINE["MHClip_ZH"]["oracle"]
    )
    report["oracle_precheck"] = {
        "en_oracle": en_oracle,
        "en_baseline_oracle": BASELINE["MHClip_EN"]["oracle"],
        "zh_oracle": zh_oracle,
        "zh_baseline_oracle": BASELINE["MHClip_ZH"]["oracle"],
        "passed": oracle_pass,
    }

    # ----- Clause 2: AP1 self-binding — probability-space average comparison -----
    prob_en = per_agg_rows["prob_average"].get("MHClip_EN", {})
    prob_zh = per_agg_rows["prob_average"].get("MHClip_ZH", {})
    prob_en_or = prob_en.get("oracle", {}).get("acc")
    prob_zh_or = prob_zh.get("oracle", {}).get("acc")
    ap1_violation = False
    if en_oracle is not None and prob_en_or is not None and prob_en_or >= en_oracle:
        ap1_violation = True
    if zh_oracle is not None and prob_zh_or is not None and prob_zh_or >= zh_oracle:
        ap1_violation = True
    report["ap1_self_binding"] = {
        "prob_avg_en_oracle": prob_en_or,
        "logit_fused_en_oracle": en_oracle,
        "prob_avg_zh_oracle": prob_zh_or,
        "logit_fused_zh_oracle": zh_oracle,
        "violation": ap1_violation,
        "note": (
            "AP1 self-binding: if prob-space average oracle >= logit-space fused oracle "
            "on EITHER dataset, v3 is retired as ensembling-in-disguise."
        ),
    }

    # ----- Clause 3: Ablation A load-bearing — Evidence-only replica -----
    evid_en = per_agg_rows["evidence_only"].get("MHClip_EN", {})
    evid_zh = per_agg_rows["evidence_only"].get("MHClip_ZH", {})
    evid_en_or = evid_en.get("oracle", {}).get("acc")
    evid_zh_or = evid_zh.get("oracle", {}).get("acc")
    ablation_a_leak = False
    if en_oracle is not None and evid_en_or is not None and evid_en_or >= en_oracle:
        ablation_a_leak = True
    if zh_oracle is not None and evid_zh_or is not None and evid_zh_or >= zh_oracle:
        ablation_a_leak = True
    report["ablation_a_load_bearing"] = {
        "evidence_only_en_oracle": evid_en_or,
        "logit_fused_en_oracle": en_oracle,
        "evidence_only_zh_oracle": evid_zh_or,
        "logit_fused_zh_oracle": zh_oracle,
        "leak": ablation_a_leak,
        "note": (
            "Ablation A: if evidence-only (=baseline replica) oracle >= fused oracle, "
            "v3 collapses to baseline. Also serves as integrity check — should "
            "approximately match baseline numbers."
        ),
    }

    # ----- Clause 4: Ablation E sign check -----
    sign_check = sign_check_ablation_e(per_agg_rows)
    report["ablation_e_sign_check"] = sign_check
    sign_pass = all(
        v.get("passed", False) for v in sign_check.values()
        if "status" not in v
    ) and all("status" not in v for v in sign_check.values())

    # ----- Unified-cell search (only meaningful if all clauses pass) -----
    all_clauses_pass = (
        oracle_pass
        and not ap1_violation
        and not ablation_a_leak
        and sign_pass
    )
    unified = None
    if all_clauses_pass:
        unified = find_unified(fused_en, fused_zh)
    if unified:
        cell, en_m, zh_m = unified
        report["unified_winner"] = {
            "cell": cell,
            "EN": {"acc": en_m["acc"], "mf": en_m["mf"], "t": en_m.get("t")},
            "ZH": {"acc": zh_m["acc"], "mf": zh_m["mf"], "t": zh_m.get("t")},
        }
    else:
        report["unified_winner"] = None

    report["all_clauses_pass"] = bool(all_clauses_pass)

    # ----- Write JSON -----
    with open(os.path.join(out_dir, "report_v3.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ----- Markdown summary -----
    lines = []
    lines.append("# prompt_paradigm v3 report — Polarity-Calibrated Probes")
    lines.append("")
    lines.append("Literature-grounded: Zhao et al. ICML 2021 (Contextual Calibration); "
                 "Zhou et al. 2022 (Prompt Consistency); POROver / SafeCoDe "
                 "(RLHF asymmetric-refusal bias, background).")
    lines.append("")
    lines.append("## Baseline (must strict-beat on BOTH under one unified cell)")
    lines.append("")
    lines.append(
        f"- MHClip_EN: ACC={BASELINE['MHClip_EN']['acc']:.4f} "
        f"macro-F1={BASELINE['MHClip_EN']['mf']:.4f} "
        f"(oracle ceiling {BASELINE['MHClip_EN']['oracle']:.4f}) "
        f"[{BASELINE['MHClip_EN']['source']}]"
    )
    lines.append(
        f"- MHClip_ZH: ACC={BASELINE['MHClip_ZH']['acc']:.4f} "
        f"macro-F1={BASELINE['MHClip_ZH']['mf']:.4f} "
        f"(oracle ceiling {BASELINE['MHClip_ZH']['oracle']:.4f}) "
        f"[{BASELINE['MHClip_ZH']['source']}]"
    )
    lines.append("")

    # Clause 1: Oracle-first
    lines.append("## Clause 1 — Oracle-first pre-check (Gate-2 prerequisite)")
    lines.append("")
    en_or_str = f"{en_oracle:.4f}" if en_oracle is not None else "—"
    zh_or_str = f"{zh_oracle:.4f}" if zh_oracle is not None else "—"
    en_or_verdict = ("PASS" if en_oracle is not None
                     and en_oracle > BASELINE["MHClip_EN"]["oracle"] else "FAIL")
    zh_or_verdict = ("PASS" if zh_oracle is not None
                     and zh_oracle > BASELINE["MHClip_ZH"]["oracle"] else "FAIL")
    lines.append(f"- EN oracle (logit_fused): {en_or_str} vs baseline "
                 f"{BASELINE['MHClip_EN']['oracle']:.4f} → **{en_or_verdict}**")
    lines.append(f"- ZH oracle (logit_fused): {zh_or_str} vs baseline "
                 f"{BASELINE['MHClip_ZH']['oracle']:.4f} → **{zh_or_verdict}**")
    lines.append(f"- **Clause 1 overall: {'PASS' if oracle_pass else 'FAIL'}**")
    lines.append("")

    # Clause 2: AP1 self-binding
    lines.append("## Clause 2 — AP1 self-binding (prob-avg vs logit-fusion)")
    lines.append("")
    pa_en_str = f"{prob_en_or:.4f}" if prob_en_or is not None else "—"
    pa_zh_str = f"{prob_zh_or:.4f}" if prob_zh_or is not None else "—"
    lines.append(f"- EN prob-avg oracle: {pa_en_str} vs logit-fused {en_or_str}")
    lines.append(f"- ZH prob-avg oracle: {pa_zh_str} vs logit-fused {zh_or_str}")
    lines.append(f"- AP1 violation (prob-avg ≥ fused on either): "
                 f"**{'TRUE' if ap1_violation else 'FALSE'}**")
    if ap1_violation:
        lines.append("- **v3 RETIRED under AP1 self-binding clause.** "
                     "Probability-space averaging matches/beats logit-space fusion; "
                     "the log-odds framing is not load-bearing.")
    lines.append("")

    # Clause 3: Ablation A load-bearing
    lines.append("## Clause 3 — Ablation A load-bearing (evidence-only vs fused)")
    lines.append("")
    evn_str = f"{evid_en_or:.4f}" if evid_en_or is not None else "—"
    evz_str = f"{evid_zh_or:.4f}" if evid_zh_or is not None else "—"
    lines.append(f"- EN evidence-only oracle: {evn_str} vs logit-fused {en_or_str}")
    lines.append(f"- ZH evidence-only oracle: {evz_str} vs logit-fused {zh_or_str}")
    lines.append(f"- Ablation A leak (evid-only ≥ fused on either): "
                 f"**{'TRUE' if ablation_a_leak else 'FALSE'}**")
    if ablation_a_leak:
        lines.append("- **v3 collapses to baseline.** Call 2 contributes no net information.")
    lines.append("")

    # Clause 4: Ablation E sign check
    lines.append("## Clause 4 — Ablation E bias-cancellation sign check")
    lines.append("")
    lines.append("Predicted drift signs: Call 1 FN-biased (pos_mean LOW) → "
                 "fused pos_mean > evidence pos_mean (Δ > 0). "
                 "Call 2-negated FP-biased (neg_mean HIGH) → "
                 "fused neg_mean < compliance-negated neg_mean (Δ < 0).")
    lines.append("")
    for ds in ["MHClip_EN", "MHClip_ZH"]:
        s = sign_check.get(ds, {})
        if "status" in s:
            lines.append(f"- {ds}: {s['status']}")
            continue
        verdict = "PASS" if s["passed"] else "FAIL"
        lines.append(
            f"- {ds}: Δpos={s['delta_pos']:+.4f} (expect +, got {s['pos_sign_observed']}), "
            f"Δneg={s['delta_neg']:+.4f} (expect −, got {s['neg_sign_observed']}) → **{verdict}**"
        )
    lines.append(f"- **Clause 4 overall: {'PASS' if sign_pass else 'FAIL'}**")
    lines.append("")

    # Aggregator table
    lines.append("## Aggregator table (each cell shows ACC / macro-F1)")
    lines.append("")
    lines.append(
        "| Aggregator | Dataset | N | Oracle | TF-Otsu | TF-GMM | TR-Otsu | TR-GMM "
        "| pos_mean | neg_mean | Bhatt-overlap |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for agg_name, _ in aggregators:
        for ds in ["MHClip_EN", "MHClip_ZH"]:
            row = per_agg_rows[agg_name][ds]
            if "error" in row:
                lines.append(f"| {agg_name} | {ds} | — | ERR: {row['error']} | | | | | | | |")
                continue
            oracle_str = (
                f"{row['oracle']['acc']:.4f}"
                if row.get("oracle") else "—"
            )
            pos_str = f"{row['pos_mean']:.4f}" if row.get("pos_mean") is not None else "—"
            neg_str = f"{row['neg_mean']:.4f}" if row.get("neg_mean") is not None else "—"
            bho_str = (f"{row['bhattacharyya_overlap']:.4f}"
                       if row.get("bhattacharyya_overlap") is not None else "—")
            lines.append(
                f"| {agg_name} | {ds} | {row['n_test']} | {oracle_str} "
                f"| {fmt(row.get('tf_otsu'))} | {fmt(row.get('tf_gmm'))} "
                f"| {fmt(row.get('tr_otsu'))} | {fmt(row.get('tr_gmm'))} "
                f"| {pos_str} | {neg_str} | {bho_str} |"
            )
    lines.append("")

    # Unified winner / verdict
    lines.append("## Unified (TR/TF × Otsu/GMM) winner — logit_fused score")
    lines.append("")
    if report["unified_winner"]:
        uw = report["unified_winner"]
        lines.append(f"Selected cell: **{uw['cell']}**")
        lines.append(
            f"- EN: ACC={uw['EN']['acc']:.4f} mF1={uw['EN']['mf']:.4f} "
            f"(Δ ACC={uw['EN']['acc']-BASELINE['MHClip_EN']['acc']:+.4f}, "
            f"Δ mF1={uw['EN']['mf']-BASELINE['MHClip_EN']['mf']:+.4f})"
        )
        lines.append(
            f"- ZH: ACC={uw['ZH']['acc']:.4f} mF1={uw['ZH']['mf']:.4f} "
            f"(Δ ACC={uw['ZH']['acc']-BASELINE['MHClip_ZH']['acc']:+.4f}, "
            f"Δ mF1={uw['ZH']['mf']-BASELINE['MHClip_ZH']['mf']:+.4f})"
        )
        lines.append("")
        lines.append("**Gate-2 status: PASS — logit-fused strict-beats baseline on both "
                     "datasets under one unified cell, and all four binding clauses pass.**")
    else:
        reasons = []
        if not oracle_pass:
            reasons.append("Clause 1 (oracle-first) FAIL")
        if ap1_violation:
            reasons.append("Clause 2 (AP1 self-binding) FAIL")
        if ablation_a_leak:
            reasons.append("Clause 3 (Ablation A load-bearing) FAIL")
        if not sign_pass:
            reasons.append("Clause 4 (Ablation E sign) FAIL")
        if not reasons:
            reasons.append("no unified (TR/TF × Otsu/GMM) cell strict-beats baseline on both")
        lines.append("Gate-2 status: **MISS** — " + "; ".join(reasons))
        lines.append("")
        lines.append("Return to Gate 1 with v4.")
    lines.append("")

    report_md = os.path.join(analysis_dir, "prompt_paradigm_report.md")
    with open(report_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote {os.path.join(out_dir, 'report_v3.json')}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
