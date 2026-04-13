"""
prompt_paradigm v2 evaluator — Factored Verdict.

Gate 2 verification for v2 with:
  1. Oracle-first pre-check on product score. If EN or ZH oracle <= baseline,
     v2 is falsified at the rank level — report miss, do NOT proceed to threshold
     search.
  2. 4 (TR/TF × Otsu/GMM) cells on product score.
  3. Ablations A (P_T alone), B (P_S alone), C (P_T + P_S sum), D (max, v1 is external).
     Each ablation runs through the same 4 cells. Ablation guard: if P_T alone or
     P_S alone strict-beats baseline on both datasets under the same unified cell,
     the factorization is not load-bearing and Gate 2 is blocked (anti-pattern 2).

Imports frozen back-half functions from src/quick_eval_all.py by reference only.

Reads:
  results/prompt_paradigm/{MHClip_EN,MHClip_ZH}/{test,train}_factored.jsonl

Writes:
  results/prompt_paradigm/report_v2.json
  results/analysis/prompt_paradigm_report.md  (overwrites v1 report — permitted)
"""

import json
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

BASELINE = {
    "MHClip_EN": {"acc": 0.7640, "mf": 0.653,
                  "oracle": 0.7764,
                  "source": "2B binary_nodef TF-Otsu"},
    "MHClip_ZH": {"acc": 0.8121, "mf": 0.787,
                  "oracle": 0.8121,
                  "source": "2B binary_nodef TF-GMM"},
}


def load_factored(path):
    """Load factored JSONL. Returns dict[vid -> {p_target, p_stance, score}]."""
    d = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vid = r.get("video_id")
            if vid is None:
                continue
            p_t = r.get("p_target")
            p_s = r.get("p_stance")
            score = r.get("score")
            if p_t is None or p_s is None or score is None:
                continue
            d[vid] = {
                "p_target": float(p_t),
                "p_stance": float(p_s),
                "score": float(score),
            }
    return d


def build_scored_arrays(d, ann, field):
    """Build (x, y) arrays using d[vid][field] as the score."""
    simple = {vid: rec[field] for vid, rec in d.items()}
    return build_arrays(simple, ann)


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


def evaluate_aggregator(ds, d_test, d_train, field_or_fn, label):
    """Evaluate a single aggregation strategy on a single dataset.

    field_or_fn: either a string field name ('score','p_target','p_stance')
                 or a callable(rec) -> float.
    Returns: dict with oracle + 4 cells.
    """
    ann = load_annotations(ds)
    if callable(field_or_fn):
        simple_test = {vid: field_or_fn(rec) for vid, rec in d_test.items()}
    else:
        simple_test = {vid: rec[field_or_fn] for vid, rec in d_test.items()}
    test_x, test_y = build_arrays(simple_test, ann)
    if len(test_x) == 0:
        return {"error": "no test overlap"}

    train_x = None
    if d_train:
        if callable(field_or_fn):
            simple_train = {vid: field_or_fn(rec) for vid, rec in d_train.items()}
        else:
            simple_train = {vid: rec[field_or_fn] for vid, rec in d_train.items()}
        train_x = np.array(list(simple_train.values()))

    oracle, oracle_t = oracle_acc(test_x, test_y)
    res = {
        "label": label,
        "n_test": int(len(test_x)),
        "n_test_pos": int(test_y.sum()),
        "oracle": {"acc": oracle, "t": oracle_t},
        **four_cells(test_x, test_y, train_x),
    }
    # Per-class stats for signature checks
    pos = test_x[test_y == 1]
    neg = test_x[test_y == 0]
    res["pos_mean"] = float(pos.mean()) if len(pos) else None
    res["neg_mean"] = float(neg.mean()) if len(neg) else None
    res["pos_std"] = float(pos.std()) if len(pos) else None
    res["neg_std"] = float(neg.std()) if len(neg) else None
    return res


def find_unified(en_row, zh_row):
    """Return (cell, en_m, zh_m) for the single best unified cell that
    STRICT-beats baseline on both datasets, or None."""
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


def main():
    out_dir = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm")
    analysis_dir = os.path.join(PROJECT_ROOT, "results", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    # Load raw factored records for both datasets
    data = {}
    for ds in ["MHClip_EN", "MHClip_ZH"]:
        base = os.path.join(out_dir, ds)
        test_path = os.path.join(base, "test_factored.jsonl")
        train_path = os.path.join(base, "train_factored.jsonl")
        if not os.path.isfile(test_path):
            data[ds] = {"error": f"missing {test_path}"}
            continue
        d_test = load_factored(test_path)
        d_train = load_factored(train_path) if os.path.isfile(train_path) else {}
        data[ds] = {"test": d_test, "train": d_train}

    # Aggregators to evaluate
    aggregators = [
        ("product", "score"),            # v2 main claim
        ("P_T_only", "p_target"),        # Ablation A
        ("P_S_only", "p_stance"),        # Ablation B
        ("sum", lambda r: 0.5 * (r["p_target"] + r["p_stance"])),   # Ablation C (averaged)
        ("max", lambda r: max(r["p_target"], r["p_stance"])),       # additional diagnostic
        ("min", lambda r: min(r["p_target"], r["p_stance"])),       # additional diagnostic
    ]

    report = {"baseline": BASELINE, "aggregators": {}}
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

    report["aggregators"] = per_agg_rows

    # ----- Oracle-first pre-check on product (the Gate 2 headline claim) -----
    product_en = per_agg_rows["product"].get("MHClip_EN", {})
    product_zh = per_agg_rows["product"].get("MHClip_ZH", {})
    en_oracle = product_en.get("oracle", {}).get("acc")
    zh_oracle = product_zh.get("oracle", {}).get("acc")
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

    # ----- Unified-cell search (only meaningful if oracle passed) -----
    unified = None
    if oracle_pass:
        unified = find_unified(product_en, product_zh)
    if unified:
        cell, en_m, zh_m = unified
        report["unified_winner"] = {
            "cell": cell,
            "EN": {"acc": en_m["acc"], "mf": en_m["mf"], "t": en_m.get("t")},
            "ZH": {"acc": zh_m["acc"], "mf": zh_m["mf"], "t": zh_m.get("t")},
        }
    else:
        report["unified_winner"] = None

    # ----- Ablation load-bearing guard -----
    # If P_T alone or P_S alone already strict-beats under a unified cell,
    # the factorization is not load-bearing.
    ablation_leak = {}
    for aname in ["P_T_only", "P_S_only"]:
        en_row = per_agg_rows[aname].get("MHClip_EN", {})
        zh_row = per_agg_rows[aname].get("MHClip_ZH", {})
        if "error" in en_row or "error" in zh_row:
            ablation_leak[aname] = {"status": "err"}
            continue
        u = find_unified(en_row, zh_row)
        ablation_leak[aname] = {
            "beats_unified": u is not None,
            "cell": u[0] if u else None,
            "EN_acc": u[1]["acc"] if u else None,
            "ZH_acc": u[2]["acc"] if u else None,
        }
    report["ablation_leak"] = ablation_leak

    # ----- Write JSON -----
    with open(os.path.join(out_dir, "report_v2.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ----- Write Markdown summary -----
    lines = []
    lines.append("# prompt_paradigm v2 report — Factored Verdict (P_T × P_S)\n")
    lines.append("## Baseline (must strict-beat on BOTH under one unified cell)\n")
    lines.append(
        f"- MHClip_EN: ACC={BASELINE['MHClip_EN']['acc']:.4f} "
        f"macro-F1={BASELINE['MHClip_EN']['mf']:.3f} "
        f"(oracle ceiling {BASELINE['MHClip_EN']['oracle']:.4f}) "
        f"[{BASELINE['MHClip_EN']['source']}]"
    )
    lines.append(
        f"- MHClip_ZH: ACC={BASELINE['MHClip_ZH']['acc']:.4f} "
        f"macro-F1={BASELINE['MHClip_ZH']['mf']:.3f} "
        f"(oracle ceiling {BASELINE['MHClip_ZH']['oracle']:.4f}) "
        f"[{BASELINE['MHClip_ZH']['source']}]\n"
    )

    lines.append("## Oracle-first pre-check (gate-2 prerequisite)\n")
    en_or_str = f"{en_oracle:.4f}" if en_oracle is not None else "—"
    zh_or_str = f"{zh_oracle:.4f}" if zh_oracle is not None else "—"
    en_or_verdict = "PASS" if en_oracle and en_oracle > BASELINE["MHClip_EN"]["oracle"] else "FAIL"
    zh_or_verdict = "PASS" if zh_oracle and zh_oracle > BASELINE["MHClip_ZH"]["oracle"] else "FAIL"
    lines.append(
        f"- EN oracle: {en_or_str} vs baseline {BASELINE['MHClip_EN']['oracle']:.4f} "
        f"→ {en_or_verdict}"
    )
    lines.append(
        f"- ZH oracle: {zh_or_str} vs baseline {BASELINE['MHClip_ZH']['oracle']:.4f} "
        f"→ {zh_or_verdict}"
    )
    lines.append(f"- **Oracle-first overall: {'PASS' if oracle_pass else 'FAIL'}**\n")

    lines.append("## Aggregator table (each cell shows ACC / macro-F1)\n")
    lines.append(
        "| Aggregator | Dataset | N | Oracle | TF-Otsu | TF-GMM | TR-Otsu | TR-GMM "
        "| pos_mean | neg_mean |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    for agg_name, _ in aggregators:
        for ds in ["MHClip_EN", "MHClip_ZH"]:
            row = per_agg_rows[agg_name][ds]
            if "error" in row:
                lines.append(
                    f"| {agg_name} | {ds} | — | ERR: {row['error']} | | | | | | |"
                )
                continue
            oracle_str = (
                f"{row['oracle']['acc']:.4f}"
                if row.get("oracle") else "—"
            )
            pos_str = f"{row['pos_mean']:.4f}" if row.get("pos_mean") is not None else "—"
            neg_str = f"{row['neg_mean']:.4f}" if row.get("neg_mean") is not None else "—"
            lines.append(
                f"| {agg_name} | {ds} | {row['n_test']} | {oracle_str} "
                f"| {fmt(row.get('tf_otsu'))} | {fmt(row.get('tf_gmm'))} "
                f"| {fmt(row.get('tr_otsu'))} | {fmt(row.get('tr_gmm'))} "
                f"| {pos_str} | {neg_str} |"
            )
    lines.append("")

    lines.append("## Unified (TR/TF × Otsu/GMM) winner — PRODUCT score\n")
    if report["unified_winner"]:
        uw = report["unified_winner"]
        lines.append(f"Selected cell: **{uw['cell']}**")
        lines.append(
            f"- EN: ACC={uw['EN']['acc']:.4f} mF1={uw['EN']['mf']:.3f} "
            f"(Δ ACC={uw['EN']['acc']-BASELINE['MHClip_EN']['acc']:+.4f})"
        )
        lines.append(
            f"- ZH: ACC={uw['ZH']['acc']:.4f} mF1={uw['ZH']['mf']:.3f} "
            f"(Δ ACC={uw['ZH']['acc']-BASELINE['MHClip_ZH']['acc']:+.4f})"
        )
        lines.append("\n**Gate-2 status: beats baseline on both datasets under one unified cell.**")
    else:
        if not oracle_pass:
            lines.append(
                "Oracle pre-check failed; no unified threshold cell can rescue the score.\n"
                "**Gate-2 status: MISS (rank-level failure). Return to Gate 1 with v3.**"
            )
        else:
            lines.append(
                "Oracle pre-check passed but no unified (TR/TF × Otsu/GMM) cell strict-beats "
                "baseline on both datasets with mF1 non-regression.\n"
                "**Gate-2 status: MISS (threshold-level failure). Return to Gate 1 with v3.**"
            )
    lines.append("")

    lines.append("## Ablation load-bearing guard\n")
    for aname, info in ablation_leak.items():
        if info.get("status") == "err":
            lines.append(f"- {aname}: ERR")
            continue
        if info.get("beats_unified"):
            lines.append(
                f"- {aname}: **LEAK** — this single factor alone strict-beats baseline under "
                f"cell {info['cell']} (EN {info['EN_acc']:.4f}, ZH {info['ZH_acc']:.4f}). "
                f"The factorization is NOT load-bearing. Gate 2 is blocked by anti-pattern 2."
            )
        else:
            lines.append(
                f"- {aname}: does not strict-beat baseline on its own. Factorization retains "
                f"the 'both factors needed' property."
            )
    lines.append("")

    lines.append(
        "## Ablation D reference (video vs text-only Call 2)\n"
        "Ablation D is established by v1 (Observe-then-Judge): v1's text-only Judge "
        "produced EN oracle 0.7391 / ZH oracle 0.7785 — strictly below baseline, "
        "confirming that removing video grounding on the verdict call destroys the signal. "
        "v2 keeps both calls video-grounded and avoids this failure mode by construction.\n"
    )

    report_md = os.path.join(analysis_dir, "prompt_paradigm_report.md")
    with open(report_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote {os.path.join(out_dir, 'report_v2.json')}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
