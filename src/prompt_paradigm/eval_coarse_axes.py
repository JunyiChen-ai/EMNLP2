"""
prompt_paradigm v6 — Coarse Axes Prompt (CAP) evaluator.

Reads:
  results/prompt_paradigm/{dataset}/{test,train}_coarse_axes_axes.jsonl
  results/prompt_paradigm/{dataset}/{test,train}_coarse_axes_control.jsonl
  results/holistic_2b/{dataset}/{test,train}_binary.jsonl   (baseline)
  results/prompt_paradigm/{dataset}/{test,train}_polarity.jsonl  (v3 p_evidence)

Reports all 8 Gate-2 clauses from docs/proposals/prompt_paradigm_v6.md §5:
  1. Oracle-first: v6 axes oracle > 0.7764 EN AND > 0.8121 ZH (strict).
  2. mF1 non-regression on one unified TR/TF × Otsu/GMM cell.
  3. Ablation A: v6 axes oracle > baseline oracle on BOTH (strict).
  4. Ablation B: v6 axes oracle > v3 p_evidence oracle on BOTH (strict).
  5. Ablation C (length-matched control): control oracle must NOT strict-beat
     baseline on EITHER dataset. If it does, v6 is rejected for AP2.
  6. P3 collapse-ratio diagnostic: EN absolute gain ≥ ZH absolute gain.
     Directional only, does not auto-reject.
  7. Format compliance ≥ 0.95 on both (null rate ≤ 5%).
  8. n_test reconciliation: EN 161, ZH 149.

Writes:
  results/prompt_paradigm/report_v6.json
  appends a v6 section to results/analysis/prompt_paradigm_report.md

Usage (always via sbatch):
  sbatch --cpus-per-task=2 --mem=4G --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh \\
    && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 \\
    && python src/prompt_paradigm/eval_coarse_axes.py"
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

# v3 p_evidence oracle is the tighter bar on ZH (0.8188 > 0.8121 baseline).
# Clause 4 enforces strict-beat on this value.
V3_EVIDENCE_ORACLE = {
    "MHClip_EN": 0.7702,
    "MHClip_ZH": 0.8188,
}

N_EXPECTED = {"MHClip_EN": 161, "MHClip_ZH": 149}


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


def load_condition(dataset, condition):
    """Load v6 axes/control scores for a dataset.

    Returns (test_pairs, train_scores, n_rows_seen, n_null)
    test_pairs: list of (score, label)
    train_scores: list of floats
    n_rows_seen: total rows in the test file (to compute null rate)
    n_null: rows with score=None in the test file
    """
    ann_to_label = build_label_map(dataset)
    test_path = os.path.join(
        PROJECT_ROOT, "results", "prompt_paradigm", dataset,
        f"test_coarse_axes_{condition}.jsonl")
    train_path = os.path.join(
        PROJECT_ROOT, "results", "prompt_paradigm", dataset,
        f"train_coarse_axes_{condition}.jsonl")

    split_csv = os.path.join(DATASET_ROOTS[dataset], "splits", "test_clean.csv")
    with open(split_csv) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    test_rows = {r["video_id"]: r for r in load_jsonl(test_path)}
    train_rows = load_jsonl(train_path)

    n_rows = len(test_rows)
    n_null = sum(1 for r in test_rows.values() if r.get("score") is None)

    test_pairs = []
    for vid in test_ids:
        r = test_rows.get(vid)
        if r is None or r.get("score") is None or vid not in ann_to_label:
            continue
        test_pairs.append((r["score"], ann_to_label[vid]))

    train_scores = [r["score"] for r in train_rows if r.get("score") is not None]
    return test_pairs, train_scores, n_rows, n_null


def load_baseline_scores(dataset):
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


def main():
    print(f"[v6-eval] starting in {PROJECT_ROOT}")
    out = {
        "baseline": BASELINE,
        "v3_evidence_oracle_bar": V3_EVIDENCE_ORACLE,
        "conditions": {},
        "format_compliance": {},
    }

    for dataset in ("MHClip_EN", "MHClip_ZH"):
        print(f"\n[v6-eval] === {dataset} ===")
        out["conditions"][dataset] = {}

        # --- Condition: axes (v6 treatment) ---
        axes_pairs, axes_train, axes_rows, axes_null = load_condition(dataset, "axes")
        print(f"[v6-eval]   axes test N={len(axes_pairs)} (rows={axes_rows}, null={axes_null})"
              f" train N={len(axes_train)}")
        if axes_pairs:
            scores = [p for p, _ in axes_pairs]
            labels = [y for _, y in axes_pairs]
            block = full_metrics_block(scores, labels, axes_train)
            out["conditions"][dataset]["axes"] = block
            print(f"[v6-eval]   axes oracle={block['oracle']['acc']:.4f} "
                  f"tf_otsu={block['tf_otsu']['acc']:.4f}/{block['tf_otsu']['mf']:.4f} "
                  f"auc={block['auc']:.4f}")

        # --- Condition: control (length-matched non-taxonomic) ---
        ctl_pairs, ctl_train, ctl_rows, ctl_null = load_condition(dataset, "control")
        print(f"[v6-eval]   control test N={len(ctl_pairs)} (rows={ctl_rows}, null={ctl_null})"
              f" train N={len(ctl_train)}")
        if ctl_pairs:
            scores = [p for p, _ in ctl_pairs]
            labels = [y for _, y in ctl_pairs]
            block = full_metrics_block(scores, labels, ctl_train)
            out["conditions"][dataset]["control"] = block
            print(f"[v6-eval]   ctrl oracle={block['oracle']['acc']:.4f} "
                  f"tf_otsu={block['tf_otsu']['acc']:.4f}/{block['tf_otsu']['mf']:.4f} "
                  f"auc={block['auc']:.4f}")

        # --- Baseline (for display and clause 3) ---
        base_pairs, base_train = load_baseline_scores(dataset)
        if base_pairs:
            scores = [p for p, _ in base_pairs]
            labels = [y for _, y in base_pairs]
            block = full_metrics_block(scores, labels, base_train)
            out["conditions"][dataset]["baseline"] = block
            print(f"[v6-eval]   base oracle={block['oracle']['acc']:.4f}")

        # --- v3 p_evidence (for clause 4 display) ---
        v3_pairs, v3_train = load_v3_evidence_scores(dataset)
        if v3_pairs:
            scores = [p for p, _ in v3_pairs]
            labels = [y for _, y in v3_pairs]
            block = full_metrics_block(scores, labels, v3_train)
            out["conditions"][dataset]["v3_evidence"] = block
            print(f"[v6-eval]   v3 oracle={block['oracle']['acc']:.4f}")

        # --- Format compliance (clause 7) ---
        axes_clean = (axes_rows - axes_null) / axes_rows if axes_rows else 0.0
        ctl_clean = (ctl_rows - ctl_null) / ctl_rows if ctl_rows else 0.0
        out["format_compliance"][dataset] = {
            "axes": {"n_rows": axes_rows, "n_null": axes_null, "clean_rate": axes_clean},
            "control": {"n_rows": ctl_rows, "n_null": ctl_null, "clean_rate": ctl_clean},
        }

    # -------- Binding clauses --------
    en_axes = out["conditions"].get("MHClip_EN", {}).get("axes")
    zh_axes = out["conditions"].get("MHClip_ZH", {}).get("axes")
    en_ctl = out["conditions"].get("MHClip_EN", {}).get("control")
    zh_ctl = out["conditions"].get("MHClip_ZH", {}).get("control")
    en_base = out["conditions"].get("MHClip_EN", {}).get("baseline")
    zh_base = out["conditions"].get("MHClip_ZH", {}).get("baseline")

    clauses = {}

    # Clause 1: oracle-first vs baseline oracle
    en_axes_o = en_axes["oracle"]["acc"] if en_axes else None
    zh_axes_o = zh_axes["oracle"]["acc"] if zh_axes else None
    clauses["1_oracle_first"] = {
        "en_axes_oracle": en_axes_o,
        "en_baseline_oracle": BASELINE["MHClip_EN"]["oracle"],
        "zh_axes_oracle": zh_axes_o,
        "zh_baseline_oracle": BASELINE["MHClip_ZH"]["oracle"],
        "passed": (en_axes_o is not None and zh_axes_o is not None
                   and en_axes_o > BASELINE["MHClip_EN"]["oracle"]
                   and zh_axes_o > BASELINE["MHClip_ZH"]["oracle"]),
    }

    # Clause 2: mF1 non-regression unified cell
    unified_winner = None
    for tname in ("tf_otsu", "tf_gmm", "tr_otsu", "tr_gmm"):
        if en_axes is None or zh_axes is None:
            break
        en_c = en_axes[tname]
        zh_c = zh_axes[tname]
        en_acc_ok = en_c["acc"] >= BASELINE["MHClip_EN"]["acc"]
        zh_acc_ok = zh_c["acc"] >= BASELINE["MHClip_ZH"]["acc"]
        en_mf_ok = en_c["mf"] >= BASELINE["MHClip_EN"]["mf"]
        zh_mf_ok = zh_c["mf"] >= BASELINE["MHClip_ZH"]["mf"]
        # strict beat on at least one metric per dataset
        en_strict = en_c["acc"] > BASELINE["MHClip_EN"]["acc"] or en_c["mf"] > BASELINE["MHClip_EN"]["mf"]
        zh_strict = zh_c["acc"] > BASELINE["MHClip_ZH"]["acc"] or zh_c["mf"] > BASELINE["MHClip_ZH"]["mf"]
        if en_acc_ok and zh_acc_ok and en_mf_ok and zh_mf_ok and en_strict and zh_strict:
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

    # Clause 3: Ablation A baseline load-bearing (strict)
    if en_axes and zh_axes and en_base and zh_base:
        en_base_o = en_base["oracle"]["acc"]
        zh_base_o = zh_base["oracle"]["acc"]
        clauses["3_ablation_A_baseline_load_bearing"] = {
            "en_axes_oracle": en_axes_o,
            "en_baseline_oracle": en_base_o,
            "zh_axes_oracle": zh_axes_o,
            "zh_baseline_oracle": zh_base_o,
            "passed": (en_axes_o > en_base_o and zh_axes_o > zh_base_o),
        }
    else:
        clauses["3_ablation_A_baseline_load_bearing"] = {"passed": False, "note": "missing data"}

    # Clause 4: Ablation B v3 p_evidence prior-art strict-beat
    clauses["4_ablation_B_v3_prior_art"] = {
        "en_axes_oracle": en_axes_o,
        "en_v3_oracle_bar": V3_EVIDENCE_ORACLE["MHClip_EN"],
        "zh_axes_oracle": zh_axes_o,
        "zh_v3_oracle_bar": V3_EVIDENCE_ORACLE["MHClip_ZH"],
        "passed": (en_axes_o is not None and zh_axes_o is not None
                   and en_axes_o > V3_EVIDENCE_ORACLE["MHClip_EN"]
                   and zh_axes_o > V3_EVIDENCE_ORACLE["MHClip_ZH"]),
    }

    # Clause 5: Ablation C length-matched control must NOT strict-beat baseline
    en_ctl_o = en_ctl["oracle"]["acc"] if en_ctl else None
    zh_ctl_o = zh_ctl["oracle"]["acc"] if zh_ctl else None
    if en_ctl and zh_ctl and en_base and zh_base:
        en_base_o = en_base["oracle"]["acc"]
        zh_base_o = zh_base["oracle"]["acc"]
        ctl_beats_en = en_ctl_o > en_base_o
        ctl_beats_zh = zh_ctl_o > zh_base_o
        clauses["5_ablation_C_length_control"] = {
            "en_control_oracle": en_ctl_o,
            "en_baseline_oracle": en_base_o,
            "zh_control_oracle": zh_ctl_o,
            "zh_baseline_oracle": zh_base_o,
            "ctl_beats_en": ctl_beats_en,
            "ctl_beats_zh": ctl_beats_zh,
            # clause passes iff control fails to strict-beat baseline on BOTH sides
            "passed": not (ctl_beats_en or ctl_beats_zh),
        }
    else:
        clauses["5_ablation_C_length_control"] = {"passed": False, "note": "missing data"}

    # Clause 6: P3 collapse-ratio directional diagnostic
    if en_axes and zh_axes and en_base and zh_base:
        en_gain = en_axes_o - en_base["oracle"]["acc"]
        zh_gain = zh_axes_o - zh_base["oracle"]["acc"]
        clauses["6_p3_collapse_ratio"] = {
            "en_gain": en_gain,
            "zh_gain": zh_gain,
            "en_ge_zh": en_gain >= zh_gain,
            "note": "directional diagnostic only, not a hard reject",
        }
    else:
        clauses["6_p3_collapse_ratio"] = {"note": "missing data"}

    # Clause 7: format compliance ≥ 0.95
    en_fc = out["format_compliance"].get("MHClip_EN", {}).get("axes", {}).get("clean_rate", 0.0)
    zh_fc = out["format_compliance"].get("MHClip_ZH", {}).get("axes", {}).get("clean_rate", 0.0)
    en_ctl_fc = out["format_compliance"].get("MHClip_EN", {}).get("control", {}).get("clean_rate", 0.0)
    zh_ctl_fc = out["format_compliance"].get("MHClip_ZH", {}).get("control", {}).get("clean_rate", 0.0)
    clauses["7_format_compliance"] = {
        "en_axes_clean_rate": en_fc,
        "zh_axes_clean_rate": zh_fc,
        "en_control_clean_rate": en_ctl_fc,
        "zh_control_clean_rate": zh_ctl_fc,
        "passed": (en_fc >= 0.95 and zh_fc >= 0.95
                   and en_ctl_fc >= 0.95 and zh_ctl_fc >= 0.95),
    }

    # Clause 8: n_test reconciliation
    en_n = en_axes["n_test"] if en_axes else None
    zh_n = zh_axes["n_test"] if zh_axes else None
    clauses["8_n_test_reconciliation"] = {
        "en_n_test": en_n, "en_expected": N_EXPECTED["MHClip_EN"],
        "zh_n_test": zh_n, "zh_expected": N_EXPECTED["MHClip_ZH"],
        "passed": (en_n == N_EXPECTED["MHClip_EN"] and zh_n == N_EXPECTED["MHClip_ZH"]),
    }

    out["clauses"] = clauses
    out["unified_winner"] = unified_winner
    # strict AND over clauses 1, 2, 3, 4, 5, 7, 8 (clause 6 is directional)
    out["all_clauses_pass"] = (
        clauses["1_oracle_first"].get("passed") is True
        and clauses["2_mf1_non_regression_unified"].get("passed") is True
        and clauses["3_ablation_A_baseline_load_bearing"].get("passed") is True
        and clauses["4_ablation_B_v3_prior_art"].get("passed") is True
        and clauses["5_ablation_C_length_control"].get("passed") is True
        and clauses["7_format_compliance"].get("passed") is True
        and clauses["8_n_test_reconciliation"].get("passed") is True
    )

    report_path = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", "report_v6.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[v6-eval] wrote {report_path}")
    print(f"[v6-eval] all_clauses_pass={out['all_clauses_pass']}")

    md_path = os.path.join(PROJECT_ROOT, "results", "analysis", "prompt_paradigm_report.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "a") as f:
        f.write("\n\n## v6 — Coarse Axes Prompt (CAP) (2026-04-13)\n\n")
        f.write(f"- Clause 1 (Oracle-first): EN axes {en_axes_o} vs {BASELINE['MHClip_EN']['oracle']} | "
                f"ZH axes {zh_axes_o} vs {BASELINE['MHClip_ZH']['oracle']} | "
                f"passed={clauses['1_oracle_first']['passed']}\n")
        f.write(f"- Clause 2 (Unified-winner): "
                f"{unified_winner['threshold_family'] if unified_winner else 'NONE'}\n")
        f.write(f"- Clause 3 (Ablation A baseline strict-beat): "
                f"passed={clauses['3_ablation_A_baseline_load_bearing'].get('passed')}\n")
        f.write(f"- Clause 4 (Ablation B v3 p_evidence strict-beat, bars EN {V3_EVIDENCE_ORACLE['MHClip_EN']} / "
                f"ZH {V3_EVIDENCE_ORACLE['MHClip_ZH']}): "
                f"passed={clauses['4_ablation_B_v3_prior_art']['passed']}\n")
        f.write(f"- Clause 5 (Ablation C length-matched control): "
                f"EN ctl {en_ctl_o} ZH ctl {zh_ctl_o} "
                f"passed={clauses['5_ablation_C_length_control'].get('passed')}\n")
        en_gain = clauses['6_p3_collapse_ratio'].get('en_gain')
        zh_gain = clauses['6_p3_collapse_ratio'].get('zh_gain')
        f.write(f"- Clause 6 (P3 collapse-ratio directional): "
                f"EN gain {en_gain} ZH gain {zh_gain} "
                f"en_ge_zh={clauses['6_p3_collapse_ratio'].get('en_ge_zh')}\n")
        f.write(f"- Clause 7 (format compliance ≥ 0.95): "
                f"EN axes {en_fc:.3f} ZH axes {zh_fc:.3f} "
                f"EN ctl {en_ctl_fc:.3f} ZH ctl {zh_ctl_fc:.3f} "
                f"passed={clauses['7_format_compliance']['passed']}\n")
        f.write(f"- Clause 8 (N_test): EN={en_n} ZH={zh_n} "
                f"passed={clauses['8_n_test_reconciliation']['passed']}\n")
        f.write(f"- **All clauses pass: {out['all_clauses_pass']}**\n")


if __name__ == "__main__":
    main()
