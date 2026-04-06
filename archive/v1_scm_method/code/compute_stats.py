#!/usr/bin/env python3
"""Compute mean±std, max, and bootstrap significance for SCM-MoE results."""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
MAIN_DIR = BASE_DIR / "seed_search_scm_qmoe_qels_mr2"
ABL_DIR  = BASE_DIR / "ablation_results"

DATASETS = ["HateMM", "MHC_En", "MHC_Ch", "ImpliHateVid"]
DATASET_LABELS = {
    "HateMM":       "HateMM",
    "MHC_En":       "MHC\\_En",
    "MHC_Ch":       "MHC\\_Ch",
    "ImpliHateVid": "ImpliHateVid",
}

BEST_CONFIGS = {
    "HateMM":       {"alpha": 0.3, "beta": 0.0},
    "MHC_En":       {"alpha": 0.1, "beta": 0.1},
    "MHC_Ch":       {"alpha": 0.1, "beta": 0.0},
    "ImpliHateVid": {"alpha": 0.3, "beta": 0.0},
}

ABLATION_VARIANTS = [
    "base_only",
    "generic_prompt",
    "flat_concat",
    "unconstrained_moe",
    "single_expert",
    "no_qels",
    "no_mr2",
    "focal",
]

VARIANT_LABELS = {
    "full":              "\\textbf{Full model}",
    "base_only":         "w/o SCM (base only)",
    "generic_prompt":    "w/o theory prompts",
    "flat_concat":       "w/o MoE (flat concat)",
    "unconstrained_moe": "w/o load balancing",
    "single_expert":     "Single expert",
    "no_qels":           "w/o QELS",
    "no_mr2":            "w/o MR\\textsuperscript{2}",
    "focal":             "Focal loss (replace MR\\textsuperscript{2})",
}

N_BOOTSTRAP = 10_000
ALPHA_SIG   = 0.05
RNG = np.random.default_rng(2026)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_results(json_path: Path) -> list[dict]:
    """Load all_results list from a results JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    # Handle both formats: list or dict with "all_results" key
    if isinstance(data, list):
        return data
    return data.get("all_results", [])


def extract_metrics(results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract ACC and F1 arrays from result list."""
    accs = np.array([r["metrics"]["acc"] for r in results])
    f1s  = np.array([r["metrics"]["f1"]  for r in results])
    return accs, f1s


def unpaired_bootstrap_test(
    x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP
) -> tuple[float, float, float]:
    """
    Unpaired bootstrap significance test.
    Tests whether mean(x) - mean(y) is significantly different from 0.
    Returns (observed_diff, p_value, ci_lower, ci_upper) for 95% CI.
    """
    obs_diff = x.mean() - y.mean()
    diffs = np.empty(n_boot)
    nx, ny = len(x), len(y)
    for i in range(n_boot):
        bx = x[RNG.integers(0, nx, size=nx)]
        by = y[RNG.integers(0, ny, size=ny)]
        diffs[i] = bx.mean() - by.mean()
    ci_lo = np.percentile(diffs, 100 * ALPHA_SIG / 2)
    ci_hi = np.percentile(diffs, 100 * (1 - ALPHA_SIG / 2))
    # p-value: proportion of bootstrap samples where diff <= 0
    # (one-sided: full model is better, i.e. diff > 0)
    p_value = np.mean(diffs <= 0)
    return obs_diff, p_value, ci_lo, ci_hi


def sig_marker(p: float) -> str:
    """Return significance marker for LaTeX."""
    if p < 0.001:
        return "$^{***}$"
    elif p < 0.01:
        return "$^{**}$"
    elif p < 0.05:
        return "$^{*}$"
    return ""


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Collect all data: {dataset: {variant: (accs, f1s)}}
    all_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for ds in DATASETS:
        all_data[ds] = {}
        cfg = BEST_CONFIGS[ds]
        alpha_str = str(cfg["alpha"])
        beta_str  = str(cfg["beta"])

        # Load full model results
        main_path = MAIN_DIR / f"{ds}_a{alpha_str}_b{beta_str}_lb0.01_off0" / "all_results.json"
        if main_path.exists():
            results = load_results(main_path)
            accs, f1s = extract_metrics(results)
            all_data[ds]["full"] = (accs, f1s)
            print(f"[OK] {ds} full: {len(accs)} seeds, "
                  f"ACC={accs.mean():.4f}±{accs.std():.4f}, "
                  f"F1={f1s.mean():.4f}±{f1s.std():.4f}")
        else:
            print(f"[WARN] Missing: {main_path}")

        # Load ablation variants
        for var in ABLATION_VARIANTS:
            abl_path = ABL_DIR / f"{ds}_{var}_off0" / "all_results.json"
            if abl_path.exists():
                results = load_results(abl_path)
                accs, f1s = extract_metrics(results)
                all_data[ds][var] = (accs, f1s)
                print(f"[OK] {ds} {var}: {len(accs)} seeds, "
                      f"ACC={accs.mean():.4f}±{accs.std():.4f}, "
                      f"F1={f1s.mean():.4f}±{f1s.std():.4f}")
            else:
                print(f"[WARN] Missing: {abl_path}")

    # ── Significance tests ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Bootstrap significance tests (full model vs ablation)")
    print("=" * 80)

    sig_results: dict[str, dict[str, dict[str, tuple]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for ds in DATASETS:
        if "full" not in all_data[ds]:
            continue
        full_accs, full_f1s = all_data[ds]["full"]
        for var in ABLATION_VARIANTS:
            if var not in all_data[ds]:
                continue
            abl_accs, abl_f1s = all_data[ds][var]

            diff_acc, p_acc, ci_lo_acc, ci_hi_acc = unpaired_bootstrap_test(full_accs, abl_accs)
            diff_f1,  p_f1,  ci_lo_f1,  ci_hi_f1  = unpaired_bootstrap_test(full_f1s,  abl_f1s)

            sig_results[ds][var]["acc"] = (diff_acc, p_acc, ci_lo_acc, ci_hi_acc)
            sig_results[ds][var]["f1"]  = (diff_f1,  p_f1,  ci_lo_f1,  ci_hi_f1)

            print(f"  {ds} | full vs {var:20s} | "
                  f"ΔACC={diff_acc:+.4f} p={p_acc:.4f}{sig_marker(p_acc):6s} | "
                  f"ΔF1={diff_f1:+.4f} p={p_f1:.4f}{sig_marker(p_f1):6s}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    print("\n\n% ──────────────── LaTeX Table ────────────────")
    print("% Paste into your paper between \\begin{table} ... \\end{table}")

    n_cols = 1 + 2 * len(DATASETS)  # variant + (ACC, F1) per dataset
    col_spec = "l" + "cc" * len(DATASETS)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study results (mean $\pm$ std over seeds). "
                 r"$^{*}$/$^{**}$/$^{***}$: $p<0.05$/$0.01$/$0.001$ "
                 r"(unpaired bootstrap, 10K resamples).}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: dataset names spanning 2 cols each
    header1 = "Variant"
    for ds in DATASETS:
        header1 += f" & \\multicolumn{{2}}{{c}}{{{DATASET_LABELS[ds]}}}"
    header1 += r" \\"
    lines.append(header1)

    # Header row 2: ACC / F1 under each dataset
    header2 = ""
    for i, ds in enumerate(DATASETS):
        cstart = 2 + 2 * i
        cend   = cstart + 1
        header2 += f"\\cmidrule(lr){{{cstart}-{cend}}} "
    lines.append(header2)

    header3 = ""
    for ds in DATASETS:
        header3 += " & ACC & F1"
    header3 += r" \\"
    lines.append(header3)
    lines.append(r"\midrule")

    # Rows
    variant_order = ["full"] + ABLATION_VARIANTS
    for vi, var in enumerate(variant_order):
        label = VARIANT_LABELS.get(var, var)
        row = label

        for ds in DATASETS:
            if var in all_data[ds]:
                accs, f1s = all_data[ds][var]
                acc_mean, acc_std = accs.mean() * 100, accs.std() * 100
                f1_mean,  f1_std  = f1s.mean() * 100,  f1s.std() * 100

                if var == "full":
                    # Bold the full model row
                    acc_str = f"\\textbf{{{acc_mean:.1f}}}$\\pm${acc_std:.1f}"
                    f1_str  = f"\\textbf{{{f1_mean:.1f}}}$\\pm${f1_std:.1f}"
                else:
                    # Add significance markers
                    acc_sig = ""
                    f1_sig  = ""
                    if ds in sig_results and var in sig_results[ds]:
                        _, p_acc, _, _ = sig_results[ds][var]["acc"]
                        _, p_f1,  _, _ = sig_results[ds][var]["f1"]
                        acc_sig = sig_marker(p_acc)
                        f1_sig  = sig_marker(p_f1)

                    acc_str = f"{acc_mean:.1f}$\\pm${acc_std:.1f}{acc_sig}"
                    f1_str  = f"{f1_mean:.1f}$\\pm${f1_std:.1f}{f1_sig}"

                row += f" & {acc_str} & {f1_str}"
            else:
                row += " & -- & --"

        row += r" \\"
        if vi == 0:
            row += r" \midrule"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    latex_output = "\n".join(lines)
    print(latex_output)

    # Also print max results summary
    print("\n\n% ──────────────── Max Results Summary ────────────────")
    for ds in DATASETS:
        for var in variant_order:
            if var in all_data[ds]:
                accs, f1s = all_data[ds][var]
                print(f"  {ds:15s} {var:22s} | "
                      f"Max ACC={accs.max()*100:.1f}% | "
                      f"Max F1={f1s.max()*100:.1f}%")

    # Save LaTeX to file
    out_path = BASE_DIR / "ablation_table.tex"
    with open(out_path, "w") as f:
        f.write(latex_output + "\n")
    print(f"\nLaTeX table saved to {out_path}")


if __name__ == "__main__":
    main()
