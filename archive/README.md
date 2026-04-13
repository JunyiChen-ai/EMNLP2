# archive/

Contents of `src/` that are NOT on the baseline critical path. Moved here
on **2026-04-13** during the post-shutdown cleanup, after end-to-end
reproduction of the 2B `binary_nodef` baseline confirmed that the only
files actually needed by the baseline command are:

```
src/baseline/data_utils.py
src/baseline/quick_eval_all.py
src/baseline/score_holistic_2b.py
```

(The 3 baseline files were relocated from top-level `src/` into a
`src/baseline/` subfolder on 2026-04-13 later that same day, to isolate
frozen baseline code from new experiment subfolders like
`src/naive_baseline/` and `src/mars_repro/`. Nothing about the archive's
own contents changed in that relocation.)

Everything else under `src/` was experimental code from earlier iterations,
prompt-paradigm v1-v6, ~120 meta-selector pilots, post-shutdown diagnostic
probes, or legacy scripts from iterations 0-3. None of it is imported by
the baseline command. None of it has been deleted; it has all been moved
to this archive.

## Layout

| Subfolder | Files | What's in it |
|---|---|---|
| `prompt_paradigm_v1_to_v6/` | 14 .py + 2 .sh | Team session 2026-04-13 prompt-paradigm track. The 6 iterations v1 (Observe-then-Judge), v2 (Factored Verdict), v3 (Polarity-Calibrated Probes), v4 (Modality-Split), v5 (Per-Rule Disjunction Readout), v6 (Coarse Axes Prompt), plus their evaluators and the v5/v6 pipeline runner shell scripts. **Note**: v3's `polarity_calibration.py` contains the silent prompt-text drift discovered late in the session — see STATE_ARCHIVE §"v3 p_evidence row correction". |
| `meta_selector_pilots/` | 156 | Team session 2026-04-13 meta-selector track. ~120 CPU pilots covering threshold methods (Otsu, GMM, GHT, Kapur, Yen, Li-Lee, Renyi, Rosin, Triangle, MAD, …), feature extraction, label-free selectors, fusion candidates, and atom-level diagnostics. Contains the rejected MAD-rule submission (sub-atom FP-phantom incident). See STATE_ARCHIVE §"Team session 2026-04-13" for the full failure catalogue. |
| `post_shutdown_probes/` | 18 | Diagnostic probes from this session and earlier ad-hoc analysis: `probe_selector_stats.py`, `probe_selector_scanfold.py`, `probe_crossconfig_fusion.py`, `probe_triple_fusion.py`, `probe_fusion_extended_lf.py`, `probe_fusion_quantile_sweep.py` (cross-config fusion finding); `diagnose_*.py` (5 prompt-pair / score-complementarity diagnostics); `analyze_*.py` (7 distributional analyses from earlier iterations). |
| `legacy_iteration_scripts/` | 17 | Pre-team-session scripts from iterations 0-3: `clip_filter.py`, `content_free_calibration.py`, `score_holistic.py`, `score_holistic_8b.py`, `score_preconditions.py`, `extract_preconditions.py`, `induce_rules.py`, `objectify_rules.py`, `refine_preconditions.py`, `score_quad.py`, `classify.py`, `infer_vllm.py`, `observe_then_judge.py` (top-level, distinct from the prompt_paradigm version), `observe_training.py`, `eval_triclass_testfit.py`, `calibrate_and_threshold.py`, `reproduce_best.py`. |

**Total**: 205 Python files (plus 2 shell scripts).

## How to re-run an archived script

Archived scripts are **NOT runnable in place**. They all do
`from data_utils import ...` and `from quick_eval_all import ...` assuming
those modules live in the same directory (they're loaded via
`sys.path.insert(0, os.path.dirname(__file__))` at the top of each script).
Once moved into `archive/<subdir>/`, the import path is broken because
`data_utils.py` and `quick_eval_all.py` now live in `src/baseline/`.

To re-run any single archived script:

```bash
cp archive/<subdir>/<script>.py src/baseline/
python src/baseline/<script>.py [args ...]
```

After the run, you can delete the copy from `src/baseline/` to keep it
clean. Do **not** edit the archived copies — they are historical artifacts.

For multi-file groups (e.g., the prompt_paradigm v1-v6 evaluators which
import from each other), copy the whole `prompt_paradigm_v1_to_v6/`
directory into `src/baseline/` as `src/baseline/prompt_paradigm/` to
restore the original package layout.

## Why archived

- **prompt_paradigm v1-v6**: all six iterations failed Gate 2. None
  produced a label-free strict-beat of baseline. v6 was running at the
  team-shutdown command and exited post-shutdown.
- **meta_selector_pilots**: ~120 CPU pilots over ~13 hours of session
  produced no passing method. Best result was the GHT grid finding that
  EN binary_nodef oracle ceiling is 0.7702 (with mF1 regression). MAD-rule
  submission was rejected for sub-atom FP-phantom dependence.
- **post_shutdown_probes**: diagnostic probes that established several
  structural negative results — selector criterion exhaustion, cross-config
  fusion oracle finding, the v3 p_evidence prompt-drift root-cause.
  None produced a passing label-free method.
- **legacy_iteration_scripts**: pre-team-session iterations 0-3. Replaced
  by the holistic baseline that's now in `src/baseline/score_holistic_2b.py`.

## Pointer to the narrative

For the full project history, decisions, and what each iteration tried
and why it failed, see `STATE_ARCHIVE.md` at the repo root. The relevant
sections are:

- §"CURRENT BASELINE" — what the live baseline is
- §"Team session 2026-04-13" — the 6-iteration prompt-paradigm + meta-selector log
- §"Post-shutdown diagnostic" — the scanfold and cross-config fusion findings
- §"Post-shutdown follow-up" — the cross-config fusion oracle cell discovery
- §"v3 p_evidence row correction" — the prompt-drift root-cause finding
- §"Baseline reproduction + src/ archival (2026-04-13 later session)" — this archival event
