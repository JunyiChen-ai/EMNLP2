# GPT Experiment Plan

## Objective

Pilot all 5 shortlisted ideas for hateful video detection on HateMM only, compare each against the existing `WholeRationaleMLP` baseline, and decide which ideas show positive signal.

The executor should optimize for a fast but defensible pilot:

- Same dataset for all runs: HateMM only
- Same seed set for all variants: 10 seeds
- Same base training infrastructure as `kill_test/run_kill_test.py`
- Simple implementations only; no architecture inflation
- Final judgment based on mean F1 and stability, not anecdotal best-seed wins

## Locked Context

These facts are fixed from the current repo and should not be reinterpreted during execution.

- Codebase read directly:
  - `kill_test/models.py`
  - `kill_test/dataset.py`
  - `kill_test/run_kill_test.py`
- Idea docs read directly:
  - `IDEA_REPORT.md`
  - `IDEA_REPORT_DETAILED_IO.md`
- Current batch contract from `kill_test/dataset.py`:
  - `text`: `[B, 768]`
  - `units`: `[B, 5, 768]`
  - `unit_mask`: `[B, 5]`
  - `audio`: `[B, 768]`
  - `frame`: `[B, 768]`
  - `label`: `[B]`
- HateMM split sizes:
  - train: `757`
  - valid: `109`
  - test: `217`
- Existing training contract from `kill_test/run_kill_test.py`:
  - optimizer: `AdamW`
  - lr: `2e-4`
  - weight decay: `0.02`
  - scheduler: cosine with 5-epoch warmup
  - early stopping patience: `10`
  - gradient clip: `1.0`
  - class weights: `[1.0, 1.5]`
  - checkpoint selection: best validation ACC
  - reported test metrics: ACC and macro-F1

## Fixed Decisions For This Pilot

### 1. Baseline

The comparison baseline is the existing `WholeRationaleMLP` from `kill_test/models.py`, rerun for 10 seeds in the new unified runner.

### 2. Seed list

Use one fixed seed list for every variant:

`[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`

### 3. Metrics to report for every variant

- mean ACC
- std ACC
- mean F1
- std F1
- worst-seed F1
- best-seed F1
- delta mean F1 vs baseline

### 4. Positive-signal rule

An idea counts as showing positive signal if either condition holds:

1. `mean F1 > baseline mean F1 + 1.0pp`
2. Stability win:
   - `mean F1 >= baseline mean F1 - 0.5pp`
   - `std F1 <= 0.8 * baseline std F1`
   - `worst-seed F1 >= baseline worst-seed F1 + 1.0pp`

If neither condition holds, the idea is not a positive pilot signal.

### 5. GTT decision

Do **real token-level GTT** for the main pilot.

Reason:

- `datasets/HateMM/generic_data.json` contains the 5 structured text fields under `generic_response`
- `kill_test/prepare_unit_embeddings.py` already proves the BERT extraction path exists
- HateMM-only extraction is feasible and keeps the idea faithful

Fallback rule:

- A field-level proxy may be used only as a temporary smoke-test helper
- It must be labeled `GTT-lite` and cannot be presented as the final GTT pilot result

### 6. VMBT decision

VMBT is treated as a training recipe on top of a very small multimodal backbone, not as a large new architecture.

Use:

- input: concat(`[text; audio; frame]`) = `[B, 2304]`
- hidden: `[B, 256]`
- output: logits `[B, 2]`

### 7. Simplicity rule

Keep hidden size `256` and dropout `0.3` unless a model requires a small deviation. Do not run broad hyperparameter sweeps. One fixed reasonable config per idea is the goal.

## Required New Artifacts

The executor should produce these artifacts.

- `kill_test/model_gtt.py`
- `kill_test/model_vmbt.py`
- `kill_test/model_borf.py`
- `kill_test/model_rcd.py`
- `kill_test/model_cmde.py`
- `kill_test/prepare_token_embeddings.py`
- `kill_test/run_gpt_experiments.py`

Recommended output location:

- `kill_test/results/gpt_experiments/`

Recommended result files:

- `seed_results.json`
- `summary.csv`
- `summary.md`
- per-run log files

## Stage 0: Freeze The Evaluation Contract

### Checklist

- [ ] Reconfirm the unified runner will use the same optimizer, scheduler, class weights, patience, and checkpoint rule as `run_kill_test.py`
- [ ] Reconfirm the baseline is exactly `WholeRationaleMLP`
- [ ] Reconfirm the seed list is fixed at `42-51`
- [ ] Reconfirm the final table columns before any full pilot is launched
- [ ] Create the result directory structure up front

### Validation Criteria

- One dry run of the baseline completes end-to-end with the new runner
- The new runner writes per-seed outputs in a stable schema
- The summary table format is fixed before launching the 10-seed sweep

## Stage 1: Add Missing GTT Features

This stage should start early because it is the only missing-feature dependency.

### Required implementation

- [ ] Create `kill_test/prepare_token_embeddings.py`
- [ ] Read `datasets/HateMM/generic_data.json`
- [ ] Use the same 5 fields already used by `prepare_unit_embeddings.py`:
  - `content_summary`
  - `target_analysis`
  - `sentiment_tone`
  - `harm_assessment`
  - `overall_judgment`
- [ ] Extract token-level BERT hidden states for each field
- [ ] Save:
  - token embeddings: shape `[5, T, 768]` per video
  - token masks: shape `[5, T]` per video
- [ ] Use `max_length=128` unless a blocker appears
- [ ] Prefer saving token features in a memory-conscious format; `float16` on disk is acceptable if converted safely during loading

### Expected files

- `embeddings/HateMM/unit_token_features.pth`
- `embeddings/HateMM/unit_token_masks.pth`

### Dataset extension requirements

- [ ] Extend data loading so token features are optional rather than mandatory
- [ ] Preserve current batch keys for all non-GTT models
- [ ] When token files are provided, add:
  - `unit_tokens`: `[B, 5, T, 768]`
  - `unit_token_mask`: `[B, 5, T]`

### Validation Criteria

- Feature extraction covers the full HateMM feature universe, or the exact number of missing IDs is documented
- One batch from the updated dataloader has the expected token tensor shapes
- Token masks are non-empty for real text and zero only for padding
- A single forward pass through GTT can consume the new batch format without shape errors

## Stage 2: Implement The 5 Pilot Models

Each idea must live in its own new file. Final inference for every model must return logits `[B, 2]`.

### 2.1 GTT: Grounded Token Trust

### Checklist

- [ ] Implement AV-conditioned token trust scoring from `audio` and `frame`
- [ ] Compute trust scores over `unit_tokens` with shape `[B, 5, T]`
- [ ] Apply trust-weighted token pooling within each field
- [ ] Aggregate the 5 trust-weighted field embeddings into one sample representation
- [ ] Classify into 2 classes
- [ ] Keep the model lightweight; no large cross-attention stack

### Required behavior

- AV is used to score token trust, not as a separate late-fusion branch
- The classifier operates on trust-edited text representations
- If an auxiliary trust regularizer is added, keep it small and fixed across seeds

### Validation Criteria

- Forward shapes:
  - input tokens `[B, 5, T, 768]`
  - trust `[B, 5, T]`
  - pooled fields `[B, 5, 768]`
  - logits `[B, 2]`
- Trust scores are bounded and finite
- Weighted pooling handles padded tokens correctly

### 2.2 VMBT: Variance-Minimized Boundary Training

### Checklist

- [ ] Implement a minimal multimodal backbone in `model_vmbt.py`
- [ ] Expose both hidden feature `[B, 256]` and logits `[B, 2]`
- [ ] Add EMA teacher support in the runner or trainer path
- [ ] Add class-conditional feature alignment loss between student and EMA teacher
- [ ] Add hard-sample reweighting based on confidence or margin
- [ ] Keep the base CE loss and current optimizer/scheduler intact

### Required behavior

- Use a simple fused backbone: `[text; audio; frame] -> 256 -> 2`
- Do not add a second large model family just for VMBT
- Treat VMBT as a regularized training mode, not as a separate evaluation rule

### Validation Criteria

- Hidden feature extraction works on one batch
- EMA updates run without breaking gradient flow
- Alignment loss is finite on batches containing both classes
- Hard-sample weights are normalized and non-degenerate

### 2.3 BORF: Boundary-Only Residual Fusion

### Checklist

- [ ] Stage 1: train a text-only base classifier on `text`
- [ ] Freeze the base text branch after stage 1
- [ ] Stage 2: train audio and frame residual branches
- [ ] Use elementwise text-audio and text-frame interactions
- [ ] Add an ambiguity gate derived from base-text confidence
- [ ] Combine `base_logit + gate * residuals`

### Required behavior

- Text is the main path
- Audio and frame act only as corrections
- Residual branches must not overwrite the base text branch during stage 2

### Validation Criteria

- Stage 1 alone reproduces a normal text-only training path
- Stage 2 updates only residual-branch parameters and gate parameters
- Gate decreases when base-text confidence is high
- Final logits differ from base logits on ambiguous samples

### 2.4 RCD: Residual Correlation Distillation

### Checklist

- [ ] Stage 1: train and freeze a text teacher on `text`
- [ ] Compute residual targets: `one_hot(label) - text_prob`
- [ ] Train an audio residual branch against the residual target
- [ ] Train a frame residual branch against the residual target
- [ ] Combine text logits with scaled residual predictions
- [ ] Tune or fix the residual scales once on validation, then freeze them across seeds

### Required behavior

- Residual branches learn corrections, not full labels
- Distillation loss is regression-style, not CE on the raw labels
- The final model can run end-to-end at test time from stored parameters

### Validation Criteria

- Residual targets are near zero for confident correct teacher predictions
- Audio and frame residual outputs have the same shape as text logits
- Combined logits improve or change predictions on some teacher-error examples
- Alpha/beta scaling is documented explicitly

### 2.5 CMDE: Cross-Modal Description Editor

### Checklist

- [ ] Score each of the 5 unit embeddings against AV evidence
- [ ] Convert support scores into edit gates or soft edit weights
- [ ] Produce edited unit embeddings without free-form text rewriting
- [ ] Pool edited units with a simple attention or weighted sum
- [ ] Add a mild edit regularizer to prevent degenerate deletion

### Required behavior

- Audio/frame edit the text representation before classification
- Audio/frame are not simply concatenated into the final classifier input
- Editing remains field-level for this pilot

### Validation Criteria

- Support scores shape `[B, 5]`
- Edited units shape `[B, 5, 768]`
- Edit gates are bounded and not collapsed to all zeros
- The regularizer stays finite and does not dominate CE

## Stage 3: Build The Unified Runner

Create `kill_test/run_gpt_experiments.py`.

### Checklist

- [ ] Support baseline + all 5 ideas from one CLI
- [ ] Reuse the current AdamW, cosine schedule, class weighting, early stopping, and gradient clipping
- [ ] Support both single-stage and multi-stage methods
- [ ] Support optional GTT token features
- [ ] Save per-seed test metrics and summary metrics
- [ ] Save enough metadata to reproduce each run

### Minimum CLI support

- `--emb_dir`
- `--ann_path`
- `--split_dir`
- `--num_seeds`
- `--seed_offset`
- `--batch_size`
- `--epochs`
- `--output_dir`
- `--models`
- `--token_feature_path`
- `--token_mask_path`

### Runner output requirements

- One record per seed per variant
- Summary metrics aggregated per variant
- A machine-readable format and a human-readable table
- Explicit failure logging if any variant crashes or is skipped

### Validation Criteria

- Baseline runs successfully through the new runner
- At least one multi-stage method and one token-level method complete a 1-seed smoke run
- Result aggregation matches the required metric columns exactly

## Stage 4: Smoke Tests Before Full Sweep

Do not launch the 10-seed full pilot until all variants pass smoke tests.

### Checklist

- [ ] Run baseline for 1 seed
- [ ] Run GTT for 1 seed
- [ ] Run VMBT for 1 seed
- [ ] Run BORF for 1 seed
- [ ] Run RCD for 1 seed
- [ ] Run CMDE for 1 seed
- [ ] Confirm no NaNs, no missing batch keys, no silent shape broadcasting bugs

### Validation Criteria

- All 6 variants finish a smoke run
- Every variant produces finite ACC/F1
- GTT token loading is stable
- Multi-stage checkpoints reload correctly for BORF and RCD

## Stage 5: Full 10-Seed Pilot Execution

### Execution order

Start token extraction as early as possible. While it runs, implement and smoke-test the non-GTT models.

Recommended order for full pilots:

1. Baseline
2. BORF
3. VMBT
4. RCD
5. CMDE
6. GTT

### Checklist

- [ ] Run baseline on all 10 seeds
- [ ] Run each of the 5 ideas on the same 10 seeds
- [ ] Keep hyperparameters fixed across seeds for a given idea
- [ ] Document any model-specific batch-size change, especially for GTT
- [ ] Save logs and metrics immediately after each seed to avoid losing work

### Validation Criteria

- Exactly 60 seed runs complete:
  - 10 baseline
  - 10 GTT
  - 10 VMBT
  - 10 BORF
  - 10 RCD
  - 10 CMDE
- Every variant has a complete 10-seed summary
- No idea is silently dropped from the final table

## Stage 6: Aggregate Results And Issue The Verdict

### Required final table

The final table must include:

| Variant | Mean ACC | Std ACC | Mean F1 | Std F1 | Worst F1 | Best F1 | Delta Mean F1 vs Baseline | Verdict |
|---------|----------|---------|---------|--------|----------|---------|---------------------------|---------|

### Required verdict labels

Use one of:

- `Positive signal`
- `Stability-only signal`
- `Borderline`
- `No signal`
- `Incomplete`

### Verdict rules

- `Positive signal`: mean F1 improvement `> 1.0pp`
- `Stability-only signal`: passes the stability win rule but not the mean-F1 rule
- `Borderline`: some improvement, but fails the locked positive-signal rule
- `No signal`: clearly fails both accuracy and stability criteria
- `Incomplete`: implementation or run failure prevented a proper 10-seed pilot

### Required analysis notes

- [ ] State which ideas improved mean F1 over baseline
- [ ] State which ideas reduced variance
- [ ] State whether any idea improved both mean and worst-seed behavior
- [ ] Call out if GTT was full token-level or only a fallback proxy
- [ ] Call out if VMBT helped stability without improving mean F1

### Validation Criteria

- The table is generated from the stored per-seed results rather than hand-edited numbers
- Baseline deltas are consistent across all rows
- The verdict uses the locked rule above rather than intuition

## Pilot Hyperparameter Defaults

These are recommended defaults to avoid uncontrolled tuning.

- Shared:
  - hidden size: `256`
  - dropout: `0.3`
  - lr: `2e-4`
  - weight decay: `0.02`
  - patience: `10`
  - epochs: `50`
- GTT:
  - `T = 128`
  - no large auxiliary loss unless clearly justified
- VMBT:
  - EMA decay: `0.999`
  - alignment weight `lambda_fda = 0.1`
- BORF:
  - ambiguity threshold `tau = 0.7`
  - gate slope `beta = 10`
- RCD:
  - choose `alpha, beta` from a small validation-only grid once, then freeze
- CMDE:
  - use a small edit regularizer, fixed across seeds

## Non-Goals

These are explicitly out of scope for this pilot.

- No four-dataset evaluation
- No combined methods such as `VMBT + BORF`
- No free-form text rewriting for CMDE
- No large architecture search
- No prompt changes or MLLM regeneration
- No reporting of cherry-picked best seeds as the main story

## Final Acceptance Criteria

This experiment plan is considered successfully executed only if all conditions below are met.

- [ ] All 5 idea implementations exist in separate new model files
- [ ] A unified runner exists and can execute baseline + all 5 ideas
- [ ] Full token-level GTT is implemented and run on HateMM
- [ ] Baseline and all 5 ideas complete 10 seeds each on the same seed list
- [ ] A final results table is produced with the required columns
- [ ] Each idea receives a verdict using the locked positive-signal rule
- [ ] The final write-up explicitly names which ideas show positive signal and which do not

If any one of the above fails, the pilot is incomplete and should be reported as such rather than softened in the summary.

## Final Acceptance Status

Date: `2026-04-07`

Execution status: `Complete`

Research outcome: `Negative / no idea passed the locked positive-signal rule`

### Final Results

| Variant | Mean ACC | Std ACC | Mean F1 | Std F1 | Worst F1 | Best F1 | Delta Mean F1 vs Baseline | Final Verdict |
|---------|----------|---------|---------|--------|----------|---------|---------------------------|---------------|
| baseline | 87.26 | 1.43 | 86.89 | 1.39 | 84.39 | 88.50 | +0.00 | Baseline |
| gtt | 85.16 | 1.32 | 84.44 | 1.29 | 81.45 | 86.06 | -2.45 | No signal |
| vmbt | 87.58 | 1.54 | 86.95 | 1.71 | 83.53 | 88.83 | +0.05 | No signal |
| borf | 87.21 | 1.25 | 86.81 | 1.23 | 84.35 | 88.00 | -0.09 | No signal |
| rcd | 87.63 | 1.16 | 87.25 | 1.14 | 85.25 | 88.50 | +0.36 | Borderline |
| cmde | 86.05 | 1.21 | 85.59 | 1.14 | 84.30 | 87.50 | -1.31 | No signal |

### Locked-Rule Outcome

- No variant exceeded the `>1.0pp` mean-F1 threshold over baseline
- No variant satisfied the full stability-win rule
- `RCD` was the strongest borderline result:
  - mean F1 `+0.36pp`
  - std F1 reduced from `1.39` to `1.14`
  - worst-seed F1 improved from `84.39` to `85.25`
  - but it missed the locked stability thresholds on both std ratio and worst-seed gain

### Interpretation

- `GTT` and `CMDE` failed as trust/editing paradigms on HateMM
- `VMBT` did not deliver the intended stability gains
- `BORF` was effectively neutral
- `RCD` is the only idea with a coherent residual-signal pattern worth limited follow-up

### Final Acceptance Checklist Status

- [x] All 5 idea implementations exist in separate new model files
- [x] A unified runner exists and can execute baseline + all 5 ideas
- [x] Full token-level GTT is implemented and run on HateMM
- [x] Baseline and all 5 ideas complete 10 seeds each on the same seed list
- [x] A final results table is produced with the required columns
- [x] Each idea receives a verdict using the locked positive-signal rule
- [x] The final write-up explicitly names which ideas show positive signal and which do not

Final acceptance decision: `Plan executed successfully; no idea qualified as a positive pilot signal under the locked criteria.`
