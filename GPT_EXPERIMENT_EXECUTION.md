# GPT Experiment Execution

## Goal
Pilot all 5 ideas (GTT, VMBT, BORF, RCD, CMDE) on HateMM, compare against WholeRationaleMLP baseline.

## Thread ID
019d62aa-b91f-79a2-aa5d-72c02dba0fb4

## Execution Log

### Stage 0: Freeze Evaluation Contract ✅
- Created result directory: `kill_test/results/gpt_experiments/`
- Confirmed: same optimizer/scheduler/class weights/patience as run_kill_test.py
- Confirmed: baseline is WholeRationaleMLP
- Confirmed: seed list [42-51]
- Dry run baseline passed (1 seed, 5 epochs → ACC=84.19, F1=83.85)

### Stage 1: Token Feature Extraction ✅
- Created `kill_test/prepare_token_embeddings.py`
- Extracted token-level BERT hidden states for all 1066 HateMM videos
- Saved: `embeddings/HateMM/unit_token_features.pth` [5, 128, 768] per video (float16)
- Saved: `embeddings/HateMM/unit_token_masks.pth` [5, 128] per video

### Stage 2: Model Implementations ✅
- `kill_test/model_gtt.py` — GroundedTokenTrust
- `kill_test/model_vmbt.py` — VMBTModel + VMBTTrainer
- `kill_test/model_borf.py` — BORFTextBase + BORFResidual + BORFFullModel
- `kill_test/model_rcd.py` — RCDTextTeacher + RCDResidualBranch + RCDCombined
- `kill_test/model_cmde.py` — CrossModalDescriptionEditor

### Stage 3: Unified Runner ✅
- `kill_test/run_gpt_experiments.py` — supports all 6 variants from one CLI

### Stage 4: Smoke Tests ✅
All 6 variants passed (1 seed, 5 epochs):
| Model    | ACC   | F1    |
|----------|-------|-------|
| baseline | 84.19 | 83.85 |
| gtt      | 83.72 | 83.35 |
| vmbt     | 86.05 | 85.62 |
| borf     | 84.65 | 84.10 |
| rcd      | 85.58 | 85.06 |
| cmde     | 84.65 | 84.35 |

### Stage 5: Full 10-Seed Pilot ✅
- Completed: all 6 variants × 10 seeds × 50 epochs
- Results: `kill_test/results/gpt_experiments/seed_results_20260407_001236.json`
- Summary: `kill_test/results/gpt_experiments/summary_20260407_001236.md`

### Stage 6: Final Results & Verdict ✅

| Variant | Mean F1 | Std F1 | Worst F1 | Δ Mean F1 | Verdict |
|---------|---------|--------|----------|-----------|---------|
| baseline | 86.89 | 1.39 | 84.39 | +0.00 | Baseline |
| gtt | 84.44 | 1.29 | 81.45 | -2.45 | No signal |
| vmbt | 86.95 | 1.71 | 83.53 | +0.05 | No signal |
| borf | 86.81 | 1.23 | 84.35 | -0.09 | No signal |
| rcd | 87.25 | 1.14 | 85.25 | +0.36 | Borderline |
| cmde | 85.59 | 1.14 | 84.30 | -1.31 | No signal |

**GPT-5.4 Final Acceptance**: Plan executed successfully. No idea passed the locked positive-signal rule.

**Key learnings**:
1. GTT & CMDE failed: AV as trust validator/text editor doesn't work on HateMM. Global AV features can't reliably validate fine-grained text.
2. VMBT failed its own goal: increased variance instead of reducing it.
3. BORF neutral: residual correction at decision level is too weak.
4. RCD is the only one with coherent directional improvement on all stability metrics. Worth bounded follow-up.

**GPT recommendation**: Only pursue RCD with 2-3 targeted changes (selective residual application, better calibration). Stop if no clear win after that.
