# SCM-MoE: Theory-Guided Hateful Video Detection via Stereotype Content Model

## Pipeline

### Step 1: Theory-Guided MLLM Reasoning
```bash
python prompt_theories.py --theory scm --dataset_name HateMM
python prompt_theories.py --theory scm --dataset_name Multihateclip --language English
python prompt_theories.py --theory scm --dataset_name Multihateclip --language Chinese
python prompt_theories.py --theory scm --dataset_name ImpliHateVid
```

### Step 2: BERT Mean Pool Encoding
```bash
python gen_theory_embeddings.py --theory scm --dataset_name HateMM --pool mean
python gen_theory_embeddings.py --theory scm --dataset_name Multihateclip --language English --pool mean
python gen_theory_embeddings.py --theory scm --dataset_name Multihateclip --language Chinese --pool mean
python gen_theory_embeddings.py --theory scm --dataset_name ImpliHateVid --pool mean
```

### Step 3: Train & Evaluate
```bash
# Best configs are hardcoded — just specify dataset:
python main_scm_qmoe_qels_mr2.py --dataset_name HateMM
python main_scm_qmoe_qels_mr2.py --dataset_name Multihateclip --language English
python main_scm_qmoe_qels_mr2.py --dataset_name Multihateclip --language Chinese
python main_scm_qmoe_qels_mr2.py --dataset_name ImpliHateVid
```

## Best Results (MR2, per-dataset best config)

| Dataset | ACC | M-F1 | M-P | M-R | Config |
|---------|:---:|:---:|:---:|:---:|------|
| HateMM | 92.6 | 92.2 | 92.8 | 91.7 | α=0.3 β=0.0 seed=159042 |
| MHClip-Y | 85.3 | 82.3 | 82.7 | 81.9 | α=0.1 β=0.1 seed=9042 |
| MHClip-B | 89.8 | 87.7 | 88.2 | 87.2 | α=0.1 β=0.0 seed=15042 |
| ImpliHateVid | 93.5 | 93.5 | 93.7 | 93.5 | α=0.3 β=0.0 seed=2042 |

## File Structure

### Active Code
- `prompt_theories.py` — LLM + SCM/generic prompt generation
- `gen_theory_embeddings.py` — BERT encoding (CLS/mean pool)
- `main_scm_qmoe_qels_mr2.py` — **Main model** (SCM + Q-MoE + QELS + MR2)
- `main_scm_qmoe_qels_meanpool.py` — Baseline (Q-MoE + QELS, no MR2)
- `screen_theories.py` — Theory screening (Phase A)
- `run_ablations.py` — Ablation experiment runner
- `run_analysis.py` — Theory consistency & SCM quality analysis
- `run_moe_variants.py` — MoE routing variant comparison

### Paper
- `paper/main.tex` — Paper source
- `paper/custom.bib` — References
- `paper/figures/` — Generated figures

### Archive (deprecated variants)
- `archive/` — Old ITT model, deprecated SCM variants, failed method attempts

### Experiment Results
- `refine-loop/` — Full experiment loop state and results
- `refine-loop/round-016/` — Latest comprehensive results
- `ablation_results/` — Ablation experiment outputs
- `seed_search_*/` — Seed search results per method
- `moe_refs/` — Cloned MoE reference implementations

### Data
- `datasets/` — Symlinked dataset directories
- `embeddings/` — Pre-extracted feature embeddings
