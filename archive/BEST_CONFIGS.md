# Best Configurations Reference

## v1: SCM + Q-MoE + MR2

Best per-dataset configs (mean±std F1, 200 seeds):

| Dataset | Config | Mean F1 | Max F1 | Path |
|---------|--------|:-------:|:------:|------|
| HateMM | α=0.3, β=0.0 | 87.1±2.6 | 92.2 | `v1_scm_method/results/seed_searches/scm_mr2/HateMM_a0.3_b0.0_lb0.01_off0` |
| MHC-EN | α=0.1, β=0.1 | 75.5±3.5 | 82.3 | `v1_scm_method/results/seed_searches/scm_mr2/MHC_En_a0.1_b0.1_lb0.01_off0` |
| MHC-ZH | α=0.1, β=0.0 | 80.0±2.6 | 87.7 | `v1_scm_method/results/seed_searches/scm_mr2/MHC_Ch_a0.1_b0.0_lb0.01_off0` |
| ImpliHateVid | α=0.3, β=0.0 | 91.0±1.1 | 93.5 | `v1_scm_method/results/seed_searches/scm_mr2/ImpliHateVid_a0.3_b0.0_lb0.01_off0` |

Model code: `v1_scm_method/code/main_scm_qmoe_qels_mr2.py`

## v1: SCM + Q-MoE (no MR2 baseline)

| Dataset | Mean F1 | Path |
|---------|:-------:|------|
| HateMM | ~85.9 | `v1_scm_method/results/seed_searches/scm_qels/` |

## v2: ITT (4-channel gating)

| Dataset | Mean F1 | Max F1 | Path |
|---------|:-------:|:------:|------|
| HateMM | 87.6±1.4 | 90.4 | `v2_multi_theory/results/itt/HateMM_off0` |
| MHC-EN | 76.0±2.6 | 82.9 | `v2_multi_theory/results/itt/MHC_En_off0` |
| MHC-ZH | 82.5±1.7 | 86.0 | `v2_multi_theory/results/itt/MHC_Ch_off0` |
| ImpliHateVid | — | — | Not available (LLM not queried) |

Model code: `v1_scm_method/code/ablation_itt.py` (also serves as ITT main)

## v1/v2: Generic Prompt Baselines

| Dataset | Generic+QMoE F1 | Generic+Flat F1 | Path |
|---------|:---------------:|:---------------:|------|
| HateMM | 87.4±1.8 | 87.3±1.6 | `v1_scm_method/results/ablations/ablation_results/HateMM_generic_prompt*` |
| MHC-EN | 74.1±2.2 | 74.3±3.9 | same pattern |
| MHC-ZH | 77.9±2.3 | 78.3±2.0 | same pattern |
| ImpliHateVid | 92.6±0.7 | 92.9±0.6 | same pattern |

## Phase A: Theory Screening (same architecture for all)

| Theory | HateMM F1 | MHC-EN F1 | MHC-ZH F1 |
|--------|:---------:|:---------:|:---------:|
| No rationale | 81.5±1.2 | 68.3±4.3 | 71.3±2.1 |
| Generic | **88.1±1.9** | 74.7±2.4 | 77.6±1.7 |
| ITT | 86.9±1.8 | **76.0±2.6** | **79.1±1.8** |
| IET | 84.6±2.0 | 74.2±4.2 | 76.3±2.4 |
| ATT | 85.7±1.6 | 71.3±2.2 | 74.5±3.0 |
| SCM | 87.1±1.7 | 76.1±2.2 | 76.2±3.0 |

Path: `v2_multi_theory/results/screening/screen_results/`

## Key Baselines

| Method | HateMM F1 | MHC-EN F1 | MHC-ZH F1 | ImpliHateVid F1 |
|--------|:---------:|:---------:|:---------:|:--------------:|
| Base (no rationale) | 79.4±2.0 | 64.0±7.2 | 68.2±2.3 | 85.5±0.7 |
| MLLM direct | 84.4 | 62.8 | 61.6 | 66.4 |
| Field-only MLP | 87.4 | 73.9 | 77.2 | 90.3 |

## Available Data

- **LLM outputs**: SCM, ITT, IET, ATT, Generic for HateMM/MHC-EN/MHC-ZH; SCM, Generic for ImpliHateVid
- **Embeddings**: BERT mean pool for all theories, stored in `embeddings/`
- **Base modalities**: text/audio/frame features symlinked from EMNLP2026
- **Datasets**: 4 datasets with train/val/test splits in `datasets/`
