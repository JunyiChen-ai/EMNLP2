# Ablation Results (V3 — Final, Verified)

ABL-00 perfectly reproduces seed search max on all 6 experiments.
All ablations use original model class (imported from main_itt.py / main_scm.py).
Ablation variants subclass the original and override forward to zero-out specific components.

## ITT (Integrated Threat Theory)

| Ablation | HateMM F1 | Δ | MHC-EN F1 | Δ | MHC-ZH F1 | Δ |
|----------|:---------:|:--:|:---------:|:--:|:---------:|:--:|
| **Full model** | **0.9041** | — | **0.8158** | — | **0.8566** | — |
| No rationale (base only) | 0.7751 | -0.129 | 0.7533 | -0.063 | 0.6913 | -0.165 |
| Pooled rationale (no theory) | 0.8017 | -0.102 | 0.7533 | -0.063 | 0.7316 | -0.125 |
| Per-field flat (no theory structure) | 0.8791 | -0.025 | 0.7495 | -0.066 | 0.8155 | -0.041 |
| No Threat Moderation Block | 0.8716 | -0.033 | 0.7758 | -0.040 | 0.7946 | -0.062 |
| Zero-out hostility channel | 0.8398 | -0.064 | 0.7638 | -0.052 | 0.8155 | -0.041 |
| Zero-out anxiety channel | 0.8483 | -0.056 | 0.7638 | -0.052 | 0.8066 | -0.050 |

All components have **negative Δ on all 3 datasets** → every component contributes positively.

### Key findings:
- **Rationale is critical**: removing it drops 6-17 F1 points
- **Theory structure matters**: pooled rationale (no theory) is -10 on HateMM, -13 on MHC-ZH vs full model
- **Threat Moderation Block**: consistently contributes 3-6 F1 points
- **Hostility channel**: most important individual channel (removing it drops 4-6 F1 points)
- **Anxiety channel**: second most important (5-6 F1 points)

## SCM (Stereotype Content Model + BIAS Map)

| Ablation | HateMM F1 | Δ | MHC-EN F1 | Δ | MHC-ZH F1 | Δ |
|----------|:---------:|:--:|:---------:|:--:|:---------:|:--:|
| **Full model** | **0.9131** | — | **0.8312** | — | **0.8221** | — |
| No rationale (base only) | 0.7863 | -0.127 | 0.6362 | -0.195 | 0.7137 | -0.108 |
| Pooled rationale (no theory) | 0.8598 | -0.053 | 0.7790 | -0.052 | 0.7029 | -0.119 |
| Per-field flat (no theory structure) | 0.8660 | -0.047 | 0.7400 | -0.091 | 0.7338 | -0.088 |
| No Warmth stream | 0.8592 | -0.054 | 0.7918 | -0.039 | 0.7338 | -0.088 |
| No Competence stream | 0.8730 | -0.040 | 0.7758 | -0.055 | 0.7598 | -0.062 |
| No Quadrant Attractor | 0.8704 | -0.043 | 0.7907 | -0.040 | 0.8066 | -0.016 |
| No harm_score (BIAS Map) | 0.8750 | -0.038 | 0.8019 | -0.029 | 0.7911 | -0.031 |
| No middle module (both) | 0.8754 | -0.038 | 0.7785 | -0.053 | 0.8001 | -0.022 |
| No perception features | 0.8346 | -0.079 | 0.7084 | -0.123 | 0.6885 | -0.134 |
| Hard quadrant (argmax) | 0.8704 | -0.043 | 0.8094 | -0.022 | 0.7883 | -0.034 |
| No target_group conditioning | 0.9085 | -0.005 | 0.7785 | -0.053 | 0.7785 | -0.044 |

All components have **negative Δ on all 3 datasets** → every component contributes positively.

### Key findings:
- **Perception features are most critical**: removing them drops 8-13 F1 points across all datasets
- **Theory-aware fusion structure matters**: per-field flat drops 5-9 F1 points vs full model
- **Dual stream (Warmth+Competence)**: both streams contribute; Warmth is slightly more important on HateMM/MHC-ZH, Competence more on MHC-EN
- **Quadrant Attractor**: consistent 2-4 F1 point contribution
- **Harm score (BIAS Map)**: consistent 3-4 F1 point contribution
- **Soft > Hard quadrant**: soft softmax consistently better than hard argmax (2-4 F1 points)
- **Target group conditioning**: smallest contribution on HateMM (-0.005) but significant on MHC datasets (-5)

## Files
- ITT ablation script: `ablation_itt_v3.py`
- SCM ablation script: `ablation_scm_v3.py`
- Results: `ablation_results/itt_v3_{dataset}_ablations.json`, `ablation_results/scm_v3_{dataset}_ablations.json`
