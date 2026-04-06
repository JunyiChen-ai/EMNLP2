# Best Model Pipelines

## ITT (Integrated Threat Theory)

### Step 1: LLM Rationale Generation
```bash
conda run -n HVGuard python prompt_theories.py --theory itt --dataset_name HateMM --max_concurrent 10
conda run -n HVGuard python prompt_theories.py --theory itt --dataset_name Multihateclip --language English --max_concurrent 10
conda run -n HVGuard python prompt_theories.py --theory itt --dataset_name Multihateclip --language Chinese --max_concurrent 10
```
- Output: `datasets/{dataset}/itt_data.json`

### Step 2: Field Embedding
```bash
conda run -n HVGuard python gen_theory_embeddings.py --theory itt --dataset_name HateMM
conda run -n HVGuard python gen_theory_embeddings.py --theory itt --dataset_name Multihateclip --language English
conda run -n HVGuard python gen_theory_embeddings.py --theory itt --dataset_name Multihateclip --language Chinese
```
- Output: `embeddings/{dataset}/itt_{field}_features.pth` (6 fields + 1 pooled rationale)

### Step 3: Training + Seed Search + Retrieval
```bash
conda run -n HVGuard python main_itt.py --dataset_name HateMM --num_runs 200 --seed_offset 0
conda run -n HVGuard python main_itt.py --dataset_name Multihateclip --language English --num_runs 200 --seed_offset 0
conda run -n HVGuard python main_itt.py --dataset_name Multihateclip --language Chinese --num_runs 200 --seed_offset 0
```
- Output: `seed_search_itt/{dataset}_off{offset}/all_results.json` + `best_model.pth`

### Best Configs
| Dataset | Seed | ACC | F1 | Whiten | kNN | k | temp | alpha | thresh |
|---------|:----:|:---:|:--:|--------|-----|:-:|:----:|:-----:|:------:|
| HateMM | 1042 | 0.9070 | 0.9041 | spca_r32 | csls | 25 | 0.1 | 0.5 | -0.18 |
| MHC-EN | 28042 | 0.8528 | 0.8158 | spca_r32 | cosine | 10 | 0.1 | 0.4 | 0.12 |
| MHC-ZH | 29042 | 0.8790 | 0.8566 | spca_r32 | cosine | 15 | 0.1 | 0.5 | None |

---

## SCM (Stereotype Content Model + BIAS Map)

### Step 1: LLM Rationale Generation
```bash
conda run -n HVGuard python prompt_theories.py --theory scm --dataset_name HateMM --max_concurrent 10
conda run -n HVGuard python prompt_theories.py --theory scm --dataset_name Multihateclip --language English --max_concurrent 10
conda run -n HVGuard python prompt_theories.py --theory scm --dataset_name Multihateclip --language Chinese --max_concurrent 10
```
- Output: `datasets/{dataset}/scm_data.json`

### Step 2: Field Embedding
```bash
conda run -n HVGuard python gen_theory_embeddings.py --theory scm --dataset_name HateMM
conda run -n HVGuard python gen_theory_embeddings.py --theory scm --dataset_name Multihateclip --language English
conda run -n HVGuard python gen_theory_embeddings.py --theory scm --dataset_name Multihateclip --language Chinese
```
- Output: `embeddings/{dataset}/scm_{field}_features.pth` (5 fields + 1 pooled rationale)

### Step 3: Training + Seed Search + Retrieval
```bash
conda run -n HVGuard python main_scm.py --dataset_name HateMM --num_runs 200 --seed_offset 0
conda run -n HVGuard python main_scm.py --dataset_name Multihateclip --language English --num_runs 200 --seed_offset 0
conda run -n HVGuard python main_scm.py --dataset_name Multihateclip --language Chinese --num_runs 200 --seed_offset 0
```
- Output: `seed_search_scm/{dataset}_off{offset}/all_results.json` + `best_model.pth`

### Best Configs
| Dataset | Seed | ACC | F1 | Whiten | kNN | k | temp | alpha | thresh |
|---------|:----:|:---:|:--:|--------|-----|:-:|:----:|:-----:|:------:|
| HateMM | 142042 | 0.9163 | 0.9131 | zca | cosine | 10 | 0.05 | 0.45 | None |
| MHC-EN | 163042 | 0.8650 | 0.8312 | spca_r48 | cosine | 15 | 0.02 | 0.2 | 0.08 |
| MHC-ZH | 12042 | 0.8535 | 0.8221 | spca_r48 | cosine | 10 | 0.05 | 0.45 | None |

---

## Shared Dependencies
- Base features (pre-computed, from EMNLP2026): `embeddings/{dataset}/text_features.pth`, `wavlm_audio_features.pth`, `frame_features.pth`
- Datasets (symlinked): `datasets/` → `/home/junyi/EMNLP2026/datasets/`
- Environment: conda env `HVGuard`
- API key: `.env` → `/home/junyi/EMNLP2026/.env`
