#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

export HUGGING_FACE_HUB_TOKEN=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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# MATCH-HVD stage 2 pipeline — bundled single GPU, 4 datasets
# Stage 2a: hate-evidence agent (HF Qwen2-VL-7B-Instruct)
# Stage 2b: nonhate-evidence agent (HF Qwen2-VL-7B-Instruct)
# Stage 2b.5: Jina-CLIP-v2 per-frame alignment
# Stage 2c: vLLM Qwen2.5-VL-7B-Instruct judgement (reads time_unit_*.json, produces judge.json)
# Stage 4 (peek): label-free parser on judge.json (CPU, included here for convenience)

python src/match_repro/run_match_agents.py --all --agent hate --split test
python src/match_repro/run_match_agents.py --all --agent nonhate --split test
python src/match_repro/time_unit_jina_clip.py --all
python src/match_repro/judgement_vllm.py --all
python src/match_repro/finalize_labelfree.py --all
