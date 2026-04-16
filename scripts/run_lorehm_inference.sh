#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

# HF token: set HUGGING_FACE_HUB_TOKEN in your env or ~/.cache/huggingface/token

# LoReHM bf16 full inference across 4 datasets.
# 2-GPU sbatch: loads llava-v1.6-34b-hf at bf16 via device_map="auto"
# sharded across both GPUs. Single-tile 336x336 per frame, 16 frames
# per video via LLaVA-Next native multi-image path. No bnb, no grid
# composite (user directive 2026-04-15).
# Priority baseline — runs alone (2-GPU cap rule).
python src/lorehm_repro/reproduce_lorehm.py --all
