#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

export HF_HUB_OFFLINE=1
export CUDA_LAUNCH_BLOCKING=1

# ALARM reproduction — Qwen2.5-VL-72B-AWQ via HF transformers.
# 5-stage pipeline: Label / make_embeddings / conduct_retrieval /
# Experience / InPredict. Uses 8 frames per video (meme→video rule)
# and 16 frames for Experience pairwise calls.
python src/alarm_repro/reproduce_alarm.py --all
