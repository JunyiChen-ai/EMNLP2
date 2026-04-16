#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

# HF token: set HUGGING_FACE_HUB_TOKEN in your env or ~/.cache/huggingface/token

# ALARM reproduction — Qwen2.5-VL-72B-AWQ via HF transformers.
# 5-stage pipeline: Label / make_embeddings / conduct_retrieval /
# Experience / InPredict. Uses 8 frames per video (meme→video rule)
# and 16 frames for Experience pairwise calls.
python src/alarm_repro/reproduce_alarm.py --all
