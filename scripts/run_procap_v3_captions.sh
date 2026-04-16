#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

# HF token: set HUGGING_FACE_HUB_TOKEN in your env or ~/.cache/huggingface/token

# Pro-Cap V3 Qwen2-VL-7B 16-frame multi-image caption generation.
# Runs test split first (needed for test prediction), then train split
# (needed for supervised PBM training + Mod-HATE K-shot support set).
# Each invocation loads vLLM once and iterates 4 datasets via --all.
python src/procap_v3_repro/generate_captions_qwen2vl.py --all --split test
python src/procap_v3_repro/generate_captions_qwen2vl.py --all --split train
