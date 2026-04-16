#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python src/naive_baseline/score_naive_2b.py \
    --all --split test \
    --model llava-hf/llava-v1.6-mistral-7b-hf \
    --batch-size 4
