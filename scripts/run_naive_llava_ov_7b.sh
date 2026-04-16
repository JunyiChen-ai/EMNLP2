#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

python src/naive_baseline/score_naive_2b.py \
    --all --split test \
    --model llava-hf/llava-onevision-qwen2-7b-ov-hf \
    --batch-size 4
