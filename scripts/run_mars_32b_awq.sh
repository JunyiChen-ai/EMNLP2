#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

export HUGGING_FACE_HUB_TOKEN=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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# MARS faithful re-run with Qwen2.5-VL-32B-Instruct-AWQ
# Replaces the 2B pilot (results/mars_2b/) with the paper's actual backbone
# 4 datasets × 4-stage pipeline, single GPU

python src/mars_repro/reproduce_mars_32b_awq.py --all --split test
