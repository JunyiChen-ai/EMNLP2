#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate lavis_baselines
cd /data/jehc223/EMNLP2

# HF token: set HUGGING_FACE_HUB_TOKEN in your env or ~/.cache/huggingface/token

# Pro-Cap LAVIS faithful repro — all 4 datasets, test split, single GPU
python src/procap_repro/reproduce_procap_lavis.py --all --split test
