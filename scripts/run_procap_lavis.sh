#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate lavis_baselines
cd /data/jehc223/EMNLP2

export HUGGING_FACE_HUB_TOKEN=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

# Pro-Cap LAVIS faithful repro — all 4 datasets, test split, single GPU
python src/procap_repro/reproduce_procap_lavis.py --all --split test
