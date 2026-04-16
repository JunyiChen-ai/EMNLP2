#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate lavis_baselines
cd /data/jehc223/EMNLP2

export HUGGING_FACE_HUB_TOKEN=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

# Pro-Cap LAVIS 8-frame variant — BOTH test and train splits, 4 datasets each
# test split serves as the standalone baseline row
# train split is the prerequisite for Mod-HATE's K-shot support set
# Bundled in one sbatch to reuse the LAVIS model load across 8 dataset passes

python src/procap_repro/reproduce_procap_lavis_8frame.py --all --split test
python src/procap_repro/reproduce_procap_lavis_8frame.py --all --split train
