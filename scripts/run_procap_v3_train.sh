#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pro-Cap V3 supervised RoBERTa-large PBM training (all 4 datasets).
# Loads captions_{train,test}.pkl per dataset, uses 80/20 stratified
# valid split (seed=2025), FIX_LAYERS=2, early stopping patience=8.
# Writes test_procap.jsonl per dataset.
python src/procap_v3_repro/train_procap.py --all --max-length 512
