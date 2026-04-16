#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

export HUGGING_FACE_HUB_TOKEN=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

# LoReHM RSA retrieval — build rel_sampl.json × 4 datasets via
# jinaai/jina-clip-v2 pooled 8-frame video features.
# Single-GPU, jina-clip-v2 is small (~600M). Serial over 4 datasets in
# one model load via --all.
python src/lorehm_repro/retrieval.py --all --pool-topk 50
