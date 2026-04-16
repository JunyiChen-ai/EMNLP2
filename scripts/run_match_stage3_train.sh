#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

# MATCH stage 3 — supervised training per dataset.
# Late-fusion MLP over 4 text streams (transcript/judge/hate/nonhate
# via BERT-base CLS) + ViViT-b + MFCC. AdamW lr=5e-4, 50 epochs,
# early stop patience 8, seed 2025. Output per dataset at
# results/match_qwen2vl_7b/<dataset>/test_match.jsonl.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/match_repro/stage3/train_match.py --all --batch-size 16
