#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

export HUGGING_FACE_HUB_TOKEN=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

# Mod-HATE video-adapted reproduction.
# LLaMA-7B + 3 pretrained LoRA modules (hate-exp, meme-captions,
# hate-speech) composed via Nevergrad NGOpt on K-shot support set loss.
# Consumes Pro-Cap V3 captions (results/procap_v3/<dataset>/captions_{split}.pkl).
# K=4 and K=8 per dataset, all 4 datasets.
python src/mod_hate_repro/reproduce_mod_hate.py --all --shots 4 8 --no-load-8bit
