#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

# HF token: set HUGGING_FACE_HUB_TOKEN in your env or ~/.cache/huggingface/token
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# MATCH-HVD pipeline RESUMPTION from stage 2b.5 onwards.
# Agent stages (2a hate + 2b nonhate) already landed all 4 × 2 = 8 json files
# from the previous 8445 run. Skip them and go directly to Jina-CLIP alignment.

python src/match_repro/time_unit_jina_clip.py --all
python src/match_repro/judgement_vllm.py --all
python src/match_repro/finalize_labelfree.py --all
