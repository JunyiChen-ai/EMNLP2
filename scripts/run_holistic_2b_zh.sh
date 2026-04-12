#!/bin/bash
#SBATCH --job-name=h2b_zh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -e

cd /data/jehc223/EMNLP2
source activate SafetyContradiction

MODEL="Qwen/Qwen3-VL-2B-Instruct"
BS=32

echo "=== MHClip_ZH clean ==="
python src/score_holistic.py --dataset MHClip_ZH --split test --model $MODEL --prompt-variant clean --batch-size $BS

echo "=== MHClip_ZH mechanism ==="
python src/score_holistic.py --dataset MHClip_ZH --split test --model $MODEL --prompt-variant mechanism --batch-size $BS

echo "=== ZH done ==="
