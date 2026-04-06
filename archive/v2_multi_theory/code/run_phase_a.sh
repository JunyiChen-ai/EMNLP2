#!/bin/bash
# Phase A: Quick Screening Pipeline
# Run all 5 prompts (4 theories + generic) on all 3 datasets, then screen.
#
# Usage: bash run_phase_a.sh [step]
#   step=1: Run LLM prompts (API calls)
#   step=2: Generate embeddings
#   step=3: Run screening
#   step=all: Run all steps sequentially

set -e
cd /home/junyi/EMNLP2

STEP=${1:-all}
THEORIES="generic itt iet att scm"

echo "============================================"
echo "Phase A: Theory Screening Pipeline"
echo "Step: $STEP"
echo "============================================"

# ---- Step 1: LLM Prompts ----
if [[ "$STEP" == "1" || "$STEP" == "all" ]]; then
    echo ""
    echo "=== Step 1: Running LLM prompts ==="
    for theory in $THEORIES; do
        echo "--- $theory on HateMM ---"
        python prompt_theories.py --theory $theory --dataset_name HateMM --max_concurrent 10
        echo "--- $theory on MHC-EN ---"
        python prompt_theories.py --theory $theory --dataset_name Multihateclip --language English --max_concurrent 10
        echo "--- $theory on MHC-ZH ---"
        python prompt_theories.py --theory $theory --dataset_name Multihateclip --language Chinese --max_concurrent 10
    done
    echo "=== Step 1 DONE ==="
fi

# ---- Step 2: Generate Embeddings ----
if [[ "$STEP" == "2" || "$STEP" == "all" ]]; then
    echo ""
    echo "=== Step 2: Generating embeddings ==="
    for theory in $THEORIES; do
        echo "--- $theory embeddings ---"
        python gen_theory_embeddings.py --theory $theory --dataset_name HateMM
        python gen_theory_embeddings.py --theory $theory --dataset_name Multihateclip --language English
        python gen_theory_embeddings.py --theory $theory --dataset_name Multihateclip --language Chinese
    done
    echo "=== Step 2 DONE ==="
fi

# ---- Step 3: Screening ----
if [[ "$STEP" == "3" || "$STEP" == "all" ]]; then
    echo ""
    echo "=== Step 3: Running screening ==="
    echo "--- HateMM ---"
    python screen_theories.py --dataset_name HateMM --num_seeds 10
    echo "--- MHC-EN ---"
    python screen_theories.py --dataset_name Multihateclip --language English --num_seeds 10
    echo "--- MHC-ZH ---"
    python screen_theories.py --dataset_name Multihateclip --language Chinese --num_seeds 10
    echo "=== Step 3 DONE ==="

    echo ""
    echo "============================================"
    echo "Phase A COMPLETE"
    echo "Check ./screen_results/ for results"
    echo "Check ./logs/ for detailed logs"
    echo "============================================"
fi
