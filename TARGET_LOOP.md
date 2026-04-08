# Target-Driven Research Loop

## Target
- Metric: Accuracy (binary classification)
- Condition: HateMM acc >= 90.0, MultiHateClip (CN+EN) acc >= 85.0
- Dataset: HateMM test set (216 samples), MultiHateClip Chinese test (199), MultiHateClip English test (199)
- Binary mapping: For MultiHateClip, Offensive+Hateful → Hateful, Normal → Normal
- Method constraints:
  1. Must use frozen Qwen3-VL-8B via vllm for MLLM inference
  2. Must include downstream tri-modal classifier (not text-only)
  3. Must be a unified method solving a problem previous methods haven't addressed
  4. Must directly input video (no downgrade/discard)
  5. Single GPU constraint
  6. Must checkpoint MLLM results for resume capability
  7. OOM fallback with dynamic batch adjustment
  8. Scientifically novel approach

## Current best: N/A (fresh start)

## Iteration Log
