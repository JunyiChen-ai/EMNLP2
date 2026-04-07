# Target-Driven Research Loop

## Target
- HateMM: accuracy > 90.0
- MHClip_EN: accuracy > 88.0
- MHClip_ZH: accuracy > 88.0
- Method constraints: elegant, scientifically motivated, one core principle, not bag of tricks

## Current Best (baseline)
- HateMM: ~87-88 F1 (simple MLP on MLLM rationale text)
- MHClip_EN / MHClip_ZH: TBD

## Approach: Observation-Grounded Classification (OGC)

**Core Principle**: Separate MLLM rationale into grounded observations vs. inferential interpretations. Use raw modalities (frame, audio) to verify observations. Gate interpretation influence by grounding confidence.

**Pipeline**:
1. Generate diagnostic rationales with Qwen3-VL-32B (2 shards, 1 GPU each)
2. Parse rationales: Section 1 (OBSERVED EVIDENCE) → observations, Sections 2-4 → interpretations
3. Encode observations and interpretations separately (sentence-transformers)
4. Extract frame features (CLIP ViT-L/14) and audio features (Wav2Vec2)
5. OGC Fusion: grounding_gate(obs, frame, audio) → g; fused = g * interpretation + (1-g) * modalities
6. Classify from [observations; fused] → logits

---

## Iteration 1 — In Progress

**Phase**: Rationale generation (Qwen3-VL-32B, sharded inference)
**Status**: Submitting Slurm jobs
