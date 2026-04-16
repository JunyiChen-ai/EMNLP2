# MATCH-HVD stage 3 — supervised training (reconstructed)

## Critical upstream gap

`external_repos/match_hvd/src/model/MATCH/MATCH.py` is **0 bytes**:

```
$ wc -l external_repos/match_hvd/src/model/MATCH/MATCH.py
0 external_repos/match_hvd/src/model/MATCH/MATCH.py
```

The MATCH model class is redacted / absent in the anonymous preview we pulled. We cannot port the class byte-for-byte. This directory contains a **minimal-but-defensible late-fusion MLP reconstruction** of the MATCH classifier, derived from three sources:

1. **Dataset loader contract** (`external_repos/match_hvd/src/model/MATCH/data/HateMM_MATCH.py`)
   - Collator emits `{trans_text_inputs, judge_answers_inputs, hate_answers_inputs, nonhate_answers_inputs, mfcc_fea, vivit_fea, labels}` per batch → the model must accept those six non-label inputs.
2. **Training loop contract** (`external_repos/match_hvd/src/main.py:211-303`)
   - `output = self.model(**inputs)` — `**`-unpacking expects the model's forward signature to match the collator keys.
   - `output["pred"]` shape `[B, num_classes]` (raw logits, not softmax).
   - `output["tsne_tensor"]` (upstream writes t-SNE viz; we emit a placeholder for interface compat).
   - Non-HVD loss path: `F.cross_entropy(pred, labels)` at `main.py:237`.
   - `self.model.name` switch at `main.py:231` — we set `name = "MATCH"` so the loop routes to the non-HVD cross-entropy branch.
3. **Hydra config** (`external_repos/match_hvd/src/config/HateMM_MATCH.yaml`)
   - `text_encoder: bert-base-uncased`, `fea_dim: 128`, `batch_size: 128`, `num_epoch: 50`, `lr: 5e-4`, `weight_decay: 5e-5`, `patience: 8`, `seed: 2025`.

## Architecture (late-fusion MLP)

```
  trans_text_inputs   ─► BERT[CLS] ─► Linear(768→128) ─┐
  judge_answers_inputs ► BERT[CLS] ─► Linear(768→128) ─┤
  hate_answers_inputs  ► BERT[CLS] ─► Linear(768→128) ─┤
  nonhate_answers_inputs► BERT[CLS] ─► Linear(768→128) ─┤
  vivit_fea  ───────── Linear(768→128) ────────────────┤ concat → [B, 6·128]
  mfcc_fea   ───────── Linear(40→128)  ────────────────┘
                                                        │
                                                        ▼
                                                  Linear(6·128 → 128)
                                                  ReLU
                                                  Dropout(0.3)
                                                  Linear(128 → 2)
```

Shared BERT encoder across the 4 text streams (parameter-efficient; matches common late-fusion practice where a single text backbone handles multiple text fields). Per-stream projection heads `text_proj` / `vivit_proj` / `mfcc_proj` → `fea_dim = 128`. The fused 6-stream vector goes through a 2-layer classifier head with ReLU + dropout.

**Architecture decisions where the brief left latitude**:
- **Shared vs per-stream BERT**: I chose one shared BERT encoder across all 4 text streams. Alternative is 4 separate BERTs (upstream-unknown). Shared is ~4× cheaper in params and memory; late-fusion literature typically uses one encoder with stream-specific projections.
- **[CLS] token vs mean-pool**: I use `last_hidden_state[:, 0]` (CLS). Upstream-unknown; CLS is the HF convention for BERT classification.
- **Dropout 0.3**: upstream unspecified; 0.3 is a reasonable default for a 6-stream late-fusion classifier at `fea_dim=128`.
- **Classifier head depth**: 2 linear layers with ReLU + dropout between them (upstream unspecified). Deeper heads didn't add signal in similar setups.

All four are flagged as "reconstruction decisions" and documented inline in `match_model.py`.

## Pipeline

| Step | File | Role |
|---|---|---|
| 1 | `extract_mfcc.py` | `librosa.load → mfcc → mean-pool [40]` → `<root>/fea/fea_audio_mfcc.pt` |
| 2 | `extract_vivit.py` | `google/vivit-b-16x2-kinetics400` on 32 frames → `[768]` CLS → `<root>/fea/fea_frames_32_google-vivit-b.pt` |
| 3 | `dataset_loaders.py` | 4 `<Dataset>_MATCH_Dataset` classes + `MATCH_Collator` (matches upstream `HateMM_MATCH.py:12-92`) |
| 4 | `match_model.py` | Reconstructed `MATCH(nn.Module)` class (this dir is the entire implementation — upstream file is 0 bytes) |
| 5 | `train_match.py` | Training loop (port of `main.py:45-303` minus hydra/wandb/t-SNE) |

### Feature extraction (stages 1 + 2)

- **`extract_mfcc.py`** — CPU. `librosa.load(mp4_path, sr=16000)` then `librosa.feature.mfcc(y, sr=16000, n_mfcc=40)` → mean-pool over time → `[40]`. Zero-vector on audio decode failure (silent / corrupt track). Resume-skip on existing keys.
- **`extract_vivit.py`** — GPU. `VivitImageProcessor(list(32_frames))` → `VivitModel` → `last_hidden_state[:, 0]` → `[768]`. Reads `<root>/frames_32/<vid>/frame_NNN.jpg` (already extracted by `src/match_repro/extract_frames_32.py`). Zero-vector on ViViT forward failure.

### Dataset loaders (stage 3)

Ports upstream `HateMM_MATCH_Dataset` / `MHClipEN_MATCH_Dataset` / `MHClipZH_MATCH_Dataset` with these adaptations:

1. Split membership from our `<root>/splits/{split}_clean.csv`.
2. Annotations from our `data_utils.load_annotations(dataset)`.
3. Labels collapsed via `_collapse_label` (mirrors `eval_generative_predictions.collapse_label`).
4. Stage 2 text streams from `results/match_qwen2vl_7b/<dataset>/{hate,nonhate,judge}.json`.
5. MFCC + ViViT features from `<root>/fea/`.
6. **New `ImpliHateVid_MATCH_Dataset`** — reuses the shared `_BaseMatchDataset` base, only overrides `DATASET_NAME`. No upstream runner exists for ImpliHateVid so this is a fresh loader with the same structure.
7. **Valid split derivation**: our `splits/` only has `train_clean.csv` + `test_clean.csv`. For early stopping we derive `valid` via a **deterministic 80/20 stratified split of `train_clean.csv`**, seed=2025 (matches upstream Hydra `seed: 2025`). `_stratified_train_valid_split` shuffles by `label` and pulls 20% per class for valid. Documented as brief adaptation #6.

The `MATCH_Collator` class is upstream-verbatim in output shape — emits exactly `{vids, trans_text_inputs, judge_answers_inputs, hate_answers_inputs, nonhate_answers_inputs, mfcc_fea, vivit_fea, labels}` (matches `HateMM_MATCH.py:83-92` byte-for-byte).

### Training loop (stage 3 main)

Ports upstream `src/main.py:45-303` with the simplifications from the brief:
- No Hydra — argparse instead
- No wandb — plain `logging` only
- No t-SNE save — model still emits `tsne_tensor` for interface compat
- Single fold type (`default`)
- Loss: `F.cross_entropy(pred, labels)` matching upstream `main.py:237`
- Optimizer: `torch.optim.AdamW(lr=5e-4, weight_decay=5e-5)`
- Scheduler: none (upstream `DummyLR` is a no-op)
- `EarlyStopping(patience=8)` on valid accuracy — matches upstream `main.py:129-131`
- Best model reloaded after training, run on test split, per-video jsonl emitted

## Output

```
results/match_qwen2vl_7b/<dataset>/
    stage3/
        best_model.pth        # best-valid-accuracy checkpoint
        train.log             # (optional) training log
    test_match.jsonl          # ← AUTHORITATIVE MATCH baseline row
```

Schema per line of `test_match.jsonl`:
```json
{
  "video_id": "...",
  "pred": 0 | 1,
  "score": <softmax P(label=1)>,
  "label": 0 | 1
}
```

Compatible with `src/naive_baseline/eval_generative_predictions.py` via the `video_id` + `pred` fields.

Per the 2026-04-15 scope correction, this `test_match.jsonl` file is the **authoritative** MATCH baseline row for the comparison table — replacing the earlier label-free peek jsonl at `test_match_peek.jsonl`.

## Deviations (all documented)

1. **MATCH.py redacted**: upstream file is 0 bytes. Reconstructed late-fusion MLP from dataset loader interface + training loop contract + Hydra config.
2. **Shared BERT encoder** across 4 text streams (upstream architecture unknown).
3. **[CLS] token** for text encoding (upstream architecture unknown, HF standard).
4. **Dropout 0.3** in classifier head (upstream unspecified).
5. **Valid split derivation** via 80/20 stratified on `train_clean.csv`, seed=2025. Upstream has a separate `valid.csv`; our clean splits don't, so we derive deterministically.
6. **MFCC shape `[40]`** via time-mean-pool after `n_mfcc=40`. Upstream's exact tensor shape is unknown but the collator's `torch.stack` requires fixed-shape-per-video, so mean-pooling is the simplest valid choice.
7. **ViViT [CLS] token** as the 768-d video feature (upstream unspecified; CLS is the natural single-vector extraction).
8. **New `ImpliHateVid_MATCH_Dataset`** — no upstream runner; fresh loader with the same interface, binary labels, same text streams (judge / hate / nonhate from stage 2).
9. **No wandb / no Hydra / no t-SNE save / single `default` fold** — brief simplifications.
10. **Stage 2 file naming**: upstream reads `data/<ds>/judge.json` while stage 2c writes `results/match_qwen2vl_7b/<ds>/judge.json` (our earlier rename). The dataset loaders point at our project paths directly.

## CLI (SafetyContradiction env)

```bash
conda activate SafetyContradiction

# Stage 1 — MFCC (CPU, all 4 datasets, both splits)
python src/match_repro/stage3/extract_mfcc.py --all --split both

# Stage 2 — ViViT (GPU)
python src/match_repro/stage3/extract_vivit.py --all --split both

# Stage 3 — train + test on one dataset:
python src/match_repro/stage3/train_match.py --dataset HateMM

# All 4 datasets sequentially:
python src/match_repro/stage3/train_match.py --all
```

## Pre-reqs

1. Stage 2 pipeline must land `hate.json`, `nonhate.json`, `judge.json` in `results/match_qwen2vl_7b/<dataset>/` (currently tracked by the director via job 8438).
2. `<root>/frames_32/<vid>/frame_NNN.jpg` extracted by `src/match_repro/extract_frames_32.py` (already done).
3. `<root>/fea/fea_audio_mfcc.pt` — produced by step 1 above.
4. `<root>/fea/fea_frames_32_google-vivit-b.pt` — produced by step 2 above.
5. `bert-base-uncased` cached in HF hub.
6. `google/vivit-b-16x2-kinetics400` cached in HF hub.

## File list

- `extract_mfcc.py` (~150 lines)
- `extract_vivit.py` (~160 lines)
- `dataset_loaders.py` (~230 lines)
- `match_model.py` (~120 lines)
- `train_match.py` (~290 lines)
- `README.md` (this file)
