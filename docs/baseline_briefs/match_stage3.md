# Task brief — MATCH-HVD stage 3 (supervised training, reconstructed)

## Why this brief exists + critical upstream gap

Upstream `external_repos/match_hvd/src/model/MATCH/MATCH.py` is
**0 bytes** in the anonymous preview — the model class file is
redacted / absent. This is confirmed by `wc -l` and by
`core_utils.load_model` failing to import `model.MATCH.model.MATCH`.

We reconstruct a minimal-but-defensible MATCH classifier from
three sources:

1. **Dataset loader** (`external_repos/match_hvd/src/model/MATCH/data/HateMM_MATCH.py`)
   — defines the input/output contract: 4 text streams (transcript,
   judge, hate, nonhate) + 2 feature streams (MFCC audio, ViViT
   video) + binary label.
2. **Training loop** (`src/main.py:211-276`) — calls
   `self.model(**inputs)` and expects `output["pred"]` (logits,
   shape `[B, 2]`) and `output["tsne_tensor"]`. Loss is plain
   `F.cross_entropy` for non-HVD models (line 237).
3. **Hydra config** (`src/config/HateMM_MATCH.yaml`) — BERT-base
   text encoder, `fea_dim=128`, AdamW lr=5e-4 wd=5e-5, 50 epochs,
   batch=128, patience=8 early stopping, DummyLR scheduler.

Our reconstruction is a **late-fusion MLP classifier** over those
inputs. Document the reconstruction explicitly as "reconstructed
from dataset loader + paper README + Hydra config; upstream MATCH.py
redacted in anonymous preview".

## Scope of this brief

**IN scope**:
- MFCC audio feature extractor (librosa)
- ViViT 32-frame feature extractor
  (`google/vivit-b-16x2-kinetics400`)
- Per-dataset dataset loaders (port upstream 3 + new
  `ImpliHateVid_MATCH.py`)
- Reconstructed `MATCH.py` classifier class
- Training loop (port upstream `main.py` minus wandb/hydra
  complexity; use plain argparse)
- Per-dataset training + test-set prediction → our standard jsonl
  schema

**OUT of scope**:
- Multi-fold 5-fold cross-validation (upstream supports but only
  for the fold type; default type is a single `"default"` split)
- Wandb logging (upstream uses it; we skip)
- Hydra configs (we hardcode the values from upstream yamls in our
  training script)
- t-SNE visualization (upstream saves; we skip)

## Where to write

- `src/match_repro/stage3/`
  - `extract_mfcc.py` — audio MFCC extractor (CPU)
  - `extract_vivit.py` — ViViT 32-frame feature extractor (GPU)
  - `dataset_loaders.py` — 4 dataset classes
    (`HateMM_MATCH_Dataset`, `MHClipEN_MATCH_Dataset`,
    `MHClipZH_MATCH_Dataset`, `ImpliHateVid_MATCH_Dataset`) + 1
    collator
  - `match_model.py` — reconstructed `MATCH` classifier class
  - `train_match.py` — training + evaluation driver
  - `README.md` — upstream gap documentation + reconstruction
    rationale

Do NOT edit existing `src/match_repro/{run_match_agents, time_unit_jina_clip, judgement_vllm, finalize_labelfree, extract_frames, extract_frames_32}.py` — those are stage 2 and already approved.

## Part 1 — MFCC extractor

```python
# extract_mfcc.py
# Per-video MFCC audio features via librosa. Output: dict[vid -> tensor]
# Saved to <dataset_root>/fea/fea_audio_mfcc.pt
```

- Use `librosa.load(mp4_path, sr=16000)` — librosa can decode mp4
  audio via soundfile + audioread
- Compute `librosa.feature.mfcc(y, sr=16000, n_mfcc=40)` → shape
  `[40, T]` where T is time frames
- Pad or pool to a fixed shape — **use mean over time**: `x.mean(axis=1)`
  → `[40]` per video. This is the standard late-fusion audio feature
  vector for the MATCH-style classifier; upstream's exact shape is
  unknown but the dataset loader's `mfcc_fea = torch.load(...)[vid]`
  + collator's `torch.stack([item['mfcc_fea'] for item in batch])`
  requires a fixed-shape tensor per video, which mean-pooling gives.
- Output: `torch.save({vid: tensor, ...}, 'fea_audio_mfcc.pt')`
- Per-dataset: runs via `--dataset X --split test+train` (both
  splits)
- CPU only. Resume support via existing dict keys.
- Videos that can't produce audio (rare — silent or corrupt tracks)
  → write a zero tensor `torch.zeros(40)`. Document.

## Part 2 — ViViT feature extractor

```python
# extract_vivit.py
# Per-video ViViT features on 32 frames via google/vivit-b-16x2-kinetics400
# Output: dict[vid -> tensor]
# Saved to <dataset_root>/fea/fea_frames_32_google-vivit-b.pt
```

- Model: `google/vivit-b-16x2-kinetics400` via
  `transformers.VivitModel` + `VivitImageProcessor`
- Input: 32 frames per video from `<root>/frames_32/<vid>/` (already
  extracted by `extract_frames_32.py`)
- Forward pass → `outputs.last_hidden_state[:, 0]` (CLS token
  embedding, shape `[768]`) or mean over all tokens — use the CLS
  token for simplicity
- Output: `torch.save({vid: tensor, ...}, 'fea_frames_32_google-vivit-b.pt')`
- Per-dataset. GPU required (ViViT is ~90M params, bf16 OK).
- Resume support via existing dict keys.

## Part 3 — Dataset loaders

Port upstream `HateMM_MATCH.py`, `MHClipEN_MATCH.py`,
`MHClipZH_MATCH.py` into `dataset_loaders.py` with these
adaptations:

- Read split membership from our `<root>/splits/{split}_clean.csv`
  (not upstream's `data/HateMM/vids/{split}.csv`)
- Read annotations from our `data_utils.load_annotations(dataset)`
  (not upstream's annotation.csv)
- Label mapping via `eval_generative_predictions.collapse_label()`
- Read `hate.json`, `nonhate.json`, `judge.json` from
  `results/match_qwen2vl_7b/<dataset>/` (produced by stage 2)
- Read MFCC + ViViT features from
  `<root>/fea/fea_audio_mfcc.pt` and
  `<root>/fea/fea_frames_32_google-vivit-b.pt` respectively
- **New ImpliHateVid loader**: copy the HateMM_MATCH structure,
  swap paths. Binary labels, no title field.
- Validation split: our clean splits only have train + test. For
  early stopping we need a valid split. **Derive via deterministic
  80/20 stratified split from `train_clean.csv`** (seed=2025, per
  upstream Hydra `seed: 2025`). Engineer implements this in the
  dataset loader's `_get_data` method.

Collator: port upstream's `HateMM_MATCH_Collator` verbatim (reads
`text_encoder` name, tokenizes the 4 text streams, stacks MFCC +
ViViT features, stacks labels).

## Part 4 — Reconstructed MATCH classifier

```python
# match_model.py
class MATCH(nn.Module):
    def __init__(self, text_encoder, fea_dim=128, vivit_dim=768, mfcc_dim=40, num_classes=2, **kwargs):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(text_encoder)
        bert_dim = self.bert.config.hidden_size  # 768 for bert-base
        self.vivit_proj = nn.Linear(vivit_dim, fea_dim)
        self.mfcc_proj = nn.Linear(mfcc_dim, fea_dim)
        self.text_proj = nn.Linear(bert_dim, fea_dim)
        # 6 streams × fea_dim (4 text + 1 video + 1 audio)
        self.classifier = nn.Sequential(
            nn.Linear(6 * fea_dim, fea_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fea_dim, num_classes),
        )

    def forward(self, trans_text_inputs, judge_answers_inputs,
                hate_answers_inputs, nonhate_answers_inputs,
                mfcc_fea, vivit_fea):
        def encode_text(inputs):
            out = self.bert(**inputs)
            cls = out.last_hidden_state[:, 0]  # CLS token
            return self.text_proj(cls)
        t1 = encode_text(trans_text_inputs)
        t2 = encode_text(judge_answers_inputs)
        t3 = encode_text(hate_answers_inputs)
        t4 = encode_text(nonhate_answers_inputs)
        v = self.vivit_proj(vivit_fea.float())
        a = self.mfcc_proj(mfcc_fea.float())
        fused = torch.cat([t1, t2, t3, t4, v, a], dim=-1)  # [B, 6*fea_dim]
        pred = self.classifier(fused)
        return {"pred": pred, "tsne_tensor": fused.detach().cpu()}
```

This is a minimal late-fusion MLP reconstruction. Document
explicitly in the README as "reconstructed from upstream dataset
loader interface + paper README + Hydra config; upstream MATCH.py
redacted".

## Part 5 — Training loop

Port upstream `src/main.py:45-210` (Trainer class) with these
simplifications:
- No Hydra (use argparse: `--dataset`, `--split`, `--text-encoder`,
  `--fea-dim`, `--num-epoch`, `--batch-size`, `--lr`, `--weight-decay`,
  `--patience`, `--seed`)
- No wandb
- No t-SNE save
- Single fold type (`default`)
- Loss: plain `F.cross_entropy(pred, labels)` — matches upstream's
  non-HVD branch (main.py:237)
- Optimizer: `torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-5)`
- Scheduler: none (upstream uses DummyLR which is a no-op)
- Early stopping on valid accuracy, patience 8
- Save best model state to a `best_model.pth` per dataset
- Reload best model and run on test split → per-video predictions

**Output**: `results/match_qwen2vl_7b/<dataset>/test_match.jsonl`
(schema `{video_id, pred, score}` — this is MATCH's AUTHORITATIVE
row per the 2026-04-15 scope correction, replacing the
label-free finalize peek)

## Syntax check

```bash
conda activate SafetyContradiction
python -m py_compile src/match_repro/stage3/extract_mfcc.py
python -m py_compile src/match_repro/stage3/extract_vivit.py
python -m py_compile src/match_repro/stage3/dataset_loaders.py
python -m py_compile src/match_repro/stage3/match_model.py
python -m py_compile src/match_repro/stage3/train_match.py
python -c "import sys; sys.path.insert(0,'src/match_repro/stage3'); import extract_mfcc, extract_vivit, dataset_loaders, match_model, train_match; print('imports ok')"
```

No sbatch, no smoke. Report file list + audit notes + syntax log +
any decisions you made about the reconstruction that I should know
about (especially architecture choices where I left you latitude).

## Ordering for director-side submission

1. `extract_mfcc.py` — CPU sbatch for all 4 datasets × both splits
2. `extract_vivit.py` — GPU sbatch for all 4 datasets × both splits
   (needs `frames_32/` which is now ready)
3. `train_match.py` — GPU sbatch per dataset (4 jobs) using the
   MFCC + ViViT features + stage 2 `hate.json`/`nonhate.json`/`judge.json`

Feature extraction prereqs are independent; MATCH training depends
on both features + stage 2 outputs.

## Prereq status (director-tracked, not engineer)

As of brief-write: stage 2 pipeline is still PD (job 8438). Stage 3
can't run until stage 2 outputs (`hate.json`, `nonhate.json`,
`judge.json`) land in `results/match_qwen2vl_7b/<dataset>/`. The
feature extraction scripts (MFCC, ViViT) don't depend on stage 2
outputs and can run as soon as written.
