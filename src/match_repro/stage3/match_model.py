"""Reconstructed MATCH classifier.

**Critical upstream gap**: `external_repos/match_hvd/src/model/MATCH/MATCH.py`
is **0 bytes** in the anonymous preview (file redacted). We cannot
port the class byte-for-byte; we reconstruct a minimal-but-defensible
late-fusion MLP from three sources:

1. **Dataset loader contract** (`src/model/MATCH/data/HateMM_MATCH.py`)
   — the collator emits
   `{trans_text_inputs, judge_answers_inputs, hate_answers_inputs,
     nonhate_answers_inputs, mfcc_fea, vivit_fea, labels}`, so the
   model must accept those six non-label inputs.
2. **Training loop contract** (`src/main.py:220-303`):
   - `output = self.model(**inputs)` — `**` unpacking, so the model
     accepts the collator keys as kwargs
   - `output["pred"]` expected (shape `[B, num_classes]`)
   - `output["tsne_tensor"]` expected (upstream saves t-SNE viz; we
     still emit it for interface compatibility, just with the fused
     representation as a placeholder)
   - Non-HVD loss path uses `F.cross_entropy(pred, labels)` —
     `main.py:237` — so `pred` is raw logits, not softmax
   - `self.model.name` is checked at `main.py:231` for the HVD
     special-case; we set `name = "MATCH"` to fall through to the
     default non-HVD branch
3. **Hydra config** (`src/config/HateMM_MATCH.yaml`) — BERT-base as
   the text encoder, `fea_dim=128`, `num_classes=2`.

Reconstruction architecture: late-fusion MLP

    ┌─ trans_text_inputs   ─► BERT[CLS] ─► Linear(768→128) ─┐
    ├─ judge_answers_inputs ► BERT[CLS] ─► Linear(768→128) ─┤
    ├─ hate_answers_inputs  ► BERT[CLS] ─► Linear(768→128) ─┤
    ├─ nonhate_answers_inputs► BERT[CLS] ─► Linear(768→128) ─┤
    ├─ vivit_fea  ───────── Linear(768→128) ────────────────┤ concat → [B, 6·128]
    └─ mfcc_fea   ───────── Linear(40→128)  ────────────────┘
                                                              ↓
                                                         Linear(6·128 → 128)
                                                         ReLU
                                                         Dropout(0.3)
                                                         Linear(128 → 2)

Shared BERT encoder across the 4 text streams (more parameter-
efficient than 4 separate encoders, matches common late-fusion
practice). The projection heads `text_proj`, `vivit_proj`,
`mfcc_proj` are separate per stream.

This reconstruction is documented explicitly in `README.md` as
"reconstructed from upstream dataset loader interface + paper README
+ Hydra config; upstream `MATCH.py` redacted in anonymous preview".
"""

from typing import Any, Dict

import torch
import torch.nn as nn


class MATCH(nn.Module):
    """Reconstructed MATCH late-fusion classifier.

    Attributes:
        name: upstream-matching `self.model.name` that `main.py:231`
              switches on. Setting to "MATCH" routes the training
              loop into the default cross-entropy branch (not the
              "HVD" branch).
        bert: shared text encoder for all 4 text streams.
        text_proj / vivit_proj / mfcc_proj: per-stream projection
              heads → `fea_dim`.
        classifier: 2-layer MLP with ReLU + dropout → `num_classes`
              logits.
    """

    name = "MATCH"

    def __init__(
        self,
        text_encoder: str = "bert-base-uncased",
        fea_dim: int = 128,
        vivit_dim: int = 768,
        mfcc_dim: int = 40,
        num_classes: int = 2,
        dropout: float = 0.3,
        **kwargs: Any,
    ):
        super().__init__()
        from transformers import AutoModel

        self.text_encoder_name = text_encoder
        self.fea_dim = fea_dim
        self.num_classes = num_classes

        self.bert = AutoModel.from_pretrained(text_encoder)
        bert_dim = self.bert.config.hidden_size

        self.text_proj = nn.Linear(bert_dim, fea_dim)
        self.vivit_proj = nn.Linear(vivit_dim, fea_dim)
        self.mfcc_proj = nn.Linear(mfcc_dim, fea_dim)

        # 4 text streams + 1 video + 1 audio = 6 streams of `fea_dim`.
        fused_dim = 6 * fea_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fea_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fea_dim, num_classes),
        )

    def _encode_text(self, inputs) -> torch.Tensor:
        """BERT [CLS]-token encoder → projected to `fea_dim`.

        `inputs` is a `BatchEncoding` from the collator (already on
        the right device after `.to(device)` in the training loop).
        """
        out = self.bert(**inputs)
        cls = out.last_hidden_state[:, 0]  # (B, hidden_size)
        return self.text_proj(cls)

    def forward(
        self,
        trans_text_inputs,
        judge_answers_inputs,
        hate_answers_inputs,
        nonhate_answers_inputs,
        mfcc_fea: torch.Tensor,
        vivit_fea: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        t1 = self._encode_text(trans_text_inputs)
        t2 = self._encode_text(judge_answers_inputs)
        t3 = self._encode_text(hate_answers_inputs)
        t4 = self._encode_text(nonhate_answers_inputs)
        v = self.vivit_proj(vivit_fea.float())
        a = self.mfcc_proj(mfcc_fea.float())
        fused = torch.cat([t1, t2, t3, t4, v, a], dim=-1)  # (B, 6·fea_dim)
        pred = self.classifier(fused)
        # Upstream training loop expects `output["pred"]` +
        # `output["tsne_tensor"]`. We emit the fused representation
        # as the t-SNE placeholder so consuming code that reads this
        # field doesn't break.
        return {"pred": pred, "tsne_tensor": fused.detach().cpu()}
