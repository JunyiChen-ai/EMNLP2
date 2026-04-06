"""
CMDE: Cross-Modal Description Editor

Core idea: Use audio/frame evidence to gate/edit unit embeddings before classification.
Audio/frame don't appear in the final classifier input — they only edit the text.

Input:
  - units: [B, 5, 768]
  - unit_mask: [B, 5]
  - audio: [B, 768]
  - frame: [B, 768]

Output: logits [B, 2]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalDescriptionEditor(nn.Module):
    def __init__(self, text_dim=768, av_dim=768, hidden=256, num_classes=2,
                 dropout=0.3, edit_reg_weight=0.1):
        super().__init__()
        self.edit_reg_weight = edit_reg_weight

        # AV projection to text space
        self.av_proj = nn.Linear(av_dim * 2, text_dim)

        # Support scoring: bilinear-style
        self.support_proj = nn.Linear(text_dim, text_dim)

        # Edit gate: converts support score to field-level gate
        self.gate_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Field attention pooling
        self.field_attn = nn.Linear(text_dim, 1)

        # Classifier: operates ONLY on edited text
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch):
        units = batch["units"]       # [B, 5, 768]
        mask = batch["unit_mask"]    # [B, 5]
        audio = batch["audio"]       # [B, 768]
        frame = batch["frame"]       # [B, 768]

        B, K, D = units.shape

        # AV evidence
        av = torch.cat([audio, frame], dim=-1)  # [B, 1536]
        av_proj = self.av_proj(av)               # [B, 768]

        # Support scores: how well each field is supported by AV
        units_proj = self.support_proj(units)     # [B, 5, 768]
        av_exp = av_proj.unsqueeze(1)             # [B, 1, 768]
        # Cosine similarity as support score
        support = F.cosine_similarity(units_proj, av_exp, dim=-1)  # [B, 5]

        # Edit gate from support scores
        gate = self.gate_net(support.unsqueeze(-1)).squeeze(-1)  # [B, 5]

        # Apply gate to edit units
        edited_units = units * gate.unsqueeze(-1)  # [B, 5, 768]

        # Attention pooling over edited fields
        attn_scores = self.field_attn(edited_units).squeeze(-1)  # [B, 5]
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)            # [B, 5]
        pooled = (attn_weights.unsqueeze(-1) * edited_units).sum(dim=1)  # [B, 768]

        logits = self.mlp(pooled)  # [B, 2]
        return logits

    def get_edit_gates(self, batch):
        """Return edit gates for analysis."""
        units = batch["units"]
        audio = batch["audio"]
        frame = batch["frame"]

        av = torch.cat([audio, frame], dim=-1)
        av_proj = self.av_proj(av)
        units_proj = self.support_proj(units)
        av_exp = av_proj.unsqueeze(1)
        support = F.cosine_similarity(units_proj, av_exp, dim=-1)
        gate = self.gate_net(support.unsqueeze(-1)).squeeze(-1)
        return gate  # [B, 5]

    def edit_regularization_loss(self, batch):
        """Prevent degenerate editing (all gates -> 0)."""
        gate = self.get_edit_gates(batch)
        # Encourage gates to stay high on average (don't delete everything)
        return self.edit_reg_weight * (1.0 - gate.mean())
