"""
GTT: Grounded Token Trust

Core idea: Use audio/frame to score which MLLM-generated text tokens are trustworthy.
Soft-mask unsupported tokens before pooling.

Input:
  - unit_tokens: [B, 5, T, 768]  (token-level BERT hidden states)
  - unit_token_mask: [B, 5, T]   (attention mask)
  - audio: [B, 768]
  - frame: [B, 768]

Output: logits [B, 2]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundedTokenTrust(nn.Module):
    def __init__(self, text_dim=768, av_dim=768, hidden=256, num_classes=2, dropout=0.3):
        super().__init__()
        # Project AV to text space
        self.av_proj = nn.Linear(av_dim * 2, text_dim)

        # Token trust scoring: compare each token with AV evidence
        self.trust_head = nn.Sequential(
            nn.Linear(text_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Field-level attention pooling over 5 fields
        self.field_attn = nn.Linear(text_dim, 1)

        # Classifier
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch):
        tokens = batch["unit_tokens"].float()      # [B, 5, T, 768]
        tok_mask = batch["unit_token_mask"].float()  # [B, 5, T]
        audio = batch["audio"]                       # [B, 768]
        frame = batch["frame"]                       # [B, 768]

        B, K, T, D = tokens.shape

        # AV projection
        av = torch.cat([audio, frame], dim=-1)  # [B, 1536]
        av_proj = self.av_proj(av)               # [B, 768]

        # Expand AV to match token dims: [B, 1, 1, 768] -> [B, K, T, 768]
        av_exp = av_proj.unsqueeze(1).unsqueeze(2).expand(-1, K, T, -1)

        # Trust scoring: concat token with AV evidence
        trust_input = torch.cat([tokens, av_exp], dim=-1)  # [B, K, T, 1536]
        trust_scores = torch.sigmoid(self.trust_head(trust_input).squeeze(-1))  # [B, K, T]

        # Mask out padding
        trust_scores = trust_scores * tok_mask  # [B, K, T]

        # Trust-weighted pooling per field
        weighted = tokens * trust_scores.unsqueeze(-1)  # [B, K, T, 768]
        denom = trust_scores.sum(dim=2, keepdim=True).clamp(min=1e-8)  # [B, K, 1]
        field_embs = weighted.sum(dim=2) / denom  # [B, K, 768]

        # Attention pooling over 5 fields
        field_scores = self.field_attn(field_embs).squeeze(-1)  # [B, K]
        field_weights = F.softmax(field_scores, dim=-1)          # [B, K]
        pooled = (field_weights.unsqueeze(-1) * field_embs).sum(dim=1)  # [B, 768]

        return self.mlp(pooled)  # [B, 2]

    def get_trust_scores(self, batch):
        """Return trust scores for analysis."""
        tokens = batch["unit_tokens"].float()
        tok_mask = batch["unit_token_mask"].float()
        audio = batch["audio"]
        frame = batch["frame"]

        B, K, T, D = tokens.shape
        av = torch.cat([audio, frame], dim=-1)
        av_proj = self.av_proj(av)
        av_exp = av_proj.unsqueeze(1).unsqueeze(2).expand(-1, K, T, -1)
        trust_input = torch.cat([tokens, av_exp], dim=-1)
        trust_scores = torch.sigmoid(self.trust_head(trust_input).squeeze(-1))
        return trust_scores * tok_mask
