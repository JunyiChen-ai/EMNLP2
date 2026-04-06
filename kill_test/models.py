"""
Four model variants for the kill test:

1. WholeRationaleMLP: baseline — whole rationale [CLS] embedding → MLP
2. TextUnitAttention: text-only — per-unit embeddings → positive attention → MLP
3. PositiveAVSupport: per-unit embeddings → AV-conditioned positive weights → MLP
4. SignedEvidenceAuditor: per-unit embeddings → AV-conditioned signed scores → accept/reject aggregation → MLP

All models output 2-class logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WholeRationaleMLP(nn.Module):
    """Variant 1: whole rationale embedding → MLP."""

    def __init__(self, text_dim=768, hidden=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch):
        # batch["text"]: [B, 768] — whole rationale embedding
        return self.mlp(batch["text"])


class TextUnitAttention(nn.Module):
    """Variant 2: per-unit text embeddings → learned positive attention → MLP."""

    def __init__(self, text_dim=768, hidden=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.attn_proj = nn.Linear(text_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch):
        # batch["units"]: [B, K, 768]
        # batch["unit_mask"]: [B, K] — 1 for valid, 0 for pad
        units = batch["units"]  # [B, K, 768]
        mask = batch["unit_mask"]  # [B, K]

        scores = self.attn_proj(units).squeeze(-1)  # [B, K]
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)  # [B, K], positive

        z = (weights.unsqueeze(-1) * units).sum(dim=1)  # [B, 768]
        return self.mlp(z)

    def get_weights(self, batch):
        """Return attention weights for analysis."""
        units = batch["units"]
        mask = batch["unit_mask"]
        scores = self.attn_proj(units).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        return F.softmax(scores, dim=-1)


class PositiveAVSupport(nn.Module):
    """Variant 3: per-unit text + AV → positive support weights → MLP."""

    def __init__(
        self, text_dim=768, av_dim=768, hidden=256, num_classes=2, dropout=0.3
    ):
        super().__init__()
        self.av_proj = nn.Linear(av_dim * 2, hidden)  # audio + frame concat
        self.support_head = nn.Linear(text_dim + hidden, 1)
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch):
        units = batch["units"]  # [B, K, 768]
        mask = batch["unit_mask"]  # [B, K]
        av = torch.cat([batch["audio"], batch["frame"]], dim=-1)  # [B, 1536]
        g = F.relu(self.av_proj(av))  # [B, hidden]

        B, K, D = units.shape
        g_exp = g.unsqueeze(1).expand(-1, K, -1)  # [B, K, hidden]
        combined = torch.cat([units, g_exp], dim=-1)  # [B, K, 768+hidden]
        scores = self.support_head(combined).squeeze(-1)  # [B, K]
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)  # [B, K], positive

        z = (weights.unsqueeze(-1) * units).sum(dim=1)  # [B, 768]
        return self.mlp(z)

    def get_weights(self, batch):
        units = batch["units"]
        mask = batch["unit_mask"]
        av = torch.cat([batch["audio"], batch["frame"]], dim=-1)
        g = F.relu(self.av_proj(av))
        B, K, D = units.shape
        g_exp = g.unsqueeze(1).expand(-1, K, -1)
        combined = torch.cat([units, g_exp], dim=-1)
        scores = self.support_head(combined).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        return F.softmax(scores, dim=-1)


class SignedEvidenceAuditor(nn.Module):
    """Variant 4: per-unit text + AV → signed audit scores → accept/reject aggregation → MLP.

    Key difference from Variants 2-3: audit scores are in [-1, +1] via tanh.
    Positive scores → accepted evidence, negative → rejected evidence.
    We aggregate accepted and rejected evidence separately.
    """

    def __init__(
        self, text_dim=768, av_dim=768, hidden=256, num_classes=2, dropout=0.3
    ):
        super().__init__()
        self.av_proj = nn.Linear(av_dim * 2, hidden)
        self.audit_head = nn.Linear(text_dim + hidden, 1)
        # Classifier takes [z_accept; z_reject; z_diff]
        self.mlp = nn.Sequential(
            nn.Linear(text_dim * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch):
        units = batch["units"]  # [B, K, 768]
        mask = batch["unit_mask"]  # [B, K]
        av = torch.cat([batch["audio"], batch["frame"]], dim=-1)  # [B, 1536]
        g = F.relu(self.av_proj(av))  # [B, hidden]

        B, K, D = units.shape
        g_exp = g.unsqueeze(1).expand(-1, K, -1)  # [B, K, hidden]
        combined = torch.cat([units, g_exp], dim=-1)  # [B, K, 768+hidden]

        # Signed audit scores
        raw = self.audit_head(combined).squeeze(-1)  # [B, K]
        s = torch.tanh(raw)  # [B, K], in [-1, 1]
        s = s * mask  # zero out padded units

        # Accept aggregation: weighted by positive part
        pos = F.relu(s)  # [B, K]
        pos_sum = pos.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        alpha_pos = pos / pos_sum  # [B, K], normalized positive weights
        z_accept = (alpha_pos.unsqueeze(-1) * units).sum(dim=1)  # [B, 768]

        # Reject aggregation: weighted by negative part (magnitude)
        neg = F.relu(-s)  # [B, K]
        neg_sum = neg.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        alpha_neg = neg / neg_sum  # [B, K], normalized negative weights
        z_reject = (alpha_neg.unsqueeze(-1) * units).sum(dim=1)  # [B, 768]

        z_diff = z_accept - z_reject  # [B, 768]
        z = torch.cat([z_accept, z_reject, z_diff], dim=-1)  # [B, 768*3]
        return self.mlp(z)

    def get_audit_scores(self, batch):
        """Return signed audit scores for analysis."""
        units = batch["units"]
        mask = batch["unit_mask"]
        av = torch.cat([batch["audio"], batch["frame"]], dim=-1)
        g = F.relu(self.av_proj(av))
        B, K, D = units.shape
        g_exp = g.unsqueeze(1).expand(-1, K, -1)
        combined = torch.cat([units, g_exp], dim=-1)
        raw = self.audit_head(combined).squeeze(-1)
        s = torch.tanh(raw) * mask
        return s

    def get_audit_stats(self, batch):
        """Return accept/ignore/reject fractions for analysis."""
        s = self.get_audit_scores(batch)  # [B, K]
        mask = batch["unit_mask"]
        valid = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1]
        accept = ((s > 0.2) & (mask == 1)).float().sum(dim=-1) / valid.squeeze()
        ignore = ((s.abs() <= 0.2) & (mask == 1)).float().sum(dim=-1) / valid.squeeze()
        reject = ((s < -0.2) & (mask == 1)).float().sum(dim=-1) / valid.squeeze()
        return {
            "accept_frac": accept.mean().item(),
            "ignore_frac": ignore.mean().item(),
            "reject_frac": reject.mean().item(),
            "mean_score": s[mask == 1].mean().item(),
            "std_score": s[mask == 1].std().item(),
        }
