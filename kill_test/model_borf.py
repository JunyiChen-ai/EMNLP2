"""
BORF: Boundary-Only Residual Fusion

Core idea: Train text-only classifier first, freeze it. Then train audio/frame
residual branches that only activate when text is uncertain.

Stage 1: Train text MLP → freeze
Stage 2: Train residual branches + ambiguity gate

Input:
  - text: [B, 768]
  - audio: [B, 768]
  - frame: [B, 768]

Output: logits [B, 2]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BORFTextBase(nn.Module):
    """Stage 1: Text-only base classifier."""

    def __init__(self, text_dim=768, hidden=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, batch):
        h = self.hidden_layer(batch["text"])  # [B, 256]
        logits = self.classifier(h)           # [B, 2]
        return logits

    def get_hidden(self, batch):
        return self.hidden_layer(batch["text"])  # [B, 256]


class BORFResidual(nn.Module):
    """Stage 2: Residual branches + ambiguity gate on top of frozen text base."""

    def __init__(self, text_hidden=256, av_dim=768, num_classes=2, tau=0.7, beta=10.0):
        super().__init__()
        self.tau = tau
        self.beta = beta

        # Audio residual: text_hidden × audio_proj -> residual logit
        self.audio_proj = nn.Linear(av_dim, text_hidden)
        self.audio_residual = nn.Linear(text_hidden, num_classes)

        # Frame residual: text_hidden × frame_proj -> residual logit
        self.frame_proj = nn.Linear(av_dim, text_hidden)
        self.frame_residual = nn.Linear(text_hidden, num_classes)

    def forward(self, h_text, base_logit, audio, frame):
        """
        Args:
            h_text: [B, 256] frozen text hidden
            base_logit: [B, 2] frozen base logits
            audio: [B, 768]
            frame: [B, 768]
        Returns:
            final_logit: [B, 2]
        """
        # Audio residual
        h_audio = self.audio_proj(audio)            # [B, 256]
        interaction_ta = h_text * h_audio            # [B, 256] elementwise
        res_audio = self.audio_residual(interaction_ta)  # [B, 2]

        # Frame residual
        h_frame = self.frame_proj(frame)             # [B, 256]
        interaction_tf = h_text * h_frame            # [B, 256]
        res_frame = self.frame_residual(interaction_tf)  # [B, 2]

        # Ambiguity gate: activate residuals only when text is uncertain
        with torch.no_grad():
            text_conf = F.softmax(base_logit, dim=-1).max(dim=1).values  # [B]
        gate = torch.sigmoid(-self.beta * (text_conf - self.tau))  # [B]
        gate = gate.unsqueeze(1)  # [B, 1]

        final_logit = base_logit + gate * (res_audio + res_frame)  # [B, 2]
        return final_logit


class BORFFullModel(nn.Module):
    """Wrapper for end-to-end inference after both stages are trained."""

    def __init__(self, text_base, residual_module):
        super().__init__()
        self.text_base = text_base
        self.residual = residual_module
        # Freeze text base
        for p in self.text_base.parameters():
            p.requires_grad = False

    def forward(self, batch):
        with torch.no_grad():
            h_text = self.text_base.get_hidden(batch)  # [B, 256]
            base_logit = self.text_base.classifier(h_text)  # [B, 2]

        return self.residual(h_text, base_logit, batch["audio"], batch["frame"])
