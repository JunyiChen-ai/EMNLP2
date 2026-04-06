"""
RCD: Residual Correlation Distillation

Core idea: Train text teacher → compute residual targets → distill into audio/frame
branches so they learn what text misses, not what text already knows.

Multi-stage:
  Stage 1: Train text teacher
  Stage 2: Compute residual targets (one_hot - text_prob)
  Stage 3: Train audio/frame residual branches (MSE on residuals)
  Inference: text_logit + α*audio_residual + β*frame_residual

Input:
  - text: [B, 768]
  - audio: [B, 768]
  - frame: [B, 768]

Output: logits [B, 2]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RCDTextTeacher(nn.Module):
    """Stage 1: Text-only teacher."""

    def __init__(self, text_dim=768, hidden=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch):
        return self.mlp(batch["text"])  # [B, 2]


class RCDResidualBranch(nn.Module):
    """Audio or frame residual branch: learns to predict residual target."""

    def __init__(self, input_dim=768, hidden=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # [B, 2]


class RCDCombined(nn.Module):
    """Inference-time combination of all branches."""

    def __init__(self, text_teacher, audio_branch, frame_branch, alpha=0.3, beta=0.3):
        super().__init__()
        self.text_teacher = text_teacher
        self.audio_branch = audio_branch
        self.frame_branch = frame_branch
        self.alpha = alpha
        self.beta = beta

        # Freeze all branches at inference
        for p in self.text_teacher.parameters():
            p.requires_grad = False
        for p in self.audio_branch.parameters():
            p.requires_grad = False
        for p in self.frame_branch.parameters():
            p.requires_grad = False

    def forward(self, batch):
        text_logit = self.text_teacher(batch)                    # [B, 2]
        audio_res = self.audio_branch(batch["audio"])            # [B, 2]
        frame_res = self.frame_branch(batch["frame"])            # [B, 2]
        return text_logit + self.alpha * audio_res + self.beta * frame_res  # [B, 2]
