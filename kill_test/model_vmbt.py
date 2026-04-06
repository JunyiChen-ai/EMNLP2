"""
VMBT: Variance-Minimized Boundary Training

Core idea: Use EMA teacher + feature distribution alignment + hard-sample boosting
to stabilize decision boundaries and reduce seed variance.

Input:
  - text: [B, 768]
  - audio: [B, 768]
  - frame: [B, 768]

Output: logits [B, 2], hidden [B, 256] (for alignment loss)

Training requires a separate VMBTTrainer that manages the EMA teacher.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class VMBTModel(nn.Module):
    """Simple multimodal backbone: [text; audio; frame] -> hidden -> logits."""

    def __init__(self, text_dim=768, av_dim=768, hidden=256, num_classes=2, dropout=0.3):
        super().__init__()
        input_dim = text_dim + av_dim * 2  # 768 + 768 + 768 = 2304
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, batch, return_hidden=False):
        x = torch.cat([batch["text"], batch["audio"], batch["frame"]], dim=-1)  # [B, 2304]
        h = self.encoder(x)     # [B, 256]
        logits = self.classifier(h)  # [B, 2]
        if return_hidden:
            return logits, h
        return logits


class VMBTTrainer:
    """Manages EMA teacher, alignment loss, and hard-sample boosting."""

    def __init__(self, model, ema_decay=0.999, lambda_fda=0.1, device="cuda"):
        self.model = model
        self.device = device
        self.ema_decay = ema_decay
        self.lambda_fda = lambda_fda

        # Create EMA teacher as a deep copy
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        for p_s, p_t in zip(self.model.parameters(), self.teacher.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_s.data, alpha=1 - self.ema_decay)

    def compute_alignment_loss(self, h_student, h_teacher, labels):
        """Class-conditional MMD between student and teacher feature distributions."""
        loss = torch.tensor(0.0, device=self.device)
        for cls in [0, 1]:
            mask = labels == cls
            if mask.sum() < 2:
                continue
            hs = h_student[mask]  # [N_cls, 256]
            ht = h_teacher[mask]  # [N_cls, 256]
            # Simple MMD: ||mean(student) - mean(teacher)||^2
            loss = loss + (hs.mean(0) - ht.mean(0)).pow(2).sum()
        return loss

    def compute_sample_weights(self, logits, labels):
        """Hard-sample boosting: low-confidence samples get higher weight."""
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            confidence = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
            weights = 1.0 / (confidence + 1e-6)
            weights = weights / weights.mean()  # normalize
        return weights

    def training_step(self, batch, criterion):
        """One training step with VMBT losses. Returns total loss."""
        batch_dev = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Student forward
        logits, h_student = self.model(batch_dev, return_hidden=True)

        # Teacher forward (no grad)
        with torch.no_grad():
            _, h_teacher = self.teacher(batch_dev, return_hidden=True)

        labels = batch_dev["label"]

        # Hard-sample weighted CE
        sample_weights = self.compute_sample_weights(logits, labels)
        ce_loss = F.cross_entropy(logits, labels, weight=criterion.weight, reduction='none')
        weighted_ce = (sample_weights * ce_loss).mean()

        # Alignment loss
        align_loss = self.compute_alignment_loss(h_student, h_teacher, labels)

        total_loss = weighted_ce + self.lambda_fda * align_loss

        return total_loss, logits

    def step_ema(self):
        self.update_teacher()
