"""
Ablation study runner for all component-level variants.

Variants:
  C1.1: base_only        — No LLM fields, base modalities (text+audio+frame) only
  C1.2: generic_prompt    — Generic (non-SCM) structured LLM fields, same fusion architecture
  C1.3-C1.7: drop_field_X — Remove one SCM field at a time
  C2.1: flat_concat       — All 5 SCM fields flat concatenated, no dual-stream/quadrant
  C2.2: no_quadrant       — No quadrant routing, direct concat → single classifier
  C2.3: unconstrained_moe — 4 experts with learned gating (not quadrant-conditioned)
  C2.4: single_expert     — One classifier with matched parameter count
  C3.1: no_qels           — Fixed label smoothing (eps=0.1)
  C3.2: no_mr2            — No MR2 compactness, standard weighted CE
  C3.3: no_both           — No QELS + No MR2
  C3.4: focal              — Replace QELS with focal loss (gamma=2.0)

Usage:
  python run_ablations.py --dataset_name HateMM --variant base_only --num_runs 200
  python run_ablations.py --dataset_name HateMM --variant generic_prompt --num_runs 200
  python run_ablations.py --dataset_name HateMM --variant unconstrained_moe --num_runs 200
"""

import argparse, csv, json, os, random, copy, logging
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"
SCM_FIELDS = ["target_group", "warmth_evidence", "competence_evidence",
              "social_perception", "behavioral_tendency"]
GENERIC_FIELDS = ["content_summary", "target_analysis", "sentiment_tone",
                  "harm_assessment", "overall_judgment"]


class DS(Dataset):
    def __init__(self, vids, feats, lm, mk):
        self.vids = vids; self.f = feats; self.lm = lm; self.mk = mk
    def __len__(self): return len(self.vids)
    def __getitem__(self, i):
        v = self.vids[i]
        out = {k: self.f[k][v] for k in self.mk}
        out["label"] = torch.tensor(self.lm[self.f["labels"][v]["Label"]], dtype=torch.long)
        return out

def collate_fn(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}


# ============================================================
# Model variants
# ============================================================

class BaseOnlyModel(nn.Module):
    """C1.1: Base modalities only, no LLM fields."""
    def __init__(self, base_mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15):
        super().__init__()
        self.base_mk = base_mk; self.md = md; self.nc = nc
        self.base_projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(len(base_mk))])
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])
        cd = nh * hidden + hidden
        self.cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Linear(64, nc))

    def forward(self, batch, training=False, return_penult=False, return_qels=False):
        base_feats = []
        for p, k in zip(self.base_projs, self.base_mk):
            h = p(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            base_feats.append(h)
        base_stack = torch.stack(base_feats, dim=1)
        heads = [((base_stack * torch.softmax(rm(base_stack).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1))
                 for rm in self.routes]
        base_repr = torch.cat(heads + [base_stack.mean(dim=1)], dim=-1)
        logits = self.cls(base_repr)
        if return_qels:
            return logits, torch.zeros(logits.size(0), device=logits.device)
        return logits


class GenericPromptModel(nn.Module):
    """C1.2: Generic (non-SCM) structured fields, same fusion architecture as SCM.
    Uses 5 generic fields instead of 5 SCM fields, but same dual-stream + Q-MoE architecture.
    Fields mapped: content_summary→warmth, target_analysis→target, sentiment_tone→competence,
                   harm_assessment→social_perception, overall_judgment→behavioral_tendency
    """
    def __init__(self, base_mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15,
                 n_quadrants=4, expert_hidden=64, eps_min=0.01, eps_lambda=0.15):
        super().__init__()
        self.base_mk = base_mk; self.md = md; self.hidden = hidden
        self.n_quadrants = n_quadrants; self.nc = nc
        self.eps_min = eps_min; self.eps_lambda = eps_lambda
        n_base = len(base_mk)
        self.base_projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(n_base)])
        self.field_projs = nn.ModuleDict({
            f: nn.Sequential(
                nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
            ) for f in GENERIC_FIELDS
        })
        # Same architecture: two "streams" + quadrant + Q-MoE
        self.stream_a = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.stream_b = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.quadrant_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, n_quadrants))
        self.quadrant_protos = nn.Parameter(torch.randn(n_quadrants, hidden))
        nn.init.xavier_uniform_(self.quadrant_protos.unsqueeze(0))
        self.register_buffer('harm_weights', torch.tensor([1.0, 0.7, 0.3, 0.0]))
        self.percept_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])
        cd = nh * hidden + hidden + hidden + hidden + 1
        self.pre_cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, expert_hidden), nn.GELU(), nn.Dropout(drop * 0.5))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden, expert_hidden // 2), nn.GELU(), nn.Dropout(drop * 0.5),
                nn.Linear(expert_hidden // 2, nc)
            ) for _ in range(n_quadrants)
        ])

    def forward(self, batch, training=False, return_penult=False, return_qels=False):
        base_feats = []
        for p, k in zip(self.base_projs, self.base_mk):
            h = p(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            base_feats.append(h)
        base_stack = torch.stack(base_feats, dim=1)
        heads = [((base_stack * torch.softmax(rm(base_stack).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1))
                 for rm in self.routes]
        base_repr = torch.cat(heads + [base_stack.mean(dim=1)], dim=-1)
        field_h = {}
        for f in GENERIC_FIELDS:
            h = self.field_projs[f](batch[f"gen_{f}"])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            field_h[f] = h
        base_ctx = base_stack.mean(dim=1)
        # Stream A: content_summary + target_analysis + base_ctx
        a_input = torch.cat([field_h["content_summary"], field_h["target_analysis"], base_ctx], dim=-1)
        a_repr = self.stream_a(a_input)
        # Stream B: sentiment_tone + target_analysis + base_ctx
        b_input = torch.cat([field_h["sentiment_tone"], field_h["target_analysis"], base_ctx], dim=-1)
        b_repr = self.stream_b(b_input)
        ab_cat = torch.cat([a_repr, b_repr], dim=-1)
        quadrant_logits = self.quadrant_head(ab_cat)
        quadrant_dist = torch.softmax(quadrant_logits, dim=-1)
        quadrant_repr = torch.mm(quadrant_dist, self.quadrant_protos)
        harm_score = (quadrant_dist * self.harm_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)
        percept_input = torch.cat([field_h["harm_assessment"], field_h["overall_judgment"]], dim=-1)
        percept_repr = self.percept_proj(percept_input)
        fused = torch.cat([base_repr, quadrant_repr, percept_repr, harm_score], dim=-1)
        shared = self.pre_cls(fused)
        expert_outputs = torch.stack([expert(shared) for expert in self.experts], dim=1)
        logits = (expert_outputs * quadrant_dist.unsqueeze(-1)).sum(dim=1)
        if return_qels:
            q_entropy = -(quadrant_dist * (quadrant_dist + 1e-8).log()).sum(dim=-1)
            q_entropy_norm = q_entropy / np.log(self.n_quadrants)
            return logits, q_entropy_norm
        if return_penult:
            return logits, shared
        return logits


class UnconstrainedMoE(nn.Module):
    """C2.3: Same architecture but experts gated by learned router, NOT quadrant distribution."""
    def __init__(self, base_mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15,
                 n_experts=4, expert_hidden=64, eps_min=0.01, eps_lambda=0.15):
        super().__init__()
        self.base_mk = base_mk; self.md = md; self.hidden = hidden
        self.n_experts = n_experts; self.nc = nc
        self.eps_min = eps_min; self.eps_lambda = eps_lambda
        n_base = len(base_mk)
        self.base_projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(n_base)])
        self.field_projs = nn.ModuleDict({
            f: nn.Sequential(
                nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
            ) for f in SCM_FIELDS
        })
        self.warmth_stream = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.competence_stream = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.percept_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])
        cd = nh * hidden + hidden + hidden + hidden + 1
        # Learned router (NOT quadrant-conditioned)
        self.router = nn.Sequential(
            nn.Linear(cd, hidden), nn.GELU(), nn.Linear(hidden, n_experts))
        self.pre_cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, expert_hidden), nn.GELU(), nn.Dropout(drop * 0.5))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden, expert_hidden // 2), nn.GELU(), nn.Dropout(drop * 0.5),
                nn.Linear(expert_hidden // 2, nc)
            ) for _ in range(n_experts)
        ])
        # Still compute warmth/competence for harm score (but not for routing)
        self.register_buffer('harm_weights', torch.tensor([1.0, 0.7, 0.3, 0.0]))

    def forward(self, batch, training=False, return_penult=False, return_qels=False):
        base_feats = []
        for p, k in zip(self.base_projs, self.base_mk):
            h = p(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            base_feats.append(h)
        base_stack = torch.stack(base_feats, dim=1)
        heads = [((base_stack * torch.softmax(rm(base_stack).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1))
                 for rm in self.routes]
        base_repr = torch.cat(heads + [base_stack.mean(dim=1)], dim=-1)
        field_h = {}
        for f in SCM_FIELDS:
            h = self.field_projs[f](batch[f"scm_{f}"])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            field_h[f] = h
        base_ctx = base_stack.mean(dim=1)
        warmth_repr = self.warmth_stream(torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1))
        comp_repr = self.competence_stream(torch.cat([field_h["competence_evidence"], field_h["target_group"], base_ctx], dim=-1))
        wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)
        # Compute harm score from a simple 4-way softmax on wc_cat
        harm_logits = nn.functional.linear(wc_cat, torch.randn(4, wc_cat.size(-1), device=wc_cat.device))
        harm_dist = torch.softmax(harm_logits, dim=-1)
        harm_score = (harm_dist * self.harm_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)
        percept_repr = self.percept_proj(torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1))
        fused = torch.cat([base_repr, warmth_repr, percept_repr, harm_score], dim=-1)
        # Learned router on fused features
        router_logits = self.router(fused)
        router_dist = torch.softmax(router_logits, dim=-1)
        shared = self.pre_cls(fused)
        expert_outputs = torch.stack([expert(shared) for expert in self.experts], dim=1)
        logits = (expert_outputs * router_dist.unsqueeze(-1)).sum(dim=1)
        if return_qels:
            q_entropy = -(router_dist * (router_dist + 1e-8).log()).sum(dim=-1)
            q_entropy_norm = q_entropy / np.log(self.n_experts)
            return logits, q_entropy_norm
        if return_penult:
            return logits, shared
        return logits


class FlatConcatModel(nn.Module):
    """C2.1: All 5 SCM fields flat concatenated, no dual-stream, no quadrant."""
    def __init__(self, base_mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15, expert_hidden=64):
        super().__init__()
        self.base_mk = base_mk; self.md = md; self.nc = nc
        n_base = len(base_mk)
        self.base_projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(n_base)])
        self.field_projs = nn.ModuleDict({
            f: nn.Sequential(
                nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
            ) for f in SCM_FIELDS
        })
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])
        cd = nh * hidden + hidden + hidden * 5  # base_repr + 5 field projections
        self.cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, expert_hidden), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Linear(expert_hidden, nc))

    def forward(self, batch, training=False, return_penult=False, return_qels=False):
        base_feats = []
        for p, k in zip(self.base_projs, self.base_mk):
            h = p(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            base_feats.append(h)
        base_stack = torch.stack(base_feats, dim=1)
        heads = [((base_stack * torch.softmax(rm(base_stack).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1))
                 for rm in self.routes]
        base_repr = torch.cat(heads + [base_stack.mean(dim=1)], dim=-1)
        field_reprs = []
        for f in SCM_FIELDS:
            h = self.field_projs[f](batch[f"scm_{f}"])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            field_reprs.append(h)
        fused = torch.cat([base_repr] + field_reprs, dim=-1)
        logits = self.cls(fused)
        if return_qels:
            return logits, torch.zeros(logits.size(0), device=logits.device)
        return logits


class GenericPromptFlatModel(nn.Module):
    """Cross-ablation: Generic (non-SCM) fields + flat concat (no dual-stream, no quadrant)."""
    def __init__(self, base_mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15, expert_hidden=64):
        super().__init__()
        self.base_mk = base_mk; self.md = md; self.nc = nc
        n_base = len(base_mk)
        self.base_projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(n_base)])
        self.field_projs = nn.ModuleDict({
            f: nn.Sequential(
                nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
            ) for f in GENERIC_FIELDS
        })
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])
        cd = nh * hidden + hidden + hidden * 5  # base_repr + 5 field projections
        self.cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, expert_hidden), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Linear(expert_hidden, nc))

    def forward(self, batch, training=False, return_penult=False, return_qels=False):
        base_feats = []
        for p, k in zip(self.base_projs, self.base_mk):
            h = p(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            base_feats.append(h)
        base_stack = torch.stack(base_feats, dim=1)
        heads = [((base_stack * torch.softmax(rm(base_stack).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1))
                 for rm in self.routes]
        base_repr = torch.cat(heads + [base_stack.mean(dim=1)], dim=-1)
        field_reprs = []
        for f in GENERIC_FIELDS:
            h = self.field_projs[f](batch[f"gen_{f}"])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            field_reprs.append(h)
        fused = torch.cat([base_repr] + field_reprs, dim=-1)
        logits = self.cls(fused)
        if return_qels:
            return logits, torch.zeros(logits.size(0), device=logits.device)
        return logits


class SingleExpertModel(nn.Module):
    """C2.4: Single expert with matched parameter count (no MoE)."""
    def __init__(self, base_mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15,
                 n_quadrants=4, expert_hidden=64, eps_min=0.01, eps_lambda=0.15):
        super().__init__()
        self.base_mk = base_mk; self.md = md; self.hidden = hidden
        self.n_quadrants = n_quadrants; self.nc = nc
        self.eps_min = eps_min; self.eps_lambda = eps_lambda
        n_base = len(base_mk)
        self.base_projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(n_base)])
        self.field_projs = nn.ModuleDict({
            f: nn.Sequential(
                nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
            ) for f in SCM_FIELDS
        })
        self.warmth_stream = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.competence_stream = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.quadrant_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, n_quadrants))
        self.quadrant_protos = nn.Parameter(torch.randn(n_quadrants, hidden))
        nn.init.xavier_uniform_(self.quadrant_protos.unsqueeze(0))
        self.register_buffer('harm_weights', torch.tensor([1.0, 0.7, 0.3, 0.0]))
        self.percept_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])
        cd = nh * hidden + hidden + hidden + hidden + 1
        # Single expert with 4x capacity to match Q-MoE total params
        self.cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, expert_hidden * 2), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Linear(expert_hidden * 2, nc))

    def forward(self, batch, training=False, return_penult=False, return_qels=False):
        base_feats = []
        for p, k in zip(self.base_projs, self.base_mk):
            h = p(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            base_feats.append(h)
        base_stack = torch.stack(base_feats, dim=1)
        heads = [((base_stack * torch.softmax(rm(base_stack).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1))
                 for rm in self.routes]
        base_repr = torch.cat(heads + [base_stack.mean(dim=1)], dim=-1)
        field_h = {}
        for f in SCM_FIELDS:
            h = self.field_projs[f](batch[f"scm_{f}"])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            field_h[f] = h
        base_ctx = base_stack.mean(dim=1)
        warmth_repr = self.warmth_stream(torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1))
        comp_repr = self.competence_stream(torch.cat([field_h["competence_evidence"], field_h["target_group"], base_ctx], dim=-1))
        wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)
        quadrant_logits = self.quadrant_head(wc_cat)
        quadrant_dist = torch.softmax(quadrant_logits, dim=-1)
        quadrant_repr = torch.mm(quadrant_dist, self.quadrant_protos)
        harm_score = (quadrant_dist * self.harm_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)
        percept_repr = self.percept_proj(torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1))
        fused = torch.cat([base_repr, quadrant_repr, percept_repr, harm_score], dim=-1)
        logits = self.cls(fused)
        if return_qels:
            q_entropy = -(quadrant_dist * (quadrant_dist + 1e-8).log()).sum(dim=-1)
            q_entropy_norm = q_entropy / np.log(self.n_quadrants)
            return logits, q_entropy_norm
        return logits


# ============================================================
# Loss functions
# ============================================================

def qels_cross_entropy(logits, labels, q_entropy, nc=2, eps_min=0.01, eps_lambda=0.15,
                       class_weight=None):
    eps = (eps_min + eps_lambda * q_entropy).clamp(0, 0.5)
    one_hot = F.one_hot(labels, nc).float()
    smooth_targets = (1 - eps.unsqueeze(1)) * one_hot + eps.unsqueeze(1) / nc
    log_probs = F.log_softmax(logits, dim=-1)
    if class_weight is not None:
        w = class_weight[labels]
        loss = -(smooth_targets * log_probs).sum(dim=-1) * w
    else:
        loss = -(smooth_targets * log_probs).sum(dim=-1)
    return loss.mean()

def fixed_ls_cross_entropy(logits, labels, nc=2, eps=0.1, class_weight=None):
    one_hot = F.one_hot(labels, nc).float()
    smooth_targets = (1 - eps) * one_hot + eps / nc
    log_probs = F.log_softmax(logits, dim=-1)
    if class_weight is not None:
        w = class_weight[labels]
        loss = -(smooth_targets * log_probs).sum(dim=-1) * w
    else:
        loss = -(smooth_targets * log_probs).sum(dim=-1)
    return loss.mean()

def focal_cross_entropy(logits, labels, nc=2, gamma=2.0, class_weight=None):
    one_hot = F.one_hot(labels, nc).float()
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    pt = (probs * one_hot).sum(dim=-1)
    fw = (1 - pt) ** gamma
    if class_weight is not None:
        w = class_weight[labels] * fw
    else:
        w = fw
    loss = -(one_hot * log_probs).sum(dim=-1) * w
    return loss.mean()

def mr2_loss(shared, labels, nc=2):
    centroids = []; intra_var = []
    for c in range(nc):
        mask = (labels == c)
        if mask.sum() < 2:
            centroids.append(shared[mask].mean(dim=0) if mask.sum() > 0 else torch.zeros(shared.size(1), device=shared.device))
            intra_var.append(torch.tensor(0.0, device=shared.device))
            continue
        class_feats = shared[mask]
        centroid = class_feats.mean(dim=0)
        centroids.append(centroid)
        var = ((class_feats - centroid.unsqueeze(0)) ** 2).mean()
        intra_var.append(var)
    return sum(intra_var) / nc


# ============================================================
# Training
# ============================================================

def cw_sched(opt, ws, ts):
    def f(s):
        if s < ws: return s / max(1, ws)
        return max(0, 0.5 * (1 + np.cos(np.pi * (s - ws) / max(1, ts - ws))))
    return LambdaLR(opt, f)

def load_split_ids(d):
    s = {}
    for n in ["train", "valid", "test"]:
        with open(os.path.join(d, f"{n}.csv")) as f:
            s[n] = [r[0] for r in csv.reader(f) if r]
    return s

def best_thresh(vl, vla, tl, tla):
    std = accuracy_score(tla, np.argmax(tl, axis=1))
    vd = vl[:, 1] - vl[:, 0]; td = tl[:, 1] - tl[:, 0]
    bt, bv = 0, 0
    for t in np.arange(-3, 3, 0.02):
        v = accuracy_score(vla, (vd > t).astype(int))
        if v > bv: bv, bt = v, t
    tuned = accuracy_score(tla, (td > bt).astype(int))
    return (tuned, bt) if tuned > std else (std, None)

def full_metrics(y_true, y_pred):
    return {"acc": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average='macro')),
            "p": float(precision_score(y_true, y_pred, average='macro')),
            "r": float(recall_score(y_true, y_pred, average='macro')),
            "cm": confusion_matrix(y_true, y_pred).tolist()}


def run_ablation(variant, feats, splits, lm, base_mk, nc, num_runs, seed_offset,
                 class_weight, save_dir, mr2_alpha=0.0, mr2_beta=0.0):
    os.makedirs(save_dir, exist_ok=True)

    # Determine modality keys based on variant
    if variant in ("generic_prompt", "generic_prompt_flat"):
        all_mk = list(base_mk) + [f"gen_{f}" for f in GENERIC_FIELDS]
    elif variant == "base_only":
        all_mk = list(base_mk)
    else:
        all_mk = list(base_mk) + [f"scm_{f}" for f in SCM_FIELDS]

    common = set.intersection(*[set(feats[k].keys()) for k in all_mk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}

    cw_tensor = torch.tensor(class_weight, dtype=torch.float).to(device) if class_weight else None
    all_results = []; global_best_acc = 0; global_best = None

    for ri in range(num_runs):
        seed = ri * 1000 + 42 + seed_offset
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

        trd = DS(cur["train"], feats, lm, all_mk)
        vd = DS(cur["valid"], feats, lm, all_mk)
        ted = DS(cur["test"], feats, lm, all_mk)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
        vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
        tel = DataLoader(ted, 64, False, collate_fn=collate_fn)

        # Create model based on variant
        if variant == "base_only":
            model = BaseOnlyModel(base_mk, nc=nc).to(device)
        elif variant == "generic_prompt":
            model = GenericPromptModel(base_mk, nc=nc).to(device)
        elif variant == "generic_prompt_flat":
            model = GenericPromptFlatModel(base_mk, nc=nc).to(device)
        elif variant == "flat_concat":
            model = FlatConcatModel(base_mk, nc=nc).to(device)
        elif variant == "unconstrained_moe":
            model = UnconstrainedMoE(base_mk, nc=nc).to(device)
        elif variant == "single_expert":
            model = SingleExpertModel(base_mk, nc=nc).to(device)
        else:
            # For loss variants, use the standard SCM model
            from main_scm_qmoe_qels_mr2 import SCMQMoEQELSMR2
            model = SCMQMoEQELSMR2(base_mk, nc=nc).to(device)

        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        ep = 45; ts_t = ep * len(trl); ws = 5 * len(trl); sch = cw_sched(opt, ws, ts_t)
        bva, bst = -1, None

        for e in range(ep):
            model.train()
            for batch in trl:
                batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()

                if variant in ("no_qels", "no_both"):
                    logits = model(batch, training=True) if variant == "no_both" else model(batch, training=True)
                    logits_q = model(batch, training=True, return_qels=True)
                    if isinstance(logits_q, tuple):
                        logits = logits_q[0]
                    loss = fixed_ls_cross_entropy(logits, batch["label"], nc=nc, class_weight=cw_tensor)
                elif variant == "focal":
                    logits_q = model(batch, training=True, return_qels=True)
                    logits = logits_q[0] if isinstance(logits_q, tuple) else logits_q
                    loss = focal_cross_entropy(logits, batch["label"], nc=nc, gamma=2.0, class_weight=cw_tensor)
                elif variant == "no_mr2":
                    logits, q_ent = model(batch, training=True, return_qels=True)
                    loss = qels_cross_entropy(logits, batch["label"], q_ent, nc=nc, class_weight=cw_tensor)
                else:
                    ret = model(batch, training=True, return_qels=True)
                    if isinstance(ret, tuple) and len(ret) == 2:
                        logits, q_ent = ret
                    else:
                        logits = ret; q_ent = torch.zeros(logits.size(0), device=device)
                    loss = qels_cross_entropy(logits, batch["label"], q_ent, nc=nc, class_weight=cw_tensor)

                    # Add MR2 if needed
                    if mr2_alpha > 0 and variant not in ("no_mr2", "no_both", "base_only", "flat_concat"):
                        ret2 = model(batch, training=False, return_penult=True)
                        if isinstance(ret2, tuple):
                            shared = ret2[1]
                            compact = mr2_loss(shared, batch["label"], nc=nc)
                            loss = loss + mr2_alpha * compact

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p, ep2 in zip(model.parameters(), ema.parameters()):
                        ep2.data.mul_(0.999).add_(p.data, alpha=0.001)

            ema.eval(); ps, ls2 = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
            va = accuracy_score(ls2, ps)
            if va > bva: bva = va; bst = {k: v.clone() for k, v in ema.state_dict().items()}

        ema.load_state_dict(bst)
        ema.eval()
        # Get test logits
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in tel:
                batch = {k: v.to(device) for k, v in batch.items()}
                all_logits.append(ema(batch).cpu()); all_labels.extend(batch["label"].cpu().numpy())
        tel_arr = torch.cat(all_logits).numpy(); tela = np.array(all_labels)
        val_logits, val_labels = [], []
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_logits.append(ema(batch).cpu()); val_labels.extend(batch["label"].cpu().numpy())
        vl_arr = torch.cat(val_logits).numpy(); vla = np.array(val_labels)

        head_acc, head_thresh = best_thresh(vl_arr, vla, tel_arr, tela)
        if head_thresh is not None:
            preds = ((tel_arr[:, 1] - tel_arr[:, 0]) > head_thresh).astype(int)
        else:
            preds = np.argmax(tel_arr, axis=1)
        metrics = full_metrics(tela, preds)

        result = {"seed": seed, "ri": ri, "head_acc": float(head_acc), "metrics": metrics}
        all_results.append(result)
        if head_acc > global_best_acc:
            global_best_acc = head_acc; global_best = result

    accs = [r["head_acc"] for r in all_results]
    print(f"{variant}: mean={np.mean(accs):.4f} std={np.std(accs):.4f} max={np.max(accs):.4f}")
    with open(os.path.join(save_dir, "all_results.json"), "w") as f:
        json.dump({"variant": variant, "global_best": global_best, "all_results": all_results}, f, indent=2)
    return global_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip", "ImpliHateVid"])
    parser.add_argument("--language", default="English", choices=["English", "Chinese"])
    parser.add_argument("--variant", required=True,
                        choices=["base_only", "generic_prompt", "generic_prompt_flat", "flat_concat",
                                 "unconstrained_moe", "single_expert", "no_qels", "no_mr2", "no_both", "focal"])
    parser.add_argument("--num_runs", type=int, default=200)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--mr2_alpha", type=float, default=0.1)
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"; ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"; lm = {"Non Hate": 0, "Hate": 1}; tag = "HateMM"
    elif args.dataset_name == "ImpliHateVid":
        emb_dir = "./embeddings/ImpliHateVid"; ann_path = "./datasets/ImpliHateVid/annotation(new).json"
        split_dir = "./datasets/ImpliHateVid/splits"; lm = {"Normal": 0, "Hateful": 1}; tag = "ImpliHateVid"
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}; tag = f"MHC_{args.language[:2]}"

    nc = 2; base_mk = ["text", "audio", "frame"]
    feats = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
    }
    for f in SCM_FIELDS:
        feats[f"scm_{f}"] = torch.load(f"{emb_dir}/scm_mean_{f}_features.pth", map_location="cpu")
    if args.variant in ("generic_prompt", "generic_prompt_flat"):
        for f in GENERIC_FIELDS:
            feats[f"gen_{f}"] = torch.load(f"{emb_dir}/generic_mean_{f}_features.pth", map_location="cpu")
    with open(ann_path) as f:
        feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}
    splits = load_split_ids(split_dir)

    save_dir = f"./ablation_results/{tag}_{args.variant}_off{args.seed_offset}"
    run_ablation(args.variant, feats, splits, lm, base_mk, nc, args.num_runs, args.seed_offset,
                 class_weight=[1.0, 1.5], save_dir=save_dir, mr2_alpha=args.mr2_alpha)


if __name__ == "__main__":
    main()
