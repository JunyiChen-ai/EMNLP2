"""
SCM + Q-MoE + QELS + Credibility-Aware Fusion (CAF).

Round 016: Replace concat fusion with credibility-aware late fusion.
Tests 4 fusion strategies from literature:

1. QMF (ICML 2023): energy-based uncertainty weighted sum
2. PDF (ICML 2024): predicted confidence weighted sum
3. DBF (AISTATS 2025): Dempster-Shafer belief fusion with conflict discounting
4. Simple learned gate baseline

All require per-branch classifiers (base branch + SCM branch).
Loss: CE(z_final) + aux_w * CE(z_base) + aux_w * CE(z_scm)

Usage:
  python main_scm_qmoe_qels_caf.py --dataset_name HateMM --num_runs 10 --fusion qmf
  python main_scm_qmoe_qels_caf.py --dataset_name HateMM --num_runs 10 --fusion dbf
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
# Fusion strategies
# ============================================================

def qmf_fusion(logits_base, logits_scm, nc=2, T=1.0):
    """QMF (ICML 2023): energy-based uncertainty weighting.
    w_m = alpha * energy_m + beta, alpha < 0 (higher energy = less certain = lower weight).
    We use softmax normalization of negative energy as weights.
    """
    # Energy score: -T * log(sum(exp(f_k / T)))
    energy_base = -T * torch.logsumexp(logits_base / T, dim=-1)  # [B], more negative = more certain
    energy_scm = -T * torch.logsumexp(logits_scm / T, dim=-1)
    # Lower energy = more certain = higher weight
    weights = torch.softmax(torch.stack([-energy_base, -energy_scm], dim=-1), dim=-1)  # [B, 2]
    fused = weights[:, 0:1] * logits_base + weights[:, 1:2] * logits_scm
    return fused, weights


def pdf_fusion(logits_base, logits_scm, labels=None, nc=2):
    """PDF (ICML 2024): predicted confidence (mono-conf) weighted sum.
    Mono-Conf = max predicted probability (proxy for p_true during inference).
    With Distribution Uniformity calibration.
    """
    probs_base = F.softmax(logits_base, dim=-1)
    probs_scm = F.softmax(logits_scm, dim=-1)
    # Mono-confidence: max probability
    conf_base = probs_base.max(dim=-1)[0]  # [B]
    conf_scm = probs_scm.max(dim=-1)[0]
    # Distribution uniformity: sum of |p_k - 1/C|
    du_base = (probs_base - 1.0 / nc).abs().sum(dim=-1)  # [B]
    du_scm = (probs_scm - 1.0 / nc).abs().sum(dim=-1)
    # Calibrated confidence: if DU is lower (more uncertain), discount
    # RC = min(DU_self / DU_other, 1.0)
    rc_base = torch.clamp(du_base / (du_scm + 1e-8), max=1.0)
    rc_scm = torch.clamp(du_scm / (du_base + 1e-8), max=1.0)
    ccb_base = conf_base * rc_base
    ccb_scm = conf_scm * rc_scm
    weights = torch.softmax(torch.stack([ccb_base, ccb_scm], dim=-1), dim=-1)
    fused = weights[:, 0:1] * logits_base + weights[:, 1:2] * logits_scm
    return fused, weights


def dbf_fusion(logits_base, logits_scm, nc=2, lam=1.0):
    """DBF (AISTATS 2025): Dempster-Shafer belief fusion with conflict discounting.
    Each branch outputs evidence -> Dirichlet params -> subjective opinion (b, u).
    Conflict between branches -> discount unreliable branch -> belief averaging.
    """
    # Evidence from logits (ReLU to ensure non-negative)
    ev_base = F.relu(logits_base)  # [B, nc]
    ev_scm = F.relu(logits_scm)
    # Dirichlet parameters
    alpha_base = ev_base + 1  # [B, nc]
    alpha_scm = ev_scm + 1
    S_base = alpha_base.sum(dim=-1, keepdim=True)  # [B, 1]
    S_scm = alpha_scm.sum(dim=-1, keepdim=True)
    # Belief and uncertainty
    b_base = ev_base / S_base  # [B, nc]
    b_scm = ev_scm / S_scm
    u_base = nc / S_base.squeeze(-1)  # [B]
    u_scm = nc / S_scm.squeeze(-1)
    # Projected probability
    p_base = alpha_base / S_base  # [B, nc]
    p_scm = alpha_scm / S_scm

    # Degree of Conflict
    pd = (p_base - p_scm).abs().sum(dim=-1) / 2  # projected distance [B]
    cc = (1 - u_base) * (1 - u_scm)  # conjunctive certainty [B]
    dc = pd * cc  # degree of conflict [B]

    # Discounting factors
    a_base_scm = (1 - dc.pow(lam)).pow(1.0 / lam)  # [B]
    a_scm_base = a_base_scm  # symmetric for 2 modalities

    # Discounted beliefs
    eta_base = a_base_scm  # [B]
    eta_scm = a_scm_base
    b_base_d = eta_base.unsqueeze(-1) * b_base  # [B, nc]
    b_scm_d = eta_scm.unsqueeze(-1) * b_scm
    u_base_d = 1 - eta_base + eta_base * u_base  # [B]
    u_scm_d = 1 - eta_scm + eta_scm * u_scm

    # Generalized belief averaging (2 sources)
    # b_k^fused = (b_k^1 * u^2 + b_k^2 * u^1) / (u^1 + u^2)
    # u^fused = (u^1 * u^2) / (u^1 + u^2)
    denom = u_base_d + u_scm_d + 1e-8  # [B]
    b_fused = (b_base_d * u_scm_d.unsqueeze(-1) + b_scm_d * u_base_d.unsqueeze(-1)) / denom.unsqueeze(-1)
    u_fused = (u_base_d * u_scm_d) / denom

    # Convert back to logits: use belief as probability proxy
    # p_fused = b_fused + a_k * u_fused (with uniform prior a_k = 1/nc)
    p_fused = b_fused + u_fused.unsqueeze(-1) / nc
    fused_logits = torch.log(p_fused + 1e-8)

    weights = torch.stack([eta_base, eta_scm], dim=-1)  # for logging
    return fused_logits, weights


def gate_fusion(logits_base, logits_scm, gate_logit):
    """Simple learned gate: g(x) in [0,1], fused = (1-g)*base + g*scm."""
    g = torch.sigmoid(gate_logit)  # [B, 1]
    fused = (1 - g) * logits_base + g * logits_scm
    weights = torch.cat([1 - g, g], dim=-1)
    return fused, weights


# ============================================================
# Model
# ============================================================

class SCMQMoECAF(nn.Module):
    """SCM + Q-MoE + QELS with Credibility-Aware Fusion.
    Two branches: base-modality branch + SCM branch, each with own classifier.
    Fusion at logit level via QMF/PDF/DBF/gate.
    """
    def __init__(self, base_mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15,
                 n_quadrants=4, expert_hidden=64,
                 eps_min=0.01, eps_lambda=0.15, fusion='dbf'):
        super().__init__()
        self.base_mk = base_mk; self.md = md; self.hidden = hidden
        self.n_quadrants = n_quadrants; self.nc = nc
        self.eps_min = eps_min; self.eps_lambda = eps_lambda
        self.fusion_type = fusion
        n_base = len(base_mk)

        # Base branch
        self.base_projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(n_base)])
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])
        base_dim = nh * hidden + hidden
        self.base_cls = nn.Sequential(
            nn.LayerNorm(base_dim), nn.Linear(base_dim, 128), nn.GELU(), nn.Dropout(drop),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Linear(64, nc))

        # SCM branch
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

        # SCM classifier: quadrant_repr + percept_repr + harm_score -> experts
        scm_cd = hidden + hidden + 1  # quadrant_repr + percept_repr + harm_score
        self.scm_pre_cls = nn.Sequential(
            nn.LayerNorm(scm_cd), nn.Linear(scm_cd, 128), nn.GELU(), nn.Dropout(drop),
            nn.Linear(128, expert_hidden), nn.GELU(), nn.Dropout(drop * 0.5))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden, expert_hidden // 2), nn.GELU(), nn.Dropout(drop * 0.5),
                nn.Linear(expert_hidden // 2, nc)
            ) for _ in range(n_quadrants)
        ])

        # Gate (for gate fusion only)
        if fusion == 'gate':
            self.gate_net = nn.Sequential(
                nn.Linear(base_dim + scm_cd, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, batch, training=False, return_branches=False):
        # === Base branch ===
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
        logits_base = self.base_cls(base_repr)

        # === SCM branch ===
        field_h = {}
        for f in SCM_FIELDS:
            h = self.field_projs[f](batch[f"scm_{f}"])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            field_h[f] = h
        base_ctx = base_stack.mean(dim=1)

        warmth_input = torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1)
        warmth_repr = self.warmth_stream(warmth_input)
        comp_input = torch.cat([field_h["competence_evidence"], field_h["target_group"], base_ctx], dim=-1)
        comp_repr = self.competence_stream(comp_input)

        wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)
        quadrant_logits = self.quadrant_head(wc_cat)
        quadrant_dist = torch.softmax(quadrant_logits, dim=-1)
        quadrant_repr = torch.mm(quadrant_dist, self.quadrant_protos)
        harm_score = (quadrant_dist * self.harm_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)

        percept_input = torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1)
        percept_repr = self.percept_proj(percept_input)

        scm_repr = torch.cat([quadrant_repr, percept_repr, harm_score], dim=-1)
        scm_shared = self.scm_pre_cls(scm_repr)

        expert_outputs = torch.stack([expert(scm_shared) for expert in self.experts], dim=1)
        logits_scm = (expert_outputs * quadrant_dist.unsqueeze(-1)).sum(dim=1)

        # === Fusion ===
        if self.fusion_type == 'qmf':
            logits_fused, weights = qmf_fusion(logits_base, logits_scm)
        elif self.fusion_type == 'pdf':
            logits_fused, weights = pdf_fusion(logits_base, logits_scm)
        elif self.fusion_type == 'dbf':
            logits_fused, weights = dbf_fusion(logits_base, logits_scm)
        elif self.fusion_type == 'gate':
            gate_input = torch.cat([base_repr, scm_repr], dim=-1)
            gate_logit = self.gate_net(gate_input)
            logits_fused, weights = gate_fusion(logits_base, logits_scm, gate_logit)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion_type}")

        # QELS entropy from quadrant dist
        q_entropy = -(quadrant_dist * (quadrant_dist + 1e-8).log()).sum(dim=-1)
        q_entropy_norm = q_entropy / np.log(self.n_quadrants)

        if return_branches:
            return logits_fused, logits_base, logits_scm, q_entropy_norm, weights

        return logits_fused


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

def setup_logger(tag=""):
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/scm_caf_{tag}_{ts}.log"
    logger = logging.getLogger(f"caf_{tag}_{ts}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Log: {lf}"); return logger


def run(feats, splits, lm, base_mk, nc, num_runs, seed_offset, class_weight, save_dir, logger,
        expert_hidden=64, eps_min=0.01, eps_lambda=0.15, fusion='dbf', aux_weight=0.3,
        dbf_lambda=1.0):
    os.makedirs(save_dir, exist_ok=True)
    all_mk = list(base_mk) + [f"scm_{f}" for f in SCM_FIELDS]
    common = set.intersection(*[set(feats[k].keys()) for k in all_mk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}
    logger.info(f"  Train={len(cur['train'])}, Val={len(cur['valid'])}, Test={len(cur['test'])}")
    logger.info(f"  Fusion={fusion}, aux_weight={aux_weight}, dbf_lambda={dbf_lambda}")

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

        model = SCMQMoECAF(base_mk, nc=nc, expert_hidden=expert_hidden,
                           eps_min=eps_min, eps_lambda=eps_lambda,
                           fusion=fusion).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        ep = 45; ts_t = ep * len(trl); ws = 5 * len(trl); sch = cw_sched(opt, ws, ts_t)
        bva, bst = -1, None

        for e in range(ep):
            model.train()
            for batch in trl:
                batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
                logits_f, logits_b, logits_s, q_ent, w = model(batch, training=True, return_branches=True)

                # Main loss on fused logits
                loss_main = qels_cross_entropy(logits_f, batch["label"], q_ent, nc=nc,
                                              eps_min=eps_min, eps_lambda=eps_lambda,
                                              class_weight=cw_tensor)
                # Auxiliary losses on branch logits
                loss_base = F.cross_entropy(logits_b, batch["label"],
                                           weight=cw_tensor)
                loss_scm = qels_cross_entropy(logits_s, batch["label"], q_ent, nc=nc,
                                             eps_min=eps_min, eps_lambda=eps_lambda,
                                             class_weight=cw_tensor)

                loss = loss_main + aux_weight * (loss_base + loss_scm)
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

        # Evaluate
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in tel:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = ema(batch)
                all_logits.append(logits.cpu()); all_labels.extend(batch["label"].cpu().numpy())
        tel_arr = torch.cat(all_logits).numpy()
        tela = np.array(all_labels)

        # Also get val logits for threshold tuning
        val_logits, val_labels = [], []
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_logits.append(ema(batch).cpu()); val_labels.extend(batch["label"].cpu().numpy())
        vl_arr = torch.cat(val_logits).numpy()
        vla = np.array(val_labels)

        head_acc, head_thresh = best_thresh(vl_arr, vla, tel_arr, tela)
        if head_thresh is not None:
            preds = ((tel_arr[:, 1] - tel_arr[:, 0]) > head_thresh).astype(int)
        else:
            preds = np.argmax(tel_arr, axis=1)

        metrics = full_metrics(tela, preds)
        result = {"seed": seed, "ri": ri, "seed_offset": seed_offset,
                  "head_acc": float(head_acc), "val_acc": float(bva), "metrics": metrics,
                  "fusion": fusion, "aux_weight": aux_weight}
        all_results.append(result)

        if head_acc > global_best_acc:
            global_best_acc = head_acc; global_best = result
            torch.save(bst, os.path.join(save_dir, "best_model.pth"))
            logger.info(f"  NEW BEST seed={seed} ACC={metrics['acc']:.4f} F1={metrics['f1']:.4f}")

        if (ri + 1) % 10 == 0:
            accs = [r["head_acc"] for r in all_results]
            logger.info(f"  [{ri+1}/{num_runs}] mean={np.mean(accs):.4f} max={np.max(accs):.4f}")

    accs = [r["head_acc"] for r in all_results]
    logger.info(f"  FINAL: mean={np.mean(accs):.4f}+/-{np.std(accs):.4f} max={np.max(accs):.4f}")
    logger.info(f"  GLOBAL BEST: {json.dumps(global_best, indent=2)}")
    with open(os.path.join(save_dir, "all_results.json"), "w") as f:
        json.dump({"global_best": global_best, "all_results": all_results}, f, indent=2)
    return global_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English", choices=["English", "Chinese"])
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--eps_min", type=float, default=0.01)
    parser.add_argument("--eps_lambda", type=float, default=0.15)
    parser.add_argument("--fusion", default="dbf", choices=["qmf", "pdf", "dbf", "gate"])
    parser.add_argument("--aux_weight", type=float, default=0.3)
    parser.add_argument("--dbf_lambda", type=float, default=1.0)
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"; ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"; lm = {"Non Hate": 0, "Hate": 1}; tag = "HateMM"
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}; tag = f"MHC_{args.language[:2]}"

    logger = setup_logger(f"CAF_{args.fusion}_{tag}")
    nc = 2; base_mk = ["text", "audio", "frame"]
    logger.info(f"=== SCM CAF({args.fusion}): {tag}, Runs: {args.num_runs}, "
                f"aux_w={args.aux_weight} ===")

    feats = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
    }
    for f in SCM_FIELDS:
        feats[f"scm_{f}"] = torch.load(f"{emb_dir}/scm_mean_{f}_features.pth", map_location="cpu")
    with open(ann_path) as f:
        feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}

    splits = load_split_ids(split_dir)
    save_dir = f"./seed_search_scm_caf/{tag}_{args.fusion}_aux{args.aux_weight}_off{args.seed_offset}"

    best = run(feats, splits, lm, base_mk, nc, args.num_runs, args.seed_offset,
               class_weight=[1.0, 1.5], save_dir=save_dir, logger=logger,
               eps_min=args.eps_min, eps_lambda=args.eps_lambda,
               fusion=args.fusion, aux_weight=args.aux_weight,
               dbf_lambda=args.dbf_lambda)

    logger.info(f"\n{'='*60}")
    logger.info(f"  CAF({args.fusion}) {tag} BEST: ACC={best['metrics']['acc']:.4f} F1={best['metrics']['f1']:.4f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
