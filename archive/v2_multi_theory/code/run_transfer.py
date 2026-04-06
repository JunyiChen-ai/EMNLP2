"""
Cross-dataset transfer experiment: HateMM <-> MHClip-EN.

Train on dataset A (train+val for model selection), evaluate on dataset B (test).
5 seeds per direction. Reports ACC and F1.
"""

import argparse, csv, json, os, random, copy, logging
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"
SCM_FIELDS = ["target_group", "warmth_evidence", "competence_evidence",
              "social_perception", "behavioral_tendency"]

# ---------------------------------------------------------------------------
# Reuse model and helpers from main script
# ---------------------------------------------------------------------------

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


class SCMQMoEQELSMR2(nn.Module):
    def __init__(self, base_mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15,
                 n_quadrants=4, expert_hidden=64, load_balance_weight=0.01,
                 eps_min=0.01, eps_lambda=0.15):
        super().__init__()
        self.base_mk = base_mk; self.md = md; self.hidden = hidden
        self.n_quadrants = n_quadrants; self.nc = nc
        self.load_balance_weight = load_balance_weight
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
        self.percept_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])
        cd = nh * hidden + hidden + hidden + hidden
        self.pre_cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, expert_hidden), nn.GELU(), nn.Dropout(drop * 0.5))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden, expert_hidden // 2), nn.GELU(), nn.Dropout(drop * 0.5),
                nn.Linear(expert_hidden // 2, nc)
            ) for _ in range(n_quadrants)
        ])

    def forward(self, batch, training=False, return_penult=False, return_qels=False,
                return_all=False):
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
        warmth_input = torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1)
        warmth_repr = self.warmth_stream(warmth_input)
        comp_input = torch.cat([field_h["competence_evidence"], field_h["target_group"], base_ctx], dim=-1)
        comp_repr = self.competence_stream(comp_input)
        wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)
        quadrant_logits = self.quadrant_head(wc_cat)
        quadrant_dist = torch.softmax(quadrant_logits, dim=-1)
        quadrant_repr = torch.mm(quadrant_dist, self.quadrant_protos)
        percept_input = torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1)
        percept_repr = self.percept_proj(percept_input)
        fused = torch.cat([base_repr, quadrant_repr, percept_repr], dim=-1)
        shared = self.pre_cls(fused)
        expert_outputs = torch.stack([expert(shared) for expert in self.experts], dim=1)
        logits = (expert_outputs * quadrant_dist.unsqueeze(-1)).sum(dim=1)
        if return_all:
            q_entropy = -(quadrant_dist * (quadrant_dist + 1e-8).log()).sum(dim=-1)
            q_entropy_norm = q_entropy / np.log(self.n_quadrants)
            return logits, q_entropy_norm, shared, quadrant_dist
        if return_qels:
            q_entropy = -(quadrant_dist * (quadrant_dist + 1e-8).log()).sum(dim=-1)
            q_entropy_norm = q_entropy / np.log(self.n_quadrants)
            return logits, q_entropy_norm
        if return_penult:
            return logits, shared
        return logits


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


def mr2_loss(shared, labels, nc=2):
    centroids, intra_var = [], []
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
    compact_loss = sum(intra_var) / nc
    if len(centroids) == 2:
        sep_loss = -F.pairwise_distance(centroids[0].unsqueeze(0), centroids[1].unsqueeze(0)).mean()
    else:
        sep_loss = torch.tensor(0.0, device=shared.device)
    return compact_loss, sep_loss


def load_balance_loss(quadrant_dist):
    usage = quadrant_dist.mean(dim=0)
    cv2 = usage.var() / (usage.mean() ** 2 + 1e-8)
    return cv2


def cw(opt, ws, ts):
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


def load_features(emb_dir, ann_path, label_map):
    """Load all features for a dataset."""
    base_mk = ["text", "audio", "frame"]
    feats = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
    }
    for field in SCM_FIELDS:
        feats[f"scm_{field}"] = torch.load(f"{emb_dir}/scm_mean_{field}_features.pth", map_location="cpu")
    with open(ann_path) as f:
        feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}
    return feats


def get_valid_ids(feats, splits, base_mk):
    """Get video IDs that have all features available."""
    all_mk = list(base_mk) + [f"scm_{f}" for f in SCM_FIELDS]
    common = set.intersection(*[set(feats[k].keys()) for k in all_mk]) & set(feats["labels"].keys())
    return {s: [v for v in splits[s] if v in common] for s in splits}


def train_one_seed(seed, train_feats, train_ids, train_lm, eval_feats, eval_ids, eval_lm,
                   base_mk, nc, mr2_alpha, mr2_beta, lb_weight, class_weight, logger):
    """Train on train_feats (train+val splits), evaluate on eval_feats (test split)."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    all_mk = list(base_mk) + [f"scm_{f}" for f in SCM_FIELDS]
    cw_tensor = torch.tensor(class_weight, dtype=torch.float).to(device) if class_weight else None

    # Build datasets
    trd = DS(train_ids["train"], train_feats, train_lm, all_mk)
    vd = DS(train_ids["valid"], train_feats, train_lm, all_mk)
    # Target test set
    ted = DS(eval_ids["test"], eval_feats, eval_lm, all_mk)

    trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
    vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
    tel = DataLoader(ted, 64, False, collate_fn=collate_fn)

    model = SCMQMoEQELSMR2(base_mk, nc=nc, expert_hidden=64).to(device)
    ema = copy.deepcopy(model)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    ep = 45; ts_t = ep * len(trl); ws = 5 * len(trl); sch = cw(opt, ws, ts_t)
    bva, bst = -1, None

    for e in range(ep):
        model.train()
        for batch in trl:
            batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
            logits, q_entropy, shared, q_dist = model(batch, training=True, return_all=True)
            ce_loss = qels_cross_entropy(logits, batch["label"], q_entropy, nc=nc,
                                         class_weight=cw_tensor)
            compact, sep = mr2_loss(shared, batch["label"], nc=nc)
            lb_loss = load_balance_loss(q_dist)
            loss = ce_loss + mr2_alpha * compact + mr2_beta * sep + lb_weight * lb_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            with torch.no_grad():
                for p, ep2 in zip(model.parameters(), ema.parameters()):
                    ep2.data.mul_(0.999).add_(p.data, alpha=0.001)

        # Val on source dataset for model selection
        ema.eval(); ps, ls2 = [], []
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(device) for k, v in batch.items()}
                ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
        va = accuracy_score(ls2, ps)
        if va > bva:
            bva = va; bst = {k: v.clone() for k, v in ema.state_dict().items()}

    # Evaluate best model on target test set
    ema.load_state_dict(bst)
    ema.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tel:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = ema(batch)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    logger.info(f"  seed={seed}: val_acc={bva:.4f} | transfer ACC={acc:.4f} F1={f1:.4f}")
    return {"seed": seed, "val_acc": float(bva), "transfer_acc": float(acc), "transfer_f1": float(f1)}


def main():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"transfer_{ts}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    lf = f"./logs/transfer_{ts}.log"
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Log: {lf}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_mk = ["text", "audio", "frame"]
    nc = 2
    seeds = [42, 1042, 2042, 3042, 4042]

    # --- Load HateMM ---
    logger.info("Loading HateMM features...")
    hmm_feats = load_features(
        f"{base_dir}/embeddings/HateMM",
        f"{base_dir}/datasets/HateMM/annotation(new).json",
        {"Non Hate": 0, "Hate": 1}
    )
    hmm_splits = load_split_ids(f"{base_dir}/datasets/HateMM/splits")
    hmm_lm = {"Non Hate": 0, "Hate": 1}
    hmm_ids = get_valid_ids(hmm_feats, hmm_splits, base_mk)
    logger.info(f"  HateMM: train={len(hmm_ids['train'])}, val={len(hmm_ids['valid'])}, test={len(hmm_ids['test'])}")

    # --- Load MHC-EN ---
    logger.info("Loading MHClip-EN features...")
    mhc_feats = load_features(
        f"{base_dir}/embeddings/Multihateclip/English",
        f"{base_dir}/datasets/Multihateclip/English/annotation(new).json",
        {"Normal": 0, "Offensive": 1, "Hateful": 1}
    )
    mhc_splits = load_split_ids(f"{base_dir}/datasets/Multihateclip/English/splits")
    mhc_lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}
    mhc_ids = get_valid_ids(mhc_feats, mhc_splits, base_mk)
    logger.info(f"  MHC-EN: train={len(mhc_ids['train'])}, val={len(mhc_ids['valid'])}, test={len(mhc_ids['test'])}")

    save_dir = f"{base_dir}/transfer_results"
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}

    # --- Direction 1: Train HateMM -> Test MHC-EN ---
    logger.info("=" * 60)
    logger.info("Direction 1: Train on HateMM -> Test on MHClip-EN")
    logger.info("  Hyperparams: alpha=0.3, beta=0.0 (HateMM best)")
    logger.info("=" * 60)
    d1_results = []
    for seed in seeds:
        r = train_one_seed(
            seed=seed,
            train_feats=hmm_feats, train_ids=hmm_ids, train_lm=hmm_lm,
            eval_feats=mhc_feats, eval_ids=mhc_ids, eval_lm=mhc_lm,
            base_mk=base_mk, nc=nc,
            mr2_alpha=0.3, mr2_beta=0.0, lb_weight=0.01,
            class_weight=[1.0, 1.5], logger=logger
        )
        d1_results.append(r)
    all_results["HateMM_to_MHC_EN"] = d1_results

    d1_accs = [r["transfer_acc"] for r in d1_results]
    d1_f1s = [r["transfer_f1"] for r in d1_results]
    logger.info(f"  HateMM->MHC-EN: ACC={np.mean(d1_accs):.4f}+/-{np.std(d1_accs):.4f}  "
                f"F1={np.mean(d1_f1s):.4f}+/-{np.std(d1_f1s):.4f}")

    # --- Direction 2: Train MHC-EN -> Test HateMM ---
    logger.info("=" * 60)
    logger.info("Direction 2: Train on MHClip-EN -> Test on HateMM")
    logger.info("  Hyperparams: alpha=0.1, beta=0.1 (MHC-EN best)")
    logger.info("=" * 60)
    d2_results = []
    for seed in seeds:
        r = train_one_seed(
            seed=seed,
            train_feats=mhc_feats, train_ids=mhc_ids, train_lm=mhc_lm,
            eval_feats=hmm_feats, eval_ids=hmm_ids, eval_lm=hmm_lm,
            base_mk=base_mk, nc=nc,
            mr2_alpha=0.1, mr2_beta=0.1, lb_weight=0.01,
            class_weight=[1.0, 1.5], logger=logger
        )
        d2_results.append(r)
    all_results["MHC_EN_to_HateMM"] = d2_results

    d2_accs = [r["transfer_acc"] for r in d2_results]
    d2_f1s = [r["transfer_f1"] for r in d2_results]
    logger.info(f"  MHC-EN->HateMM: ACC={np.mean(d2_accs):.4f}+/-{np.std(d2_accs):.4f}  "
                f"F1={np.mean(d2_f1s):.4f}+/-{np.std(d2_f1s):.4f}")

    # --- Summary ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("CROSS-DATASET TRANSFER SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  HateMM -> MHC-EN:  ACC={np.mean(d1_accs):.4f}+/-{np.std(d1_accs):.4f}  "
                f"F1={np.mean(d1_f1s):.4f}+/-{np.std(d1_f1s):.4f}")
    logger.info(f"  MHC-EN -> HateMM:  ACC={np.mean(d2_accs):.4f}+/-{np.std(d2_accs):.4f}  "
                f"F1={np.mean(d2_f1s):.4f}+/-{np.std(d2_f1s):.4f}")
    logger.info("=" * 60)

    # Save results
    with open(os.path.join(save_dir, "transfer_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {save_dir}/transfer_results.json")


if __name__ == "__main__":
    main()
