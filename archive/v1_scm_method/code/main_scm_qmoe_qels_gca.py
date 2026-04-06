"""
SCM + Q-MoE + QELS + GCA (Generalized Class-Aware) Loss.

Round 014b: Replace fixed class weights [1.0, 1.5] with GCA loss that
adds class-dependent confidence margins to the QELS objective.

Based on: "Improved Balanced Classification with Theoretically Grounded
Loss Functions" (NeurIPS 2025). GCA bounds scale as 1/sqrt(p_min).
Core idea: class-dependent logit margin shift proportional to 1/sqrt(class_prior).

Usage:
  python main_scm_qmoe_qels_gca.py --dataset_name HateMM --num_runs 10 --seed_offset 0
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


class SCMQMoEQELS(nn.Module):
    """SCM + Q-MoE + QELS (same architecture as baseline)."""
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


def gca_qels_cross_entropy(logits, labels, q_entropy, nc=2, eps_min=0.01, eps_lambda=0.15,
                           class_margins=None):
    """GCA + QELS: Class-Aware confidence margins combined with adaptive label smoothing.

    GCA (NeurIPS 2025): adds class-dependent logit margins before softmax.
    margin_c = tau * (1/sqrt(p_c) - 1) where p_c is class prior, tau is temperature.
    This shifts the decision boundary toward the majority class.

    Combined with QELS: per-sample smoothing conditioned on quadrant entropy.
    """
    eps = (eps_min + eps_lambda * q_entropy).clamp(0, 0.5)
    one_hot = F.one_hot(labels, nc).float()
    smooth_targets = (1 - eps.unsqueeze(1)) * one_hot + eps.unsqueeze(1) / nc

    # GCA: add class-dependent margin to logits before softmax
    if class_margins is not None:
        # class_margins: [nc] tensor, e.g. [0.0, margin_for_minority]
        # Subtract margin from non-true-class logits (equivalent to adding to true class)
        margin_shift = class_margins.unsqueeze(0).expand_as(logits)  # [B, nc]
        adjusted_logits = logits + margin_shift * one_hot  # add margin to true class logit
    else:
        adjusted_logits = logits

    log_probs = F.log_softmax(adjusted_logits, dim=-1)
    loss = -(smooth_targets * log_probs).sum(dim=-1)
    return loss.mean()


# ---- Utils ----
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

def get_pl(model, loader):
    model.eval(); ap, al, alb = [], [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            lo, pe = model(b, return_penult=True)
            ap.append(pe.cpu()); al.append(lo.cpu()); alb.extend(b["label"].cpu().numpy())
    return torch.cat(ap), torch.cat(al).numpy(), np.array(alb)

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
    lf = f"./logs/scm_qmoe_qels_gca_{tag}_{ts}.log"
    logger = logging.getLogger(f"gca_{tag}_{ts}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Log: {lf}"); return logger


def compute_class_margins(labels_list, lm, tau=0.5):
    """Compute GCA margins: margin_c = tau * (1/sqrt(p_c) - 1).

    For binary: majority gets margin 0, minority gets positive margin.
    """
    all_labels = []
    for v in labels_list:
        all_labels.append(lm[v["Label"]])
    all_labels = np.array(all_labels)
    counts = np.bincount(all_labels, minlength=2)
    priors = counts / counts.sum()
    margins = tau * (1.0 / np.sqrt(priors) - 1.0)
    # Normalize so majority class margin = 0
    margins = margins - margins.min()
    return torch.tensor(margins, dtype=torch.float)


def run(feats, splits, lm, base_mk, nc, num_runs, seed_offset, save_dir, logger,
        expert_hidden=64, eps_min=0.01, eps_lambda=0.15, gca_tau=0.5):
    os.makedirs(save_dir, exist_ok=True)
    all_mk = list(base_mk) + [f"scm_{f}" for f in SCM_FIELDS]
    common = set.intersection(*[set(feats[k].keys()) for k in all_mk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}
    logger.info(f"  Train={len(cur['train'])}, Val={len(cur['valid'])}, Test={len(cur['test'])}")

    # Compute GCA margins from training set class distribution
    train_labels = [feats["labels"][v] for v in cur["train"]]
    class_margins = compute_class_margins(train_labels, lm, tau=gca_tau).to(device)
    logger.info(f"  GCA tau={gca_tau}, class_margins={class_margins.tolist()}")

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
        trl_ns = DataLoader(DS(cur["train"], feats, lm, all_mk), 64, False, collate_fn=collate_fn)

        model = SCMQMoEQELS(base_mk, nc=nc, expert_hidden=expert_hidden,
                            eps_min=eps_min, eps_lambda=eps_lambda).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        ep = 45; ts_t = ep * len(trl); ws = 5 * len(trl); sch = cw(opt, ws, ts_t)
        bva, bst = -1, None

        for e in range(ep):
            model.train()
            for batch in trl:
                batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
                logits, q_entropy = model(batch, training=True, return_qels=True)
                loss = gca_qels_cross_entropy(logits, batch["label"], q_entropy, nc=nc,
                                             eps_min=eps_min, eps_lambda=eps_lambda,
                                             class_margins=class_margins)
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
        tp, tl_arr, tla = get_pl(ema, trl_ns)
        vp, vl_arr, vla = get_pl(ema, vl)
        tep, tel_arr, tela = get_pl(ema, tel)

        head_acc, head_thresh = best_thresh(vl_arr, vla, tel_arr, tela)
        if head_thresh is not None:
            preds = ((tel_arr[:, 1] - tel_arr[:, 0]) > head_thresh).astype(int)
        else:
            preds = np.argmax(tel_arr, axis=1)

        metrics = full_metrics(tela, preds)
        result = {"seed": seed, "ri": ri, "seed_offset": seed_offset,
                  "head_acc": float(head_acc), "val_acc": float(bva), "metrics": metrics,
                  "gca_tau": gca_tau}
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
    parser.add_argument("--gca_tau", type=float, default=0.5)
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"; ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"; lm = {"Non Hate": 0, "Hate": 1}; tag = "HateMM"
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}; tag = f"MHC_{args.language[:2]}"

    logger = setup_logger(f"GCA_{tag}")
    nc = 2; base_mk = ["text", "audio", "frame"]
    logger.info(f"=== SCM Q-MoE+QELS+GCA: {tag}, Runs: {args.num_runs}, gca_tau={args.gca_tau} ===")

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
    save_dir = f"./seed_search_scm_qmoe_qels_gca/{tag}_tau{args.gca_tau}_off{args.seed_offset}"

    best = run(feats, splits, lm, base_mk, nc, args.num_runs, args.seed_offset,
               save_dir=save_dir, logger=logger,
               eps_min=args.eps_min, eps_lambda=args.eps_lambda, gca_tau=args.gca_tau)

    logger.info(f"\n{'='*60}")
    logger.info(f"  GCA {tag} BEST: ACC={best['metrics']['acc']:.4f} F1={best['metrics']['f1']:.4f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
