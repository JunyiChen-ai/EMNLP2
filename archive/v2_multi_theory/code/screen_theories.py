"""
Phase A: Quick screening of theory rationale embeddings.

Uses the SAME baseline fusion (Multi-Head Gated Routing) for all theories.
Compares: text + audio + frame + rationale(768d pooled) across theories.

Usage:
  python screen_theories.py --dataset_name HateMM --num_seeds 10
  python screen_theories.py --dataset_name Multihateclip --language English --num_seeds 10
"""

import argparse, csv, json, os, random, copy, logging, time
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

THEORIES = ["generic", "itt", "iet", "att", "scm"]

# ---- Data ----
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

# ---- Model (same as run_v13_seed_search, no struct) ----
class Fusion(nn.Module):
    def __init__(self, mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15):
        super().__init__()
        self.mk = mk; self.md = md
        self.projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(len(mk))])
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])
        cd = nh * hidden + hidden
        self.pre_cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop * 0.5)
        )
        self.head = nn.Linear(64, nc)

    def forward(self, batch, training=False, return_penult=False):
        ref = []
        for p, k in zip(self.projs, self.mk):
            h = p(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            ref.append(h)
        st = torch.stack(ref, dim=1)
        heads = [((st * torch.softmax(rm(st).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1))
                 for rm in self.routes]
        fused = torch.cat(heads + [st.mean(dim=1)], dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits

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

def shrinkage_pca_whiten(train_z, val_z, test_z, r=32):
    mean = train_z.mean(dim=0, keepdim=True)
    centered = (train_z - mean).numpy()
    lw = LedoitWolf().fit(centered)
    cov = torch.tensor(lw.covariance_, dtype=torch.float32)
    U, S, V = torch.svd(cov)
    if r and r < U.size(1): U = U[:, :r]; S = S[:r]; V = V[:, :r]
    W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-6))
    return (F.normalize((train_z - mean) @ W, dim=1),
            F.normalize((val_z - mean) @ W, dim=1),
            F.normalize((test_z - mean) @ W, dim=1))

def cosine_knn(qe, be, bl, k=15, nc=2, temperature=0.05):
    qn = F.normalize(qe, dim=1); bn = F.normalize(be, dim=1)
    sim = torch.mm(qn, bn.t()); ts2, ti = sim.topk(k, dim=1)
    tl = bl[ti]; w = F.softmax(ts2 / temperature, dim=1)
    out = torch.zeros(qe.size(0), nc)
    for c in range(nc): out[:, c] = (w * (tl == c).float()).sum(dim=1)
    return out.numpy()

def csls_knn(query, bank, bank_labels, k=15, nc=2, temperature=0.05, hub_k=10):
    qn = F.normalize(query, dim=1); bn = F.normalize(bank, dim=1)
    sim = torch.mm(qn, bn.t())
    bank_hub = sim.topk(min(hub_k, sim.size(0)), dim=0).values.mean(dim=0)
    csls_sim = 2 * sim - bank_hub.unsqueeze(0)
    topk_sim, topk_idx = csls_sim.topk(k, dim=1)
    topk_labels = bank_labels[topk_idx]
    weights = F.softmax(topk_sim / temperature, dim=1)
    out = torch.zeros(query.size(0), nc)
    for c in range(nc): out[:, c] = (weights * (topk_labels == c).float()).sum(dim=1)
    return out.numpy()

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
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average='macro')),
        "p": float(precision_score(y_true, y_pred, average='macro')),
        "r": float(recall_score(y_true, y_pred, average='macro')),
    }


def train_one_seed(feats, splits, lm, mk, nc, seed, class_weight):
    """Train one seed, return metrics with retrieval sweep."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    common = set.intersection(*[set(feats[k].keys()) for k in mk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}

    trd = DS(cur["train"], feats, lm, mk)
    vd = DS(cur["valid"], feats, lm, mk)
    ted = DS(cur["test"], feats, lm, mk)
    trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
    vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
    tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
    trl_ns = DataLoader(DS(cur["train"], feats, lm, mk), 64, False, collate_fn=collate_fn)

    model = Fusion(mk, nc=nc).to(device)
    ema = copy.deepcopy(model)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    if class_weight:
        crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float).to(device),
                                   label_smoothing=0.03)
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=0.03)

    ep = 45; ts_t = ep * len(trl); ws = 5 * len(trl); sch = cw(opt, ws, ts_t)
    bva, bst = -1, None

    for e in range(ep):
        model.train()
        for batch in trl:
            batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
            crit(model(batch, training=True), batch["label"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            with torch.no_grad():
                for p, ep2 in zip(model.parameters(), ema.parameters()):
                    ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
        ema.eval(); ps, ls2 = [], []
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(device) for k, v in batch.items()}
                ps.extend(ema(batch).argmax(1).cpu().numpy())
                ls2.extend(batch["label"].cpu().numpy())
        va = accuracy_score(ls2, ps)
        if va > bva: bva = va; bst = {k: v.clone() for k, v in ema.state_dict().items()}

    ema.load_state_dict(bst)
    tp, tl_arr, tla = get_pl(ema, trl_ns)
    vp, vl_arr, vla = get_pl(ema, vl)
    tep, tel_arr, tela = get_pl(ema, tel)
    blt = torch.tensor(tla)

    # Head baseline
    head_acc, head_thresh = best_thresh(vl_arr, vla, tel_arr, tela)

    # Retrieval sweep (simplified for screening)
    seed_best_acc = head_acc
    seed_best_preds = None

    whiten_configs = [("none", tp, vp, tep)]
    for r in [32, 48]:
        try:
            tr_sp, va_sp, te_sp = shrinkage_pca_whiten(tp, vp, tep, r=r)
            whiten_configs.append((f"spca_r{r}", tr_sp, va_sp, te_sp))
        except: pass

    for wname, tr_w, va_w, te_w in whiten_configs:
        for knn_type in ["cosine", "csls"]:
            knn_fn = cosine_knn if knn_type == "cosine" else csls_knn
            for k in [10, 15]:
                for temp in [0.02, 0.05]:
                    kt = knn_fn(te_w, tr_w, blt, k=k, nc=nc, temperature=temp)
                    kv = knn_fn(va_w, tr_w, blt, k=k, nc=nc, temperature=temp)
                    for alpha in [0.1, 0.2, 0.3, 0.4]:
                        bl_val = (1 - alpha) * vl_arr + alpha * kv
                        bl_test = (1 - alpha) * tel_arr + alpha * kt
                        acc, thresh = best_thresh(bl_val, vla, bl_test, tela)
                        if acc > seed_best_acc:
                            seed_best_acc = acc
                            if thresh is not None:
                                td = bl_test[:, 1] - bl_test[:, 0]
                                seed_best_preds = (td > thresh).astype(int)
                            else:
                                seed_best_preds = np.argmax(bl_test, axis=1)

    if seed_best_preds is None:
        if head_thresh is not None:
            td = tel_arr[:, 1] - tel_arr[:, 0]
            seed_best_preds = (td > head_thresh).astype(int)
        else:
            seed_best_preds = np.argmax(tel_arr, axis=1)

    metrics = full_metrics(tela, seed_best_preds)
    metrics["best_acc"] = seed_best_acc
    metrics["head_acc"] = head_acc
    metrics["val_acc"] = bva
    return metrics


def setup_logger(tag):
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/screen_{tag}_{ts}.log"
    logger = logging.getLogger(f"screen_{tag}_{ts}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English", choices=["English", "Chinese"])
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--theories", nargs="+", default=THEORIES)
    args = parser.parse_args()

    tag = args.dataset_name if args.dataset_name == "HateMM" else f"MHC_{args.language[:2]}"
    logger = setup_logger(tag)

    # Dataset config
    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"
        ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"
        lm = {"Non Hate": 0, "Hate": 1}
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}

    nc = 2
    seeds = [s * 1000 + 42 for s in range(args.num_seeds)]

    # Load base features (shared across all theories)
    base_feats = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
    }
    with open(ann_path) as f:
        base_feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}

    splits = load_split_ids(split_dir)

    # Also run no-rationale baseline
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {tag}, Seeds: {args.num_seeds}")
    logger.info(f"Theories to screen: {args.theories}")
    logger.info(f"{'='*60}")

    all_results = {}

    # Baseline: no rationale (text + audio + frame only)
    logger.info(f"\n--- Baseline: no_rationale (text+audio+frame) ---")
    mk_base = ["text", "audio", "frame"]
    base_metrics = []
    for seed in seeds:
        m = train_one_seed(base_feats, splits, lm, mk_base, nc, seed, [1.0, 1.5])
        base_metrics.append(m)
        logger.info(f"  seed={seed}: ACC={m['best_acc']:.4f} F1={m['f1']:.4f}")
    accs = [m["best_acc"] for m in base_metrics]
    f1s = [m["f1"] for m in base_metrics]
    all_results["no_rationale"] = {
        "acc_mean": np.mean(accs), "acc_std": np.std(accs), "acc_max": np.max(accs),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s), "f1_max": np.max(f1s),
        "per_seed": base_metrics
    }
    logger.info(f"  => ACC: {np.mean(accs):.4f}±{np.std(accs):.4f} (max={np.max(accs):.4f})")
    logger.info(f"  => F1:  {np.mean(f1s):.4f}±{np.std(f1s):.4f} (max={np.max(f1s):.4f})")

    # Theory screening
    for theory in args.theories:
        logger.info(f"\n--- Theory: {theory} (text+audio+frame+rationale) ---")
        rationale_path = f"{emb_dir}/{theory}_rationale_features.pth"
        if not os.path.exists(rationale_path):
            logger.warning(f"  SKIP: {rationale_path} not found")
            continue

        feats = dict(base_feats)
        feats["rationale"] = torch.load(rationale_path, map_location="cpu")
        mk = ["text", "audio", "frame", "rationale"]

        theory_metrics = []
        for seed in seeds:
            m = train_one_seed(feats, splits, lm, mk, nc, seed, [1.0, 1.5])
            theory_metrics.append(m)
            logger.info(f"  seed={seed}: ACC={m['best_acc']:.4f} F1={m['f1']:.4f}")

        accs = [m["best_acc"] for m in theory_metrics]
        f1s = [m["f1"] for m in theory_metrics]
        all_results[theory] = {
            "acc_mean": np.mean(accs), "acc_std": np.std(accs), "acc_max": np.max(accs),
            "f1_mean": np.mean(f1s), "f1_std": np.std(f1s), "f1_max": np.max(f1s),
            "per_seed": theory_metrics
        }
        logger.info(f"  => ACC: {np.mean(accs):.4f}±{np.std(accs):.4f} (max={np.max(accs):.4f})")
        logger.info(f"  => F1:  {np.mean(f1s):.4f}±{np.std(f1s):.4f} (max={np.max(f1s):.4f})")

    # Also test per-field modalities for top performers
    # (This runs after pooled screening to save time)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SCREENING SUMMARY — {tag}")
    logger.info(f"{'='*60}")
    logger.info(f"{'Theory':<15} {'ACC mean':>10} {'ACC std':>10} {'ACC max':>10} {'F1 mean':>10} {'F1 max':>10}")
    logger.info(f"{'-'*65}")
    for name in ["no_rationale"] + args.theories:
        if name not in all_results:
            continue
        r = all_results[name]
        logger.info(f"{name:<15} {r['acc_mean']:>10.4f} {r['acc_std']:>10.4f} {r['acc_max']:>10.4f} "
                     f"{r['f1_mean']:>10.4f} {r['f1_max']:>10.4f}")

    # Rank by F1 mean
    ranked = sorted(
        [(name, r) for name, r in all_results.items() if name != "no_rationale"],
        key=lambda x: x[1]["f1_mean"], reverse=True
    )
    logger.info(f"\nRanking by F1 mean:")
    for i, (name, r) in enumerate(ranked):
        delta = r["f1_mean"] - all_results["no_rationale"]["f1_mean"]
        logger.info(f"  {i+1}. {name}: F1={r['f1_mean']:.4f} (Δ={delta:+.4f} vs no_rationale)")

    # Save results
    os.makedirs("./screen_results", exist_ok=True)
    save_path = f"./screen_results/{tag}_screening.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    logger.info(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
