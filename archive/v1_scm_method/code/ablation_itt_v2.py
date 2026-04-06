"""
Ablation study for ITT model — V2 (strict).
Each ablation properly removes the corresponding model structure, not zero-filling.

Usage:
  conda run -n HVGuard python ablation_itt_v2.py --dataset_name HateMM
  conda run -n HVGuard python ablation_itt_v2.py --dataset_name Multihateclip --language English
  conda run -n HVGuard python ablation_itt_v2.py --dataset_name Multihateclip --language Chinese
"""

import argparse, csv, json, os, random, copy, logging
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

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

# ---- Generic flat fusion (for baselines) ----
class FlatFusion(nn.Module):
    """All modalities treated equally via Multi-Head Gated Routing. No theory structure."""
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
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop * 0.5))
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


# ---- ITT Full Model (configurable) ----
class ITTFusion(nn.Module):
    """
    Configurable ITT model. Only builds modules for the fields/channels actually used.

    Args:
        base_mk: base modality keys
        channel_map: dict {channel_name: field_key} for active channels
        modulator_fields: list of field keys used for moderation (empty = no moderation)
        extra_fields: list of additional field keys to include as flat modalities
    """
    def __init__(self, base_mk, channel_map, modulator_fields=None, extra_fields=None,
                 hidden=192, nh=4, nc=2, drop=0.15, md=0.15):
        super().__init__()
        self.base_mk = base_mk; self.md = md; self.hidden = hidden
        self.channel_map = channel_map
        self.modulator_fields = modulator_fields or []
        self.extra_fields = extra_fields or []
        n_base = len(base_mk)
        n_channels = len(channel_map)
        has_moderation = len(self.modulator_fields) == 2

        # Base modality projections
        self.base_projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(n_base)])

        # Channel field projections — deterministic order matching main_itt.py
        seen = set()
        all_field_keys = []
        for fk in list(channel_map.values()) + list(self.modulator_fields) + list(self.extra_fields):
            if fk not in seen:
                all_field_keys.append(fk)
                seen.add(fk)

        self.field_projs = nn.ModuleDict(
            [(fk, nn.Sequential(
                nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
            )) for fk in all_field_keys]
        )

        # Channel attention modules
        self.channel_attn = nn.ModuleDict({
            ch_name: nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop),
                nn.Linear(hidden, hidden)
            ) for ch_name in channel_map
        })

        # Threat Accumulator
        if n_channels > 0:
            self.channel_gate = nn.Sequential(nn.Linear(hidden * n_channels, n_channels))
            self.accum_proj = nn.Sequential(
                nn.Linear(hidden * n_channels, hidden * 2), nn.GELU(), nn.Dropout(drop),
                nn.LayerNorm(hidden * 2))

        # Extra field projections (treated as flat modalities concatenated)
        n_extra = len(self.extra_fields)

        # Moderation block (only if 2 modulator fields)
        self.has_moderation = has_moderation
        if has_moderation:
            self.mod_a = nn.Linear(hidden, 1)
            self.mod_b = nn.Linear(hidden, 1)
            self.mod_interact = nn.Linear(2, 1)

        # Base routing
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])

        # Classifier dim
        cd = nh * hidden + hidden  # base routing
        if n_channels > 0:
            cd += hidden * 2  # threat accumulator
        if has_moderation:
            cd += 1  # moderation scalar
        cd += n_extra * hidden  # extra fields

        self.pre_cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop * 0.5))
        self.head = nn.Linear(64, nc)

    def forward(self, batch, training=False, return_penult=False):
        # Base
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
        base_ctx = base_stack.mean(dim=1)

        # Encode all active fields
        field_h = {}
        for fk, proj in self.field_projs.items():
            h = proj(batch[fk])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            field_h[fk] = h

        parts = [base_repr]

        # Channels
        if len(self.channel_map) > 0:
            channel_outputs = []
            for ch_name, ch_field in self.channel_map.items():
                ch_input = torch.cat([field_h[ch_field], base_ctx], dim=-1)
                ch_out = self.channel_attn[ch_name](ch_input)
                channel_outputs.append(ch_out)
            ch_cat = torch.cat(channel_outputs, dim=-1)
            threat_repr = self.accum_proj(ch_cat)

            # Moderation
            if self.has_moderation:
                mod_keys = self.modulator_fields
                sa = torch.sigmoid(self.mod_a(field_h[mod_keys[0]]))
                sb = torch.sigmoid(self.mod_b(field_h[mod_keys[1]]))
                moderation = torch.sigmoid(self.mod_interact(torch.cat([sa, sb], dim=-1)))
                threat_repr = threat_repr * (0.5 + 0.5 * moderation)
                parts.append(threat_repr)
                parts.append(moderation)
            else:
                parts.append(threat_repr)

        # Extra fields
        for fk in self.extra_fields:
            parts.append(field_h[fk])

        fused = torch.cat(parts, dim=-1)
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
    mean = train_z.mean(dim=0, keepdim=True); c = (train_z - mean).numpy()
    lw = LedoitWolf().fit(c); cov = torch.tensor(lw.covariance_, dtype=torch.float32)
    U, S, V = torch.svd(cov)
    if r and r < U.size(1): U = U[:, :r]; S = S[:r]; V = V[:, :r]
    W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-6))
    return tuple(F.normalize((x - mean) @ W, dim=1) for x in [train_z, val_z, test_z])

def zca_whiten(train_z, val_z, test_z):
    mean = train_z.mean(dim=0, keepdim=True); c = train_z - mean
    cov = (c.t() @ c) / (c.size(0) - 1)
    U, S, V = torch.svd(cov + 1e-5 * torch.eye(cov.size(0)))
    W = U @ torch.diag(1.0 / torch.sqrt(S)) @ V.t()
    return tuple(F.normalize((x - mean) @ W, dim=1) for x in [train_z, val_z, test_z])

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
    topk_labels = bank_labels[topk_idx]; weights = F.softmax(topk_sim / temperature, dim=1)
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
    return {"acc": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average='macro')),
            "p": float(precision_score(y_true, y_pred, average='macro')),
            "r": float(recall_score(y_true, y_pred, average='macro'))}


def train_and_eval(model_fn, feats, splits, lm, mk, nc, seed, class_weight, best_config):
    """Train model from scratch and evaluate with given retrieval config.
    model_fn: callable that returns a new model instance (called AFTER seed is set)."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    common = set.intersection(*[set(feats[k].keys()) for k in mk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}

    trl = DataLoader(DS(cur["train"], feats, lm, mk), 32, True, collate_fn=collate_fn)
    vl = DataLoader(DS(cur["valid"], feats, lm, mk), 64, False, collate_fn=collate_fn)
    tel = DataLoader(DS(cur["test"], feats, lm, mk), 64, False, collate_fn=collate_fn)
    trl_ns = DataLoader(DS(cur["train"], feats, lm, mk), 64, False, collate_fn=collate_fn)

    model = model_fn().to(device)
    ema = copy.deepcopy(model)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float).to(device), label_smoothing=0.03)
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
                ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
        va = accuracy_score(ls2, ps)
        if va > bva: bva = va; bst = {k: v.clone() for k, v in ema.state_dict().items()}

    ema.load_state_dict(bst)

    # Retrieval
    tp, tl_arr, tla = get_pl(ema, trl_ns)
    vp, vl_arr, vla = get_pl(ema, vl)
    tep, tel_arr, tela = get_pl(ema, tel)
    blt = torch.tensor(tla)

    wname = best_config.get("whiten", "none")
    if wname == "zca": tp, vp, tep = zca_whiten(tp, vp, tep)
    elif wname.startswith("spca_r"): tp, vp, tep = shrinkage_pca_whiten(tp, vp, tep, r=int(wname.split("r")[1]))

    knn_type = best_config.get("knn_type", "none")
    if knn_type != "none":
        k = best_config["k"]; temp = best_config["temp"]; alpha = best_config["alpha"]
        knn_fn = cosine_knn if knn_type == "cosine" else csls_knn
        kt = knn_fn(tep, tp, blt, k=k, nc=nc, temperature=temp)
        kv = knn_fn(vp, tp, blt, k=k, nc=nc, temperature=temp)
        bl_val = (1 - alpha) * vl_arr + alpha * kv; bl_test = (1 - alpha) * tel_arr + alpha * kt
    else:
        bl_val = vl_arr; bl_test = tel_arr

    acc, thresh = best_thresh(bl_val, vla, bl_test, tela)
    if thresh is not None: preds = ((bl_test[:, 1] - bl_test[:, 0]) > thresh).astype(int)
    else: preds = np.argmax(bl_test, axis=1)
    return full_metrics(tela, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English", choices=["English", "Chinese"])
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"; ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"; lm = {"Non Hate": 0, "Hate": 1}; tag = "HateMM"
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}; tag = f"MHC_{args.language[:2]}"

    with open(f"./seed_search_itt/{tag}_off0/all_results.json") as f:
        best = json.load(f)["global_best"]
    seed = best["seed"]; best_config = best["best_config"]

    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"ablv2_itt_{tag}_{ts}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(f"./logs/ablation_itt_v2_{tag}_{ts}.log", encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)

    nc = 2; base_mk = ["text", "audio", "frame"]; cw_val = [1.0, 1.5]

    # Load all features
    feats = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
    }
    itt_fields = ["target_salience", "realistic_threat", "symbolic_threat",
                  "anxiety_discomfort", "stereotype_support", "hostility_prescription"]
    for f in itt_fields:
        feats[f"itt_{f}"] = torch.load(f"{emb_dir}/itt_{f}_features.pth", map_location="cpu")
    feats["itt_rationale"] = torch.load(f"{emb_dir}/itt_rationale_features.pth", map_location="cpu")
    with open(ann_path) as f2:
        feats["labels"] = {d["Video_ID"]: d for d in json.load(f2)}
    splits = load_split_ids(split_dir)

    # Full model config
    full_channel_map = {
        "realistic": "itt_realistic_threat",
        "symbolic": "itt_symbolic_threat",
        "anxiety": "itt_anxiety_discomfort",
        "hostility": "itt_hostility_prescription",
    }
    full_modulators = ["itt_target_salience", "itt_stereotype_support"]
    all_itt_mk = [f"itt_{f}" for f in itt_fields]

    logger.info(f"=== ITT Ablation V2 (strict): {tag} ===")
    logger.info(f"Best seed={seed}, config={best_config}")

    results = []

    def run_abl(abl_id, desc, model_fn, mk):
        logger.info(f"  [{abl_id}] {desc}")
        m = train_and_eval(model_fn, feats, splits, lm, mk, nc, seed, cw_val, best_config)
        logger.info(f"    ACC={m['acc']:.4f} F1={m['f1']:.4f} P={m['p']:.4f} R={m['r']:.4f}")
        results.append({"abl_id": abl_id, "desc": desc, **m})

    # ---- Ablations (all model_fn are lambdas, created AFTER seed is set) ----

    # ABL-00: Full model
    run_abl("ABL-00", "Full ITT model",
        lambda: ITTFusion(base_mk, full_channel_map, full_modulators),
        base_mk + all_itt_mk)

    # ABL-01: No rationale (base only, 3 modalities)
    run_abl("ABL-01", "No rationale (base only)",
        lambda: FlatFusion(base_mk, nc=nc),
        base_mk)

    # ABL-02: Pooled rationale as flat modality (no theory structure)
    run_abl("ABL-02", "Pooled rationale (no theory fusion)",
        lambda: FlatFusion(base_mk + ["itt_rationale"], nc=nc),
        base_mk + ["itt_rationale"])

    # ABL-03: Per-field as flat modalities (no channels, no moderation — just all 9 modalities in flat routing)
    run_abl("ABL-03", "Per-field flat fusion (no theory structure)",
        lambda: FlatFusion(base_mk + all_itt_mk, nc=nc),
        base_mk + all_itt_mk)

    # ABL-04: Full channels but NO moderation block
    run_abl("ABL-04", "No Threat Moderation Block",
        lambda: ITTFusion(base_mk, full_channel_map, modulator_fields=[]),
        base_mk + [v for v in all_itt_mk if v not in ["itt_target_salience", "itt_stereotype_support"]])

    # ABL-05: No parallel channels — all 4 threat fields as flat extras (no channel structure, no moderation)
    run_abl("ABL-05", "No parallel channels (threat fields as flat extras)",
        lambda: ITTFusion(base_mk, channel_map={}, modulator_fields=[],
                  extra_fields=["itt_realistic_threat", "itt_symbolic_threat",
                                "itt_anxiety_discomfort", "itt_hostility_prescription"]),
        base_mk + all_itt_mk)

    # ABL-06 to ABL-09: Remove individual channels (3 remaining channels + moderation)
    for rm_ch, rm_field in [("realistic", "itt_realistic_threat"),
                             ("symbolic", "itt_symbolic_threat"),
                             ("anxiety", "itt_anxiety_discomfort"),
                             ("hostility", "itt_hostility_prescription")]:
        ch_map = {k: v for k, v in full_channel_map.items() if k != rm_ch}
        mk_used = base_mk + [v for v in all_itt_mk if v != rm_field]
        run_abl(f"ABL-ch-no_{rm_ch}", f"Remove {rm_ch} channel (structure removed)",
            lambda cm=ch_map: ITTFusion(base_mk, cm, full_modulators),
            mk_used)

    # ABL-10 to ABL-15: Remove individual fields (remove from structure too)
    run_abl("ABL-field-no_target_salience", "Remove target_salience (no moderation)",
        lambda: ITTFusion(base_mk, full_channel_map, modulator_fields=[]),
        base_mk + [f"itt_{f}" for f in itt_fields if f != "target_salience"])

    run_abl("ABL-field-no_stereotype_support", "Remove stereotype_support (no moderation)",
        lambda: ITTFusion(base_mk, full_channel_map, modulator_fields=[]),
        base_mk + [f"itt_{f}" for f in itt_fields if f != "stereotype_support"])

    for rm_field, rm_ch in [("realistic_threat", "realistic"),
                             ("symbolic_threat", "symbolic"),
                             ("anxiety_discomfort", "anxiety"),
                             ("hostility_prescription", "hostility")]:
        ch_map = {k: v for k, v in full_channel_map.items() if k != rm_ch}
        mk_used = base_mk + [f"itt_{f}" for f in itt_fields if f != rm_field]
        run_abl(f"ABL-field-no_{rm_field}", f"Remove {rm_field} field+channel",
            lambda cm=ch_map: ITTFusion(base_mk, cm, full_modulators),
            mk_used)

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"ABLATION SUMMARY V2 — ITT {tag}")
    logger.info(f"{'='*80}")
    logger.info(f"{'ID':<35} {'Description':<42} {'ACC':>7} {'F1':>7} {'Δ F1':>7}")
    logger.info(f"{'-'*98}")
    base_f1 = results[0]["f1"]
    for r in results:
        delta = r["f1"] - base_f1
        logger.info(f"{r['abl_id']:<35} {r['desc']:<42} {r['acc']:>7.4f} {r['f1']:>7.4f} {delta:>+7.4f}")

    os.makedirs("./ablation_results", exist_ok=True)
    save_path = f"./ablation_results/itt_v2_{tag}_ablations.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
