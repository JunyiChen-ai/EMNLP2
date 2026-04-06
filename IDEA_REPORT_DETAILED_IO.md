# 5个方法的详细数据流 (Step-by-Step Input/Output)

**当前pipeline基础**：
```
视频 → [冻结] MLLM → 5个文本字段(JSON) → BERT [CLS] → unit embeddings [5, 768]
                                                      → whole rationale [768]
视频 → WavLM → audio [768]
视频 → ResNet/ViT → frame [768]
全部 → 分类器 → logits [2]
```

所有方法共享 Step 0（已有的冻结特征），区别在于 Step 0 之后怎么用这些特征。

---

## Idea 1: Grounded Token Trust (GTT) — 基于token信任度的grounding融合

**核心思想**：用audio/frame来判断MLLM生成的每个文本token是否有"证据支撑"，没有支撑的token降权。

### Step 0: 特征准备（已有，不变）

| 数据 | 维度 | 来源 |
|------|------|------|
| 5个字段的BERT token embeddings | `[B, 5, T, 768]` | **需要新提取**：不再用[CLS]，而是保留每个token的hidden state。T=max_len=128 |
| audio feature | `[B, 768]` | 已有 wavlm_audio_features.pth |
| frame feature | `[B, 768]` | 已有 frame_features.pth |
| AV concat | `[B, 1536]` | concat(audio, frame) |

> ⚠️ 与现有pipeline的区别：需要重新跑BERT，保留token-level embeddings而不仅仅是[CLS]。

### Step 1: Token-Level Trust Scoring

**输入**：
- token embeddings: `[B, 5, T, 768]` — 5个字段，每个字段T个token
- AV features: `[B, 1536]`

**操作**：
```
AV_proj = Linear(1536, 768)(AV)           # [B, 768]
AV_expanded = AV_proj.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 768]

# 每个token与AV的相似度作为"支撑分数"
trust_score = sigmoid(cosine_sim(token_emb, AV_expanded))  # [B, 5, T]
# 或用一个小MLP：
trust_score = sigmoid(MLP([token_emb; AV_expanded]))        # [B, 5, T]
```

**输出**：
- trust_score: `[B, 5, T]` — 每个token的信任度 ∈ [0, 1]

### Step 2: Trust-Weighted Pooling

**输入**：
- token embeddings: `[B, 5, T, 768]`
- trust_score: `[B, 5, T]`
- attention_mask: `[B, 5, T]`（padding mask）

**操作**：
```
# 用trust score做加权平均池化（替代[CLS]）
weighted_mask = trust_score * attention_mask           # [B, 5, T]
weighted_emb = (token_emb * weighted_mask.unsqueeze(-1)).sum(dim=2)  # [B, 5, 768]
weighted_emb = weighted_emb / weighted_mask.sum(dim=2, keepdim=True)  # [B, 5, 768]
```

**输出**：
- trust-weighted unit embeddings: `[B, 5, 768]` — 每个字段的信任度加权表示

### Step 3: 分类

**输入**：
- trust-weighted units: `[B, 5, 768]`
- （可选）AV features: `[B, 1536]`

**操作**：
```
# 方案A: 注意力聚合5个字段 → 768 → MLP → 2
pooled = attention_pool(trust_weighted_units)  # [B, 768]
logits = MLP(pooled)                          # [B, 2]

# 方案B: concat AV
fused = concat(pooled, AV)                    # [B, 2304]
logits = MLP(fused)                           # [B, 2]
```

**输出**：
- logits: `[B, 2]`

### 训练信号
- 主loss: CrossEntropy(logits, label)
- （可选）辅助loss: 鼓励trust_score的分布不要太平坦（entropy regularization）

---

## Idea 2: Variance-Minimized Boundary Training (VMBT) — 方差最小化边界训练

**核心思想**：用EMA教师+分布对齐正则化来稳定决策边界，减少seed variance。

### Step 0: 特征准备（已有，不变）

| 数据 | 维度 | 来源 |
|------|------|------|
| whole rationale embedding | `[B, 768]` | 已有 generic_rationale_features.pth |
| unit embeddings | `[B, 5, 768]` | 已有 unit_features.pth |
| audio | `[B, 768]` | 已有 |
| frame | `[B, 768]` | 已有 |

### Step 1: 基础分类器（任何现有架构）

**输入**：同现有pipeline
**输出**：
- feature（分类前的hidden）: `[B, 256]`
- logits: `[B, 2]`

### Step 2: EMA Teacher Snapshot

**输入**：当前模型参数 θ

**操作**：
```
# 指数移动平均更新教师
θ_teacher = α * θ_teacher + (1 - α) * θ_student    # α=0.999
feature_teacher = teacher_forward(x)                 # [B, 256]，stop_grad
```

**输出**：
- feature_teacher: `[B, 256]`（frozen，不回传梯度）

### Step 3: Feature Distribution Alignment (Proxy-FDA)

**输入**：
- feature_student: `[B, 256]`（当前模型的hidden表示）
- feature_teacher: `[B, 256]`（EMA教师的hidden表示）

**操作**：
```
# 计算每个类别的特征分布
for cls in [0, 1]:
    mask_cls = (label == cls)
    feat_s = feature_student[mask_cls]    # [B_cls, 256]
    feat_t = feature_teacher[mask_cls]    # [B_cls, 256]

    # 最近邻图对齐：学生特征分布应与教师分布保持一致
    # 用MMD或CKA衡量分布距离
    L_fda += MMD(feat_s, feat_t)
```

**输出**：
- L_fda: scalar（分布对齐损失）

### Step 4: Hard-Sample Boosting

**输入**：
- logits: `[B, 2]`
- label: `[B]`
- margin: softmax(logits) 的预测置信度

**操作**：
```
# 低margin样本（靠近边界的）获得更高权重
confidence = softmax(logits).gather(1, label.unsqueeze(1))  # [B, 1]
sample_weight = 1.0 / (confidence + ε)                      # 越不确定越大
sample_weight = sample_weight / sample_weight.mean()         # 归一化
```

**输出**：
- sample_weight: `[B]`

### Step 5: 总训练目标

**输入**：logits, label, L_fda, sample_weight

```
L_ce = (sample_weight * CrossEntropy(logits, label, reduction='none')).mean()
L_total = L_ce + λ * L_fda      # λ=0.1
```

**输出**：L_total（反向传播）

### 评估指标（新）
- mean F1 over 20 seeds
- std F1 over 20 seeds
- worst-seed F1
- best-seed F1

---

## Idea 3: Boundary-Only Residual Fusion (BORF) — 仅在边界处的残差融合

**核心思想**：先训text-only分类器，冻住。然后训audio/frame残差分支，只在text不确定时修正。

### Step 0: 特征准备（已有，不变）

同上（whole rationale [768], audio [768], frame [768]）

### Step 1: Text-Only Base Classifier（先训练，然后冻结）

**输入**：text embedding `[B, 768]`

**操作**：
```
h_text = ReLU(Linear(768, 256)(text))   # [B, 256]
base_logit = Linear(256, 2)(h_text)     # [B, 2]
```

**输出**：
- base_logit: `[B, 2]`（冻结后作为基础分数）
- h_text: `[B, 256]`（冻结的text hidden）

### Step 2: Text×Audio 一阶残差分支

**输入**：
- h_text: `[B, 256]`（冻结）
- audio: `[B, 768]`

**操作**：
```
h_audio = Linear(768, 256)(audio)       # [B, 256]

# Taylor一阶交互项：逐元素乘
interaction_ta = h_text * h_audio       # [B, 256]（text×audio交叉项）

residual_ta = Linear(256, 2)(interaction_ta)  # [B, 2]
```

**输出**：
- residual_ta: `[B, 2]`（text×audio的修正量）

### Step 3: Text×Frame 一阶残差分支

**输入**：
- h_text: `[B, 256]`（冻结）
- frame: `[B, 768]`

**操作**：（同上结构）
```
h_frame = Linear(768, 256)(frame)       # [B, 256]
interaction_tf = h_text * h_frame       # [B, 256]
residual_tf = Linear(256, 2)(interaction_tf)  # [B, 2]
```

**输出**：
- residual_tf: `[B, 2]`

### Step 4: Ambiguity Gate（文本不确定时才用残差）

**输入**：
- base_logit: `[B, 2]`（冻结）
- residual_ta: `[B, 2]`
- residual_tf: `[B, 2]`

**操作**：
```
# 文本置信度 → 门控
text_confidence = softmax(base_logit).max(dim=1).values  # [B]，越高=越确定
gate = sigmoid(-β * (text_confidence - τ))                # [B]，τ=0.7, β=10
# text_confidence高 → gate≈0（不需要修正）
# text_confidence低 → gate≈1（需要修正）

gate = gate.unsqueeze(1)  # [B, 1]
final_logit = base_logit + gate * (residual_ta + residual_tf)  # [B, 2]
```

**输出**：
- final_logit: `[B, 2]`

### Step 5: 训练（两阶段）

```
阶段1: 训练text-only base classifier → 冻结
阶段2: 只训练残差分支(h_audio, h_frame projections + residual heads + gate参数)
       Loss = weighted_CE(final_logit, label)
       # 对hard samples（base_logit错误或低margin的）加权
```

---

## Idea 4: Residual Correlation Distillation (RCD) — 残差相关性蒸馏

**核心思想**：先训text-only教师。找出教师犯错/不确定的样本。只在这些样本上训练audio/frame学生，让它们学"text漏掉的信号"。

### Step 0: 特征准备（已有，不变）

同上

### Step 1: 训练Text-Only Teacher

**输入**：text embedding `[B, 768]`

**操作**：标准MLP训练

**输出**：
- text_logit: `[B, 2]`
- text_prob: `[B, 2]`（softmax后）
- 记录每个样本的text_correct: `[N]` bool
- 记录每个样本的text_confidence: `[N]` float

### Step 2: 定义 Residual Target（蒸馏目标）

**输入**：
- text_prob: `[N, 2]`（全训练集）
- true_label: `[N]`

**操作**：
```
# 残差 = 真实标签 - text预测的概率
one_hot_label = one_hot(true_label, 2)        # [N, 2]
residual_target = one_hot_label - text_prob    # [N, 2]

# residual_target解释：
# text正确且自信 → residual ≈ 0（audio/frame不需要学）
# text错误 → residual大（audio/frame需要学这个修正）
# text不确定 → residual中等
```

**输出**：
- residual_target: `[N, 2]`（每个样本的修正目标）

### Step 3: 训练 Audio Residual Branch

**输入**：
- audio: `[B, 768]`
- residual_target: `[B, 2]`
- sample_weight: `[B]`（text越错/越不确定 → 权重越大）

**操作**：
```
h_audio = ReLU(Linear(768, 256)(audio))
audio_residual = Linear(256, 2)(h_audio)    # [B, 2]，预测残差

# 加权MSE loss（不是CE！因为目标是连续残差）
loss_audio = (sample_weight * MSE(audio_residual, residual_target)).mean()
```

**输出**：
- audio_residual: `[B, 2]`（audio学到的修正量）

### Step 4: 训练 Frame Residual Branch（同上结构）

**输入**：frame `[B, 768]`, residual_target `[B, 2]`

**输出**：frame_residual `[B, 2]`

### Step 5: 推理时组合

**输入**：text_logit `[B, 2]`, audio_residual `[B, 2]`, frame_residual `[B, 2]`

```
final_logit = text_logit + α * audio_residual + β * frame_residual  # [B, 2]
# α, β 可以是固定超参(如0.3)，或用验证集搜索
```

**输出**：final_logit `[B, 2]`

### 训练流程总结
```
Phase 1: 训练text MLP → 冻结 → 计算residual_target
Phase 2: 训练audio branch（MSE on residual_target）
Phase 3: 训练frame branch（MSE on residual_target）
Phase 4: 在验证集上搜索 α, β
```

---

## Idea 5: Cross-Modal Description Editor (CMDE) — 跨模态文本编辑器

**核心思想**：用audio/frame的证据来决定MLLM生成的5个字段哪些该保留、哪些该降权，编辑后再分类。

### Step 0: 特征准备（已有，不变）

| 数据 | 维度 | 来源 |
|------|------|------|
| unit embeddings（5个字段[CLS]） | `[B, 5, 768]` | 已有 unit_features.pth |
| audio | `[B, 768]` | 已有 |
| frame | `[B, 768]` | 已有 |

### Step 1: 计算每个字段的跨模态支撑分数

**输入**：
- units: `[B, 5, 768]`
- audio: `[B, 768]`
- frame: `[B, 768]`

**操作**：
```
AV = concat(audio, frame)                    # [B, 1536]
AV_proj = Linear(1536, 768)(AV)              # [B, 768]

# 每个字段与AV的相关性
# units: [B, 5, 768], AV_proj: [B, 1, 768]
support_score = cosine_sim(units, AV_proj.unsqueeze(1))  # [B, 5]
# 或用bilinear：
support_score = (units @ W @ AV_proj.unsqueeze(-1)).squeeze(-1)  # [B, 5]
```

**输出**：
- support_score: `[B, 5]`（每个字段的跨模态支撑度）

### Step 2: 编辑（字段级门控）

**输入**：
- units: `[B, 5, 768]`
- support_score: `[B, 5]`

**操作**：
```
# 方案A: Soft masking（简单版）
edit_gate = sigmoid(Linear(1, 1)(support_score.unsqueeze(-1)))  # [B, 5, 1]
edited_units = units * edit_gate                                 # [B, 5, 768]

# 方案B: Contrastive reweighting（进阶版）
# 高支撑 → 保留原样
# 低支撑 → 用AV_proj替换/混合
mix_ratio = sigmoid(support_score).unsqueeze(-1)                 # [B, 5, 1]
edited_units = mix_ratio * units + (1 - mix_ratio) * AV_proj.unsqueeze(1)  # [B, 5, 768]
```

**输出**：
- edited_units: `[B, 5, 768]`（编辑后的字段表示）

### Step 3: 聚合 + 分类

**输入**：edited_units `[B, 5, 768]`

**操作**：
```
# 注意力池化
attn_weights = softmax(Linear(768, 1)(edited_units).squeeze(-1))  # [B, 5]
pooled = (edited_units * attn_weights.unsqueeze(-1)).sum(dim=1)   # [B, 768]

logits = MLP(pooled)  # [B, 2]
```

**输出**：logits `[B, 2]`

### Step 4: 训练

```
L_ce = CrossEntropy(logits, label)

# 辅助loss: 编辑不应太激进（防止删掉所有字段）
L_edit_reg = -mean(edit_gate)  # 鼓励保留（防止全部mask掉）
# 或 KL(edited_units, original_units) 限制编辑幅度

L_total = L_ce + λ * L_edit_reg
```

### 与现有方法的本质区别
```
现有方法:  text [768] ——concat/attention——→ [text; audio; frame] → classifier
CMDE:     text [5, 768] ——audio/frame编辑→ edited_text [5, 768] → classifier
                          ↑
                    audio/frame不出现在最终分类输入中
                    它们的作用是"编辑"文本，而不是"拼接"
```

---

## 5个方法对比总览

| 方法 | 新增模块 | 新增参数量(估) | 需要重新提取特征？ | 训练阶段数 | 风险 |
|------|---------|-------------|---------------|----------|------|
| GTT | Token trust head + weighted pooling | ~1M | ✅ 需要BERT token embeddings | 1 (end-to-end) | 中 |
| VMBT | EMA teacher + FDA loss | ~0 (只加loss) | ❌ | 1 (加正则) | 低 |
| BORF | 2个残差分支 + gate | ~200K | ❌ | 2 (先text，后残差) | 低-中 |
| RCD | 2个残差学生分支 | ~200K | ❌ | 3 (teacher→audio→frame) | 中 |
| CMDE | 支撑打分器 + 编辑门控 | ~500K | ❌ | 1 (end-to-end) | 高 |

| 方法 | 核心I/O变化 | 对audio/frame的角色定义 |
|------|-----------|---------------------|
| GTT | token-level加权池化替代[CLS] | **验证者**：判断每个token是否有AV支撑 |
| VMBT | 加正则loss，架构不变 | **不变**：只改训练稳定性 |
| BORF | 加残差分支到logit上 | **修正者**：只在text不确定时修正logit |
| RCD | AV分支学残差target | **补充者**：学text犯错时的修正信号 |
| CMDE | AV编辑text表示后再分类 | **编辑者**：重写text表示，自己不参与分类 |
