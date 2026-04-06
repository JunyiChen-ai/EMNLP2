# Archive Index

所有历史实验代码、结果和文档。项目根目录已清空，从零开始。

---

## 顶层文件

| 文件 | 说明 |
|------|------|
| `BEST_CONFIGS.md` | **所有历史最优配置汇总**（每个理论×每个数据集的best config和performance） |
| `IDEA_REPORT.md` | 原始idea文档：5种社会心理学理论的筛选过程和理论背景 |
| `EXPERIMENT_PLAN.md` | 完整实验计划：数据集、评估协议、消融矩阵 |
| `README.md` | 旧项目README |
| `prompt_theories.py` | **5种理论的prompt定义**（Generic/ITT/IET/ATT/SCM），生成结构化JSON |
| `gen_theory_embeddings.py` | BERT编码脚本：将LLM输出的文本字段编码为768d embedding |
| `run_dual_pool_ensemble.py` | 双池化ensemble尝试（未成功） |
| `run_supproto_search.py` | SupProto超参搜索脚本 |

---

## v1_scm_method/ — SCM-MoE方法（Review Rounds 1-4，最终6.5/10）

以SCM（Stereotype Content Model）为核心理论，Q-MoE为架构的统一方法。

### code/

| 文件 | 说明 |
|------|------|
| `main_scm_qmoe_qels_mr2.py` | **主模型**：SCM + Q-MoE + QELS + MR2。双warmth/competence流 → 4象限composer → 4专家MoE |
| `main_scm_qmoe_qels.py` | 无MR2的baseline版本 |
| `main_scm_qmoe_qels_meanpool.py` | Mean pool变体（无CLS） |
| `main_scm_qmoe_qels_caf.py` | +CAF融合变体（QMF/DBF/PDF/Gate四种credibility-aware fusion） |
| `main_scm_qmoe_qels_sp2.py` | +SupProto v2（CVPR 2025） |
| `main_scm_qmoe_qels_vib.py` | +Variational Information Bottleneck（ICLR 2025） |
| `main_scm_qmoe_qels_gca.py` | +Generalized Class-Aware Loss（NeurIPS 2025） |
| `main_scm_qmoe_qels_mixup.py` | +Mixup增强 |
| `main_scm_qmoe_qels_rank.py` | +Ranking loss |
| `main_scm_qmoe_qels_sp.py` | +SupProto v1 |
| `main_scm_qmoe_qels_aux.py` | +辅助loss变体 |
| `main_scm_qmoe_scl.py` | +Supervised Contrastive Learning |
| `main_scm.py` | 最早的SCM模型（无Q-MoE） |
| `main_scm_qmoe.py` | SCM + Q-MoE（无QELS） |
| `main_scm_hetmoe.py` | SCM + Heterogeneous MoE变体 |
| `main_scm_v2_qmoe_qels.py` | SCM v2（增加endorsement_context字段） |
| `main_itt.py` | **ITT模型**：4通道threat gating架构 |
| `ablation_itt.py` | ITT消融实验runner（也用于ITT seed search） |
| `ablation_itt_v2.py` | ITT消融v2 |
| `ablation_itt_v3.py` | ITT消融v3 |
| `ablation_scm.py` | SCM消融实验runner |
| `ablation_scm_v2.py` | SCM消融v2 |
| `ablation_scm_v3.py` | SCM消融v3（最终版） |
| `run_ablations.py` | **统一消融runner**：支持所有ablation variants (base_only, generic_prompt, drop_field等) |
| `run_analysis.py` | 理论一致性分析（象限分布、专家特化、entropy相关性） |
| `run_moe_variants.py` | MoE路由变体对比（top-k, soft, hash等） |
| `compute_stats.py` | 结果统计计算 |
| `leakage_audit.py` | 字段泄露审计（单字段LR + tuned MLP + MLLM直接分类） |
| `gen_theory_embeddings_sbert.py` | SBERT编码变体（未采用） |
| `prompt_theories_v2.py` | Prompt v2（更详细的指令） |

### results/

| 目录 | 说明 |
|------|------|
| `seed_searches/scm_mr2/` | **SCM+MR2主结果**：200 seeds × 多种(α,β)配置 × 4数据集 |
| `seed_searches/scm_qels/` | SCM+QELS（无MR2）baseline |
| `seed_searches/scm_qmoe/` | SCM+Q-MoE（无QELS/MR2） |
| `seed_searches/scm_base/` | 早期SCM |
| `seed_searches/scm_caf/` | CAF融合变体结果 |
| `seed_searches/scm_gca/` | GCA loss变体结果 |
| `seed_searches/scm_sp2/` | SupProto v2结果 |
| `seed_searches/scm_vib/` | VIB变体结果 |
| `seed_searches/scm_*` | 其他变体（mixup, rank, mp, aux, scl, hetmoe等） |
| `ablations/ablation_results/` | **完整消融结果**：42个子目录，涵盖所有component removal |
| `refine_loop/` | 16轮自动refine loop的完整记录 |
| `transfer/` | 跨数据集transfer实验（HateMM↔MHC-EN） |
| `logs/` | 32K+训练日志文件 |
| `analysis/` | Leakage audit结果JSON |

### 文档

| 文件 | 说明 |
|------|------|
| `ABLATION_RESULTS.md` | 消融实验结果汇总表 |
| `BEST_PIPELINES.md` | 各数据集最优pipeline配置 |
| `ablation_table.tex` | LaTeX格式消融表 |

---

## v2_multi_theory/ — 多理论框架（Review Rounds 5-7，最终8.1/10）

从SCM-only pivot到5理论对比框架。

### code/

| 文件 | 说明 |
|------|------|
| `screen_theories.py` | **Phase A理论筛选**：相同Fusion架构下对比5种理论（capacity-matched） |
| `nli_validation.py` | SCM字段的NLI外部验证（DeBERTa-v3-large） |
| `nli_validation_itt.py` | ITT字段的NLI外部验证 |
| `run_transfer.py` | 跨数据集transfer实验脚本 |
| `run_phase_a.sh` | Phase A批量运行脚本 |

### results/

| 目录 | 说明 |
|------|------|
| `screening/screen_results/` | **5理论×3数据集筛选结果**（same architecture） |
| `itt/` | **ITT 200-seed结果**：HateMM/MHC-EN/MHC-ZH（4-channel架构） |
| `nli_validation/` | SCM和ITT的NLI验证结果JSON |

### 文档

| 文件 | 说明 |
|------|------|
| `review_docs/AUTO_REVIEW.md` | 7轮review完整记录（含debate transcript） |
| `review_docs/REVIEWER_MEMORY.md` | GPT reviewer的持久记忆 |
| `review_docs/REVIEW_STATE.json` | Review loop最终状态（8.1/10, weak accept） |
| `PAPER_OUTLINE.md` | 旧版paper outline |

---

## moe_refs/ — MoE参考实现

| 目录 | 说明 |
|------|------|
| `DynMoE/` | Dynamic MoE（含DeepSpeed 0.9.5） |
| `MH-MoE/` | Multi-Head MoE |
| `MomentumSMoE/` | Momentum Sparse MoE |
| `ReMoE/` | ReMoE（含Megatron-LM） |
| `soft-mixture-of-experts/` | Soft MoE |
| `soft-moe-pytorch/` | Soft MoE PyTorch实现 |
| `stablemoe/` | StableMoE |

---

## 可复用数据

以下数据在新方法开发中可直接使用：

- **LLM输出**：`datasets/*/scm_data.json`, `itt_data.json`, `generic_data.json` 等（已生成，不需重新query）
- **BERT Embeddings**：`embeddings/*/scm_mean_*`, `itt_mean_*`, `generic_mean_*`（已生成）
- **Base模态特征**：text/audio/frame features（symlink from EMNLP2026）
- **数据集splits**：`datasets/*/splits/train.csv|valid.csv|test.csv`
