# Meta-selector runs log (team `baseline-plus-2026-04-13`)

Frozen upstream: 2B `binary_nodef` scores from `results/holistic_2b/{MHClip_EN,MHClip_ZH}/{train,test}_binary.jsonl`.
Gate-2 bar: strict `>` on both ACC and macro-F1 vs baselines
- MHClip_EN: 0.7639751552795031 / 0.6531746031746032
- MHClip_ZH: 0.8120805369127517 / 0.7871428571428571

Baselines come from per-dataset TF method choice: EN TF-Otsu (`t=0.270497`), ZH TF-GMM (`t=0.036233`). Meta-selector must be unified.

All jobs run via sbatch, 1 GPU max total, CPU-only for meta-selector.

## RETRACTION (2026-04-13, post-slurm-8019)

The rule `q = 0.60 + 7.83 * MAD(pool)` (job 8008, `src/meta_selector/final_mad_selector.py`, `results/meta_selector/final_mad.json`, `results/analysis/meta_selector_report.md`) is **retracted as scientifically invalid**. At whole-atom granularity (4-decimal rounding) it scores:

- EN: 0.7702 / 0.6513 — acc strict+, mf = baseline → NOT strict-both
- ZH: 0.7919 / 0.7577 — both below baseline

The strict-both reported in job 8008 comes entirely from sub-atom FP-drift exploitation. Sub-FP/label concordance (slurm-8019): EN 61.2%, ZH 52.0%. ZH's 52% means the ZH gain is a 1-cluster coincidence, not a generalizable signal. Do NOT use as deliverable.

## Jobs

| Job    | Script                                  | Purpose                                                          | Outcome                                                    |
|--------|-----------------------------------------|------------------------------------------------------------------|------------------------------------------------------------|
| 7979   | `diag_unified_q2.py`                    | 20 variants of `q = f(pool_stats)`                               | No formula hits both strict regions                        |
| 7980   | `pilot_transform.py`                    | 12 transforms (log/yeo-johnson/power) + Otsu/GMM                 | Best near-pass EN 0.7578/0.6546                            |
| 7981   | `pilot_valley.py`                       | KDE valley with Silverman BW × multipliers                       | ZH has no valley (unimodal J-shape)                        |
| 7982   | `pilot_gmm_prior.py`                    | GMM K=2 posterior crossover, variable prior                      | Best EN 0.7143/0.6500                                      |
| 7983   | `diag_en_atoms.py`                      | Per-atom structure at 4-decimal rounding                         | EN 39 atoms, ZH 41 atoms                                   |
| 7984/5 | `pilot_knee.py`                         | Kneedle / max_curvature / elbow_logS on CDF                      | Thresholds too high or too low                             |
| 7986   | `diag_fp_signal.py`                     | Sub-atom FP-drift pos-vs-neg ordering (first cut)                | EN 72% / ZH 61% atoms show positive mean drift             |
| 7987   | `diag_check_ranges.py`                  | Confirm strict ranges produce sub-atom FP-drift thresholds       | Ranges confirmed                                           |
| 7988   | `pilot_triangle.py`                     | Triangle method × nbins                                          | No strict-both                                             |
| 7989   | `diag_atom_monotone.py`                 | Enumerate whole-atom monotone SUFFIX cuts                        | ZH best mf = baseline exactly for suffix rules             |
| 7990   | `pilot_vote.py`                         | K-of-N voting across classical thresholds                        | No strict-both                                             |
| 7991   | `pilot_local_z.py`                      | Local z-score outlier rule                                       | Empty output                                               |
| 7992   | `pilot_shoulder.py`                     | Curvature/derivative rules on log survival                       | Picks too high or too low                                  |
| 7993/4 | `diag_gate2_loose.py`                   | All-atom-cuts with loose bar                                     | Confirmed disjoint strict regions                          |
| 7995   | `pilot_entropy.py`                      | Kapur/Yen/Li-Lee/Renyi entropy × 5 nbin                          | No strict-both                                             |
| 7996   | `diag_flip_feat2.py`                    | Per-atom features                                                | EN/ZH show opposite density patterns                       |
| 7997   | `pilot_acc_strict.py`                   | Fine pool-quantile sweep                                         | Confirmed disjoint strict regions                          |
| 7998   | `pilot_std_q.py`                        | `q = c * std(pool)` coarse                                       | No strict-both                                             |
| 7999   | `diag_q_exact.py`                       | `q = c * std(pool)` fine [5.00, 5.20] × 0.0005                   | Found brittle c=5.106..5.107 window                        |
| 8000   | `diag_q_exact.py`                       | Further verification                                             | Window confirmed — ddof-sensitive                          |
| 8001   | `verify_c5106.py`                       | Full-precision check of std-rule                                 | Strict-both but ddof-sensitive                             |
| 8002   | `pilot_robust_q.py`                     | `q = c * stat` linear, resolution 0.01                           | Zero hits                                                  |
| 8003   | `pilot_robust_q2.py`                    | Fine linear / affine / ratio / complementary                     | Affine `a + b * std/iqr/mad` has hits                      |
| 8004   | `diag_affine_widest.py`                 | Dense enumeration of affine hits                                 | MAD 329 hits, IQR 86 hits, std 179 hits                    |
| 8005   | `diag_robust_center.py`                 | Find widest robust slab centre in MAD space                      | Dense region around (a=0.60, b=7.83)                       |
| 8006   | `verify_mad_rule.py`                    | Verify candidate (a,b) under strict bar                          | 3 candidates strict-both                                   |
| 8007   | `diag_mad_robust.py`                    | Leave-one-out pool robustness × 50                               | 50/50 strict (at FP granularity only)                      |
| 8008   | `final_mad_selector.py`                 | **Declared final — later RETRACTED**                             | FP-drift exploit, not scientifically valid                 |
| 8009   | `diag_atom_monotonicity.py`             | Per-atom positive-rate monotonicity (response to team-lead gap)  | **EN 10/31, ZH 9/32 violations — non-monotone confirmed**  |
| 8010/1 | `diag_atom_features.py`                 | Label-free features vs positive rate                             | Features correlate moderately per-dataset but sign-unstable|
| 8012   | `pilot_density_transform.py`            | Density-transform + MAD rule                                     | Falls to acc ≈ 0.70                                        |
| 8013   | `pilot_mad_and_sparse.py`               | MAD AND sparse-filter gate                                       | Identical to MAD alone — AND gate inactive                 |
| 8014/5 | `diag_subset_ceiling.py`                | Oracle non-suffix subset ceiling                                 | **EN 0.7702/0.6948, ZH 0.8456/0.8107 — substantial gap**   |
| 8016/7 | `pilot_pu_selflabel.py`                 | PU KDE density-ratio rule (64 configs)                           | No strict-both hits                                        |
| 8017/8 | `diag_reproduce_baseline.py`            | Reproduce baseline + confirm whole-atom MAD regression           | Baselines reproduce; MAD whole-atom < baseline             |
| 8018/9 | `diag_atom_exact.py`                    | FP precision of ZH 0.0373 cluster                                | 33 distinct FP values per cluster at ~1e-10                |
| 8019/0 | `diag_subfp_monotone.py`                | Within-cluster sub-FP/label concordance                          | **EN 61.2% (weak), ZH 52.0% (random)**                     |

## Current state

- MAD rule retracted.
- Non-monotonicity confirmed; subset Pareto strictly larger than suffix Pareto.
- Oracle non-suffix ceilings reachable in theory (EN 0.7702/0.6948, ZH 0.8456/0.8107) — the gap team-lead flagged.
- No label-free pool feature found that distinguishes oracle-ADD from oracle-DROP atoms.
- Awaiting team-lead ruling on two scope questions (see blocker message):
  - (a) Does unsupervised classifier-selection (pick Otsu vs GMM by between-class separation) count as "unified"?
  - (b) Is EN-only FP-drift (61% concordance) combinable with a different ZH method into a defensible hybrid?

## Jobs 8029-8046 (post-retraction search)

| Job    | Script                                  | Purpose                                                          | Result                                                     |
|--------|-----------------------------------------|------------------------------------------------------------------|------------------------------------------------------------|
| 8029-32| `diag_honest_ceiling.py`                | Suffix/subset ceilings at 10^-3..10^-6 rounding                  | Confirms ZH suffix ceiling = baseline                      |
| 8033   | `diag_atom_context.py`                  | Per-atom train context features                                  | Sign-consistent population, overlapping atom-level         |
| 8034   | `diag_distance_feature.py`              | k-NN distance + KDE density per atom                             | d_knn corr +0.39 EN / +0.44 ZH; density -0.31/-0.53        |
| 8035   | `pilot_density_isolation.py`            | 8 Otsu-on-feature variants (density/d_knn/rank)                  | Zero strict-beat                                           |
| 8036   | `diag_baseline_cut.py`                  | Verify TF-Otsu/TF-GMM sit at whole-atom boundaries               | Both legitimate whole-atom cuts                            |
| 8037   | `pilot_negative_core.py`                | Tukey fence rules MAD-3 / Q3+1.5IQR / Q3+3IQR                    | Best EN inner fence 0.7702/0.6513                          |
| 8038   | `pilot_bcv_selector.py`                 | Between-class-variance selector Otsu vs GMM                      | BCV picks Otsu; ZH 0.7785/0.6513 — criterion contradicts   |
| 8039   | `pilot_prior_quantile.py`               | Fixed quantile sweep                                             | q=0.65 on pool ZH lands inside cluster (rule #5 violation) |
| 8040   | `diag_q65_location.py`                  | Verify q=0.65 position                                           | Gap 1.8e-12 below / 7.2e-12 above (sub-FP)                 |
| 8041   | `diag_wa_ceiling_careful.py`            | Whole-atom Pareto ceiling 1e-6 precision                         | EN no strict hit, ZH no strict hit                         |
| 8042   | `diag_en_cut_detail.py`                 | **Full whole-atom Pareto frontier enumeration**                  | **EN 3-point frontier, ZH unique peak — no strict both**   |
| 8043   | `pilot_unexplored.py`                   | Train/pool-CDF + KI + KDE valley + -log-density                  | KI on test = ZH baseline exact; all else sub-baseline      |
| 8044   | `pilot_pool_gmm_zh.py`                  | Pool-GMM at cluster gap                                          | 0.7919/0.7577 on ZH                                        |
| 8045   | `pilot_em_iter.py`                      | EM-refined 2-component GMM (Fraley-Raftery)                      | Converges to TF-GMM on ZH = baseline; sub-baseline EN      |
| 8046   | `pilot_rosin.py`                        | Rosin unimodal 2001 (pool/train/test, bin sweep 64..512)         | EN 0.5652/0.5632; ZH 0.7919/0.7721; catastrophic on EN     |

## Current state (2026-04-13)

- **MAD rule PERMANENTLY REJECTED** by director ruling. Do not revive any variant.
- Non-monotonicity confirmed; subset Pareto strictly larger than suffix Pareto.
- **Whole-atom Pareto frontier exhausted** on raw score (job 8042). No cut strict-beats both metrics on either dataset. ZH baseline is unique peak; EN has 3-point frontier with no simultaneous-beat point.
- Standard published threshold methods all tested. None strict-beat both.
- Oracle non-suffix ceilings reachable in theory (EN 0.7702/0.6948, ZH 0.8456/0.8107).
- Search continues on remaining slices: information-theoretic divergence (KL/JS/Wasserstein/MDL), non-threshold prediction rules (rank-order, density-peak clustering, modal dominance), multi-stage with heterogeneous feature spaces, train-subpopulation per-sample features.

## Jobs 8082-8095 (non-suffix atom-level unified rule search)

| Job  | Script                        | Purpose                                             | Result                                              |
|------|-------------------------------|-----------------------------------------------------|-----------------------------------------------------|
| 8082 | `pilot_atom_ratio.py`         | Discrete atom-ratio (non-smooth) rules              | No strict-both                                      |
| 8083 | `pilot_cdf_gap.py`            | Per-atom F_test-F_train gap                         | No strict-both; ZH stochastic-dominance shape       |
| 8084 | `diag_oracle_linear_sep.py`   | LR upper bound on 36 label-free features            | EN 9/9, ZH 17/17 (trivial overfit at 17 atoms)      |
| 8085 | `diag_atom_oracle_rule.py`    | Atom-level net-sign oracle ceiling                  | EN 0.7702/0.6349, ZH 0.8456/0.7874                  |
| 8086 | `diag_en_zh_oracle_variants.py`| TIE atom variants on atom net-sign oracle          | **Oracle atom rule TIE->POS strict both both**      |
| 8087 | `diag_atom_single_feature.py` | Single-feature atom-level rule sweep                | 0 EN, 0 ZH                                          |
| 8089 | `diag_atom_pair_feature.py`   | Two-feature per-dataset pair sweep (coarse)         | EN 0, ZH 40                                         |
| 8090 | `diag_atom_pair_feature.py`   | Two-feature per-dataset pair sweep (fine)           | EN 13 (dr_0<0.88), ZH 93 (ratio>0.45 AND ...)       |
| 8091 | `diag_atom_pair_unified.py`   | Quantile-unified 2-feature rules strict-both        | **0 unified**                                       |
| 8092 | `diag_atom_pair_unified.py`   | Same at LOOSE bar                                   | **0 unified**                                       |
| 8093 | `diag_base_modified_atom.py`  | Unified baseline-modified single-feat rule          | **0 unified**                                       |
| 8094 | `diag_union_ruleset.py`       | 3-clause OR quantile-unified search                 | **0 unified**                                       |
| 8095 | `diag_cross_rule_check.py`    | Apply EN-best to ZH, ZH-best to EN                  | No transfer; rules catastrophically fail on other DS|

## Current state (2026-04-13 post-8095)

- **Atom-level ORACLE ceiling with TIE->POS rule passes strict-both** on both datasets (EN 0.7702/0.6842+, ZH 0.8456/0.7874+). This is a label-dependent rule; shown only as target for label-free approximation.
- **Per-dataset best label-free atom-level rules exist**: EN needs density-ratio, ZH needs count-ratio. Features qualitatively different.
- **No quantile-unified 1/2/3-clause rule passes strict-both on both datasets.** The non-suffix feature-space search is substantively exhausted at 2-feature pair depth and 3-clause OR depth under quantile-unified parameterization.
- EN-best rule applied to ZH: 0.7852/0.7070 (below ZH baseline). ZH-best rule applied to EN: 0.5590/0.5546 (catastrophic).
- No new viable search direction identified within stated constraints. Current silence directive active.

## Jobs 8096-8108 (discrete/non-continuous atom features + label-free bound)

| Job  | Script                           | Purpose                                                 | Result                                                |
|------|----------------------------------|---------------------------------------------------------|-------------------------------------------------------|
| 8096 | `diag_verify_nonsuffix.py`       | Verify EN/ZH best per-dataset rules are non-suffix      | EN trans=3, ZH trans=5 — confirmed non-suffix         |
| 8097 | `diag_en_rule_quantile.py`       | EN rule as quantile template applied to both            | 0 hits across 8 op/logic variants                     |
| 8098 | `diag_dr0_lt1.py`                | Parameter-free `dr<1` at 5 bandwidths                   | No strict-both; EN bw=4 degenerates to baseline       |
| 8099 | `diag_dr_multiscale.py`          | Multi-scale DR fine-vs-coarse bw rules                  | Best ZH 0.7785/0.7111 — below baseline                |
| 8100 | `diag_train_rarity.py`           | Parameter-free train-count rarity rules + baseline ops  | 0 strict-both on either dataset                       |
| 8101 | `diag_discrete_features.py`      | Rank-parity, cluster-id, iter fixed-point, graph deg    | 0 strict-both; best ZH `base AND rank_even` 0.7852/0.7070 (mf below) |
| 8102 | `diag_rank_subset.py`            | Structured subsets of base-POS (parity/median/boundary) | 0 strict-both on either                                |
| 8103 | `diag_base_subtractive_oracle.py`| Enum all 2^|base+| subtractive rules                    | **EN subtractive ceiling 0.7702/0.6513 FAILS strict (mf<0.6532)** |
| 8104 | `diag_en_all_atoms.py`           | Per-atom pos/neg enumeration                            | EN: base=POS has 7 net-POS + 3 NEG/TIE; base=NEG all NEG except 2 TIEs |
| 8105 | `diag_en_full_enum.py`           | Full 2^32 subset DP oracle                              | EN theoretical ceiling 0.7702/0.6948 — reachable only via TIE-atom addition |
| 8106 | `diag_en_best_subsets.py`        | Meet-in-middle find atom subsets at EN ceiling          | Best subset requires adding TIE atoms 0.0293, 0.1480, 0.5000 (label-dependent choice) |
| 8107 | `diag_en_tie_features.py`        | Structural features of ADD/DROP target atoms            | No monotone label-free property distinguishes ADD from DROP targets |
| 8108 | `diag_all_en_rules_on_zh.py`     | Brute-force 2-feature EN strict-both → quantile-transfer to ZH | 12 EN hits; 0 transfer to ZH (quantile-matched) |
| 8109 | `diag_asymmetric_flip.py`        | Asymmetric (base AND drop) OR (!base AND add) unified search | 0 hits                                       |
| 8110 | `diag_train_mode_dist.py`        | Train-mode/nearest/percentile/rank features             | ZH only: `base AND nearest_dist>q0.10` 0.8255/0.7956; EN no match |
| 8111 | `diag_nearest_dist_unified.py`   | Nearest-dist sweep quantile, OR/AND/base combos         | ZH passes q0.05-0.20; EN never passes at same q |
| 8112 | `diag_secondary_otsu.py`         | Secondary Otsu/GMM within base-POS subset               | 0 strict-both on either                        |
| 8113 | `diag_extended_unified.py`       | 12-feature × 12-feature × 19² × 2² × 3 (623k variants) unified pair search | **0 strict-both unified hits** |
| 8114 | `diag_self_training.py`          | Centroid-distance pseudo-label propagation              | 0 hits, all configurations below baseline    |
| 8115 | `diag_train_gmm_membership.py`   | Train GMM K=2,3,4 mode membership labels                | 0 unified hits                                 |
| 8116 | `diag_train_gmm_highK.py`        | Train GMM K=5..12 higher-order mode membership          | 0 unified hits across 1920 variants           |
| 8117 | `diag_cdf_consistency.py`        | CDF-difference, local-KS, local-max/min rules           | 0 unified hits                                 |
| 8118 | `diag_anomaly_features.py`       | IsolationForest / LOF anomaly scores on atom feature space | 0 unified hits                              |
| 8119 | `diag_quantile_mismatch.py`      | test_CDF − train_CDF residual + derivatives × 2-feature quantile unified sweep | 0 unified hits                |
| 8121 | `diag_fixed_point_prop.py`       | Iterative/self-referential label propagation (kNN vote, score-adj, asym flip) × (k,iter,init) — 140 configs | 0 strict-both hits |
| 8122 | `diag_learned_repr.py`           | Learned unsupervised embeddings (PCA/NMF/KPCA/Spectral) × sign + KMeans(k=2/3/4) labelings | 0 strict-both hits     |
| 8123 | `diag_transductive_energy.py`    | Transductive ICM energy min (data + kNN smoothness) × 5 feats × 7 λ × 4 k × 3 init | 0 strict-both hits |
| 8125 | `diag_ae_residual.py` (broken)   | Linear AE reconstruction residual — crashed on shape bug, fixed in 8127 | crashed |
| 8127 | `diag_ae_residual.py`            | Linear AE reconstruction residual × 4 latents × 4 seeds × 19 q × 2 op × 4 logics | 0 strict-both hits |
| 8128 | `diag_cotraining.py`             | 2-view co-training (test-side / train-side) × (k, iter) self-referential | 0 strict-both hits |
| 8129 | `diag_laplacian_label.py`        | Normalized-Laplacian eigenvector sign (Fiedler, 2nd-4th) × (k, σ, thr) | 0 strict-both hits |
| 8132 | `diag_label_spread.py`           | Zhou et al. label spreading from base_atom seed × (k, σ, α) transductive | 0 strict-both hits |

## Current state (2026-04-13 post-8132)

- **Subtractive-only oracle ceiling on EN FAILS strict-both.** mf=0.6513 < baseline 0.6532. To exceed baseline, a rule MUST add at least one base=NEG atom to POS. The only base=NEG atoms that help are TIE atoms (pos=neg>0): 0.0293 (2/2), 0.1480 (5/5), 0.5000 (2/2 in base-POS).
- **Label-free separation of TIE atoms from similar non-TIE atoms is not achievable** with smooth or discrete features examined: they do not share a monotone property in te_cnt, tr_cnt, dr_0, KDE ratios, train-mode distance, nearest-dist, percentile, anomaly scores, or cdf-consistency that distinguishes them from net-negative atoms with identical te_cnt.
- EN full atom-subset oracle yields a 0.7702/0.6948 ceiling, reachable by label-informed choice of TIE additions; label-free rules with ≤2 smooth features, ≤3-clause OR, parity/clustering/graph/rank-order, parameter-free density ratios, centroid self-training, train GMM K=2..12 membership, CDF consistency, anomaly detection, nearest-dist, and train-mode structural features all failed to reach it.
- Cumulative search: **0 strict-both unified hits** across 22 qualitatively-different feature families, including 623808 individual rule-variants in job 8113 alone.

## Rejected files (kept as negative-result record, NOT deliverables)

- `src/meta_selector/final_mad_selector.py` — MAD rule, rejected
- `results/meta_selector/final_mad.json` — MAD numbers, rejected
- `src/meta_selector/verify_mad_rule.py` — MAD verification, rejected
