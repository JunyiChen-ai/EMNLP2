# prompt_paradigm v6 — Coarse Axes Prompt (CAP)

> **POST-HOC CORRECTION NOTICE (2026-04-13 later session, root-cause pass)**: this
> proposal cites "v3 p_evidence ZH 0.8188" as a prior-art target. The root cause is
> NOT an atom-level phantom — v3's `polarity_calibration.py:83` deleted the sentence
> `"You are a content moderation analyst."` from the user message, so v3 Call 1 is
> scored on a different prompt. 78% of EN videos and 73% of ZH videos differ from
> baseline (max |Δ|=0.2151). Clauses P2 and Ablation B here are against a shifted-
> prompt target, not a perturbation of baseline. See STATE_ARCHIVE §"v3 p_evidence
> row correction".

**Author:** prompt-paradigm (Teammate A)
**Date:** 2026-04-13
**Status:** Gate 1 PROPOSAL — awaiting team-lead approval
**Predecessor:** v5 Per-Rule Disjunction Readout — REJECTED at Gate 2 (structural finding: per-rule variance ~100x below threshold; rule-1 monopoly pattern identical on EN and ZH → constrained-decode position-1 bias, not rule-specific reasoning)

---

## TL;DR

**One input-side change, nothing else.** Replace the verbatim 9-rule (YouTube) / 8-rule (Bilibili) rule list inside the frozen `BINARY_PROMPT` with a platform-derived 2-axis policy statement ("target × hostility"), keep the rest of the call identical to the 2B binary baseline, and score the single Yes/No with the same logprob extraction as `src/score_holistic_2b.py`. One call. One forward pass. One threshold. No fusion, no aggregation, no decomposition, no ensembling. The only lever is what the model reads before deciding.

---

## 1. Story (4-point)

### 1.1 Phenomenon
2B's latent policy-evaluation circuit is COARSER than the platform's verbatim rule list. v5 gave us the empirical anchor: per-rule train-set variance across positions is 8.87e-5 (EN) / 9.53e-5 (ZH) — two orders of magnitude below the 0.01 threshold we pre-registered. Rule 1 holds 10× the mass of rules 2-K on EN and 3× on ZH, and the monopoly is identical across two completely different rule sets. This pattern is NOT explainable by hate content; it is explainable by decode position. But the same finding can be restated positively:

> At 2B, the single binary decision is not computed by per-rule evaluation followed by disjunction. It is computed by a SINGLE coarse assessment whose internal granularity is far below K=9 or K=8.

If the model's internal policy circuit is coarse, presenting it with a fine-grained 9-item list forces it to spend attention on differentiations it cannot actually perform, which leaks noise into the one binary position we read.

### 1.2 Mechanism
The frozen YouTube rule list decomposes — syntactically and semantically — into `[TARGET: protected group]` × `[ACTION: violence | hatred | dehumanization | inferiority | supremacism | conspiracy | denial]`. Every one of the 9 rules instantiates this schema. The Bilibili rule list decomposes analogously into `[TARGET: protected group OR person OR circumstance]` × `[ACTION: discrimination | abuse | hostility | mocking]`. Both platforms' rule lists are, operationally, 2-dimensional taxonomies flattened into K bullet points for human readability.

The v6 mechanism: rewrite the rules block as the 2-axis statement the platform's own rules ALREADY encode, leaving everything else — frames, title, transcript, question wording, answer format, decoding parameters — unchanged. The model sees policy at its native granularity. The binary Yes/No position is now a projection of a 2-way decision the model can actually compute, not a disjunction-over-9 the model has been silently collapsing anyway.

### 1.3 Prediction (falsifiable)
- **P1 — Unified cell strict-beat.** v6 produces a single TR/TF × Otsu/GMM unified cell where EN ACC/mF1 strict-beats 0.7640/0.6532 AND ZH ACC/mF1 strict-beats 0.8121/0.7871.
- **P2 — Prior-art strict-beat.** v6 oracle strict-beats v3 p_evidence on BOTH datasets (EN > 0.7702, ZH > 0.8188).
- **P3 — Gain is larger where the collapse is tighter.** EN collapses 9 → 2; ZH collapses 8 → 2. EN should show a larger absolute gain than ZH.
- **P4 — Axes content is load-bearing.** An explicit length-matched non-taxonomic control (2 lines of platform-flavored safety language that do NOT encode the target×hostility schema) must fail to beat baseline. This isolates "axes content" from "shorter prompt".
- **P5 — Per-rule variance artifact persists under v6.** If v6 works, re-running the v5 per-rule readout on top of v6-style prompting should STILL show position-1 bias — the v5 finding was about decode geometry, not about rule information content. (Diagnostic, not a pass/fail clause.)

### 1.4 Falsifying counterfactual
v6 FAILS its story if ANY of these happens:
- (a) **Length is the lever, not content.** v6 beats baseline but the length-matched non-taxonomic control ALSO beats baseline by a comparable amount. Conclusion: we found a prompt-length hack, not a policy-representation insight. REJECTED under AP2.
- (b) **Both datasets regress.** v6 fails to strict-beat on either EN or ZH. Conclusion: 2B's internal policy circuit is NOT the target×hostility schema; the v5 variance was a decoder artifact unrelated to a coarser representation. REJECTED honestly.
- (c) **Only one dataset wins.** v6 strict-beats on one dataset and regresses on the other. The "shared latent policy representation" story dies (gain should be predictable from collapse ratio, not from language). REJECTED.
- (d) **v3 p_evidence still wins.** Even if v6 beats baseline, if v6 < v3 p_evidence on either dataset, the contribution is smaller than already-known prior art. REJECTED.

---

## 2. Rule-compliance audit

### 2.1 1-call hard constraint (carried from v5)
**PASS.** v6 is literally one call to Qwen3-VL-2B-Instruct, one vLLM `chat` invocation per video, one single-token greedy decode with logprobs=20, one Yes/No extracted. No observer/judge split, no per-rule decomposition, no polarity fusion, no modality split. Fewer moving parts than v3/v4/v5.

### 2.2 AP1 (no ensembling / repeated queries)
**PASS.** The design has a single named role: "platform-native binary moderation". Not K samples of one prompt, not multiple prompts averaged, not any form of self-consistency. Removing the one call removes the method entirely — there is nothing to ensemble over.

### 2.3 AP2 (no engineering tricks without a hateful-video story)
**PASS conditional on the length-matched control clause.** The phenomenon (coarse internal policy circuit) is anchored empirically in the v5 per-rule variance. The mechanism (re-state the rules at the model's native granularity) is mechanistically specific. The falsifying prediction P4 is the explicit check that the gain is not just "shorter prompts help" — clause C in §5 implements this.

### 2.4 AP3 (no external datasets)
**PASS.** The 2-axis taxonomy is DERIVED BY ME from re-reading the existing frozen YouTube/Bilibili rule lists. No auxiliary hate dataset, no Hateful-Memes warmup, no retrieval, no external lexicon. The taxonomy is a constant string embedded in the v6 scorer; I will paste the text directly into the proposal below so it is auditable and frozen.

### 2.5 Frozen-file rule
**PASS.** `src/score_holistic_2b.py` is NOT edited. A new scorer at `src/prompt_paradigm/coarse_axes_prompt.py` replicates the message-building pattern from `src/prompt_paradigm/modality_split.py` but with the coarse-axes string instead of YOUTUBE_RULES/BILIBILI_RULES. Evaluator at `src/prompt_paradigm/eval_coarse_axes.py`. Pipeline runner at `src/prompt_paradigm/run_v6_pipeline.sh` (reuses v5 runner pattern).

### 2.6 Label-free integrity
**PASS.** No test labels are read. The only pieces of the target benchmark used are: (a) the clean-split video IDs (for iteration order), (b) the frames/title/transcript of each video, and (c) the platform's own rule lists (which are part of the benchmark spec, not labels). The collapsed taxonomies are authored by me before any v6 test-set score is computed.

---

## 3. Design

### 3.1 Single call specification
Unchanged from `src/score_holistic_2b.py` baseline EXCEPT the `rules_text` string:

- Model: `Qwen/Qwen3-VL-2B-Instruct`, bf16, vLLM, `temperature=0`, `logprobs=20`, `max_tokens=1`, `allowed_token_ids=union(Yes,No)`.
- `limit_mm_per_prompt={"video":1,"image":8}`, `allowed_local_media_path="/data/jehc223"`, `mm_processor_kwargs={"max_pixels":100352}`, `max_model_len=32768`, `gpu_memory_utilization=0.92`.
- System: `"You are a content moderation analyst."`
- User prompt template: reuse frozen `BINARY_PROMPT` verbatim (the template that interpolates `{title}`, `{transcript}`, `{rules}`). The ONLY change: the `rules` argument.
- Media content: frames or video URL, same as `build_media_content` in `score_holistic_2b.py`.
- Score: `p_yes / (p_yes + p_no)` from position-0 logprobs. One number per video. Written as `{"video_id": str, "score": float}` for direct compatibility with `src/quick_eval_all.py::load_scores_file`.

### 3.2 The coarse-axes strings (frozen, auditable, no external data)

**YOUTUBE_COARSE_AXES** (applied for `MHClip_EN`):
```
1. The content identifies a target that is a group defined by a protected attribute (race, ethnicity, national origin, religion, sexual orientation, gender identity, disability, or similar).
2. The content expresses hostility toward that target through violence, incitement, dehumanization, inferiority claims, supremacist framing, conspiratorial attacks, denial of past harms, or slurs and stereotypes.
```

Derivation note: axis 1 is the shared subject of every YouTube rule ("groups based on protected status"). Axis 2 enumerates, verbatim-paraphrased, the action vocabulary of rules 1–9 (violence/incite/dehumanize/inferiority/supremacism/conspiracy/deny/slurs).

**BILIBILI_COARSE_AXES** (applied for `MHClip_ZH`):
```
1. The content identifies a target that is a group defined by a protected attribute, an identifiable person, or a circumstance involving death, sickness, disability, or disaster.
2. The content expresses hostility toward that target through discrimination, personal attacks, abuse, hateful ideologies or symbols, mocking, or incitement to confrontation.
```

Derivation note: axis 1 combines the subject scope of Bilibili rules 1, 2, 4, 7 ("protected groups"), rule 3 ("personal identity"), and rules 5, 6 ("circumstances of death/sickness/disability/disaster"). Axis 2 is the action vocabulary of rules 1–8 (discrimination/attack/abuse/ideology/mocking/incitement).

Both strings are embedded as module-level constants in `src/prompt_paradigm/coarse_axes_prompt.py`. Their content is frozen at proposal time. If Gate 2 regresses, I do NOT tune the strings and re-run — the proposal is a single binding shot.

### 3.3 What happens to the binary question?
UNCHANGED. The question is still `"Does this video contain content that violates any of the above rules?"` from frozen `BINARY_PROMPT`. The word "rules" now refers to the 2-axis statement instead of the 9/8-item list. The yes/no answer format and extraction logic are identical to the baseline.

### 3.4 Aggregation / fusion / thresholding
NONE. There is nothing to aggregate — one call produces one score. Threshold selection uses the unified TR/TF × Otsu/GMM family via `src/quick_eval_all.py`, same as every prior iteration.

---

## 4. Implementation plan

### 4.1 Files to create (all new; no frozen file edits)
1. `src/prompt_paradigm/coarse_axes_prompt.py` — single-call scorer. Pattern mirrors `modality_split.py` but one call per video instead of two. Module constants `YOUTUBE_COARSE_AXES` and `BILIBILI_COARSE_AXES` hold the frozen taxonomy strings. Selects axes by `args.dataset`. Reuses `build_binary_token_ids`, `extract_binary_score`, `build_media_content` patterns from `modality_split.py`. Output JSONL `{"video_id", "score"}`. Supports `--unconstrained` (not expected to be needed; 1-token decode is trivial), `--limit` for pilot, `--out-suffix`.
2. `src/prompt_paradigm/eval_coarse_axes.py` — evaluator with the 8 binding clauses listed in §5. Reuses `otsu_threshold`, `gmm_threshold`, `metrics`, `load_scores_file`, `build_arrays` from frozen `src/quick_eval_all.py`. Loads baseline (from `results/holistic_2b/*/test_binary.jsonl` + `train_binary.jsonl`), v3 p_evidence (from `results/prompt_paradigm/*/test_polarity.jsonl`), and v6 coarse-axes scores. Writes `results/prompt_paradigm/report_v6.json` and appends a `v6 — Coarse Axes Prompt` section to `results/analysis/prompt_paradigm_report.md`.
3. `src/prompt_paradigm/run_v6_pipeline.sh` — pipeline runner following v5 structure: 3 waves (W1 test, W2 train, W3 CPU eval), own-ID polling every 30s, re-startable (skips waves whose outputs already exist), logs all job IDs to `docs/experiments/prompt_paradigm_runs.md`. Reuses the v5 runner template with s/v5/v6/g.

### 4.2 Length-matched non-taxonomic control (clause C)
A frozen constant `YOUTUBE_LENGTH_CONTROL` / `BILIBILI_LENGTH_CONTROL` inside the same scorer, matched line-for-line and approximately character-for-character to the coarse-axes strings, but the content is boilerplate safety language (e.g., "This platform's policies exist to protect user wellbeing and to foster respectful discourse. Users are asked to consider the impact of their content before publishing."). Authored at proposal time, frozen at proposal time.

The control is run via `--control` flag on the same scorer. It produces a second JSONL file `test_coarse_axes_control.jsonl` (and train). Clause C in §5 compares v6 vs this control.

### 4.3 Pilot
Before the full pipeline, a 10-video pilot on EN test (same pattern as v5 pilot) verifies the scorer runs, loads the frozen constants, and produces non-null scores for the coarse-axes and control conditions. Expected wall clock for the pilot: ~2 minutes. Pilot success criterion: 10/10 valid scores on both conditions.

### 4.4 Expected wall clock
- W1 test (EN 161 + ZH 157 scored per condition × 2 conditions): ~1 hour (2 concurrent GPU jobs, 2 conditions run sequentially within each dataset or as separate jobs — see runner).
- W2 train (EN 550 + ZH 579 per condition × 2 conditions): ~1.5-2 hours.
- W3 eval: ~5 minutes.
- **Total: ~3 hours.** Within v5 budget.

Trade-off: running the control adds ~50% wall-clock over v5. If team-lead prefers to skip the control, I can merge clauses A and C by requiring v6 to strict-beat the known-winners-baseline-and-v3-prior-art suite AND re-test on a different model (e.g., swapping the 8B safety-asymmetry result as an implicit control). But I do not recommend dropping the control — clause C is the cleanest AP2 defense, and the prompt-length-only null is the most plausible confounder to rule out.

---

## 5. Gate 2 binding clauses (8, all must pass for ACCEPT)

All clauses below are pre-registered. No clause added or removed after this proposal is sent. `all_clauses_pass = all(c.pass for c in clauses)`.

| # | Clause | Pass criterion |
|---|---|---|
| 1 | **Oracle-first** | v6 test oracle (label-sweep diagnostic) strict-beats baseline oracle on BOTH datasets. EN > 0.7764 AND ZH > 0.8121. If this fails, v6 is rejected before threshold-family search. |
| 2 | **mF1 non-regression, unified cell** | There EXISTS one TR/TF × Otsu/GMM label (same on both datasets) whose v6 score produces EN ACC ≥ 0.7640 AND EN mF1 ≥ 0.6532 AND ZH ACC ≥ 0.8121 AND ZH mF1 ≥ 0.7871, with strict-beat on at least one metric per dataset. |
| 3 | **Ablation A — baseline load-bearing** | v6 oracle > baseline oracle on BOTH (i.e., strict inequality; no leak). |
| 4 | **Ablation B — prior-art strict-beat** | v6 oracle > v3 p_evidence oracle on BOTH (EN > 0.7702, ZH > 0.8188). |
| 5 | **Ablation C — axes-content load-bearing (the critical AP2 clause)** | `length-matched non-taxonomic control` oracle must NOT strict-beat baseline on either dataset. If the control also beats baseline, v6's gain is confounded with prompt length, and v6 is REJECTED even if clauses 1–4 pass. |
| 6 | **Prediction P3 — collapse-ratio monotonicity (diagnostic)** | EN absolute gain (v6 oracle − baseline oracle) ≥ ZH absolute gain (v6 oracle − baseline oracle), reflecting that 9→2 collapse is tighter than 8→2. This is a directional sanity check for the story. Failure does NOT automatically reject v6, but I must explain any inversion in the Gate 2 writeup. |
| 7 | **Format compliance** | `n_null_scores / n_total ≤ 0.05` on both datasets. Since this is a 1-token decode with `allowed_token_ids`, null should be ~0%. This clause just asserts the scorer did not silently drop videos. |
| 8 | **n_test reconciliation** | EN test scored = 161, ZH test scored = 149 (matching baseline). If fewer, explain the drop. |

Clauses 1, 2, 3, 4, 5, 7, 8 are strict pass/fail. Clause 6 is directional diagnostic. `all_clauses_pass` is the strict AND of clauses 1, 2, 3, 4, 5, 7, 8.

---

## 6. Relationship to v1-v5 (failure continuity)

| v | Lever | Output-side or Input-side? | Falsified mechanism |
|---|---|---|---|
| v1 | 2-call observer→judge text cascade | output (call structure) | text-cascade drift; Observer narrative degrades under 2B |
| v2 | AND-gate of two binary calls | output (decision fusion) | AND-gate compression kills recall |
| v3 | Polarity-flip logit fusion | output (answer-space manipulation) | Call 2's bias not opposite-signed enough at 2B; prob-avg ≥ fused (AP1 self-binding) |
| v4 | Modality split + rank-noisy-OR | output (aggregation) | Joint prompt already exploits cross-modal complementarity better |
| v5 | Per-rule K-position readout | output (decoding structure) | Position-1 decoder bias; rules 2-K indistinguishable within each dataset |
| **v6** | **Coarse axes prompt (replace rules block)** | **INPUT (conditioning)** | TBD — falsifier is clauses 5 (axes-content load-bearing) and 2 (unified non-regression) |

Five failed levers on the output side; v6 pivots to the input side. Inside the input side, v6 narrows further to the rules text specifically — the ONE element of the input that v1-v5 all preserved verbatim. If v6 fails, the natural next narrowing is to other input elements (transcript preprocessing, frame selection, question reformulation).

---

## 7. Gate 1 acceptance request

I am asking for Gate 1 approval of v6 as specified above. If approved, I will:
1. Implement the three files (`coarse_axes_prompt.py`, `eval_coarse_axes.py`, `run_v6_pipeline.sh`).
2. Run the 10-video pilot for both conditions.
3. Launch the v6 pipeline via the runner.
4. Report Gate 2 with the 8-clause verdict table.

If the team-lead wants the length-matched non-taxonomic control strings reviewed before I freeze them, I will send the proposed strings as a separate message before starting implementation so the "frozen at proposal time" constraint is honored with explicit team-lead sign-off.

---

## 8. Open questions for the team-lead (non-blocking)

1. Is the clause 5 (length-matched control) wall-clock trade-off acceptable? (v6 runs ~50% longer than v5 because of the control.) I strongly recommend keeping it — it is the cleanest AP2 defense.
2. Do you want to audit the coarse-axes strings for EN and ZH before I commit them to code, or is the proposal-level text (§3.2) sufficient to treat them as frozen?
3. Is the runner script reuse sufficient, or should I commit a stricter-ledger variant (e.g., per-wave heartbeats to the run log)?

Ready to proceed on your signal.
