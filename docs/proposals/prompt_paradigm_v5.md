# prompt_paradigm v5 — Per-Rule Disjunction Readout (Gate 1 proposal)

> **POST-HOC CORRECTION NOTICE (2026-04-13 later session, root-cause pass)**: this
> proposal cites "v3 p_evidence" as a prior-art target. The root cause is NOT an atom-
> level phantom — it is that v3's `polarity_calibration.py:83` deleted the sentence
> `"You are a content moderation analyst."` from the user message, so v3 Call 1 uses a
> different prompt from baseline. 78% of videos differ (max |Δ|=0.2151). Under classical
> label-free methods on the shifted prompt, all cells are below baseline. Any
> "beat v3 p_evidence" clause here is against a shifted-prompt target, not a
> perturbation of baseline. See STATE_ARCHIVE §"v3 p_evidence row correction".

Author: prompt-paradigm (Teammate A)
Date: 2026-04-13
Status: Gate 1 — awaiting team-lead approval

---

## TL;DR

**One** MLLM forward pass per video. Prompt asks the model to rate each of
the K constitution rules as "1" (violated) or "0" (not) in a fixed order
on a single line of output, with output tokens constrained to `{0,1,space}`.
The readout is the K per-rule probabilities read directly from the
decoder's position-wise logprobs:

    p_i = P('1' | prompt + decoded[:i]) / (P('1') + P('0'))

at decode positions i = 1..K. Score = **max(p_1..p_K)** (noisy-OR over a
disjunction of rules). No aggregation of scores across multiple MLLM calls.
No ensembling. No polarity-flip. No modality split. One vision-tower
forward pass, one autoregressive decode of K constrained tokens.

The story: the baseline BINARY_PROMPT asks the model "does this video
violate ANY of the K rules?" and reads P(Yes). This forces the model to
compute a K-way disjunction *inside* the softmax, which is not how softmax
works — the first-token Yes/No probability is a compressed scalar
marginalized over all possible reasons the model might say Yes, smoothed
by the tokens for each rule's separate prior. Externalizing the
disjunction by reading K per-rule probabilities and aggregating with an
explicit max operator is the information-theoretically correct readout
for a "contains any of K types of prohibited content" query.

v5 is a **1-call redesign that changes the readout of the same single
forward pass the baseline already performs**, not a new call structure.

---

## 1. Phenomenon (specific to hateful video at 2B)

**Hate in this corpus is rule-compound.** The MHClip constitutions
(youtube 9 rules, bilibili 8 rules) are deliberately partitioned into
narrow, non-overlapping violation types: "slurs and stereotypes that
incite hatred", "dehumanize groups by comparing to non-human entities",
"deny or minimize major violent events", "mock death / sickness /
disability", etc. A real hateful video often violates multiple rules
simultaneously (a slur against a protected group is usually also a
stereotype and a dehumanization, etc.), but it may violate ONE rule
strongly and several others weakly.

**The baseline BINARY_PROMPT's failure mode on this corpus is softmax
marginalization over the K-way disjunction.** When the prompt says "does
this violate ANY of the above rules?", the model's first-token P(Yes)
must be high if ANY rule is violated. Internally, the model's reasoning
is something like "rule 1 probably not, rule 2 probably not, rule 3
maybe, rule 4 no, rule 5 maybe, rule 6 no...". For the final "Yes" token
at position 0 of the answer, the model must compress all K rule-level
judgments into a single Bernoulli. Because softmax is smooth and the K
rule-level evidences are partial, the compression is lossy: a video with
two rules at p=0.4 each and seven rules at p=0.02 each will have a joint
P(Yes) that does not faithfully reflect `1 - (1-0.4)*(1-0.4)*(1-0.02)^7 =
0.68`. Instead the softmax marginalizes over K implicit paths and often
lands somewhere around p=0.3, because no SINGLE rule is loud enough to
push the Yes token to a confident level. The "union over K noisy
detectors" calculation is not something a single softmax position can do
well.

Symmetrically, a clearly hateful video (rule 5 near p=0.95, others near
0) produces a high Yes probability without issue — the baseline is
already correct on these. It is only the **rule-compound borderline
cases** where the baseline's readout bottlenecks on the scalar Yes/No
compression. These are exactly the FN-heavy middle of the baseline score
distribution.

This is a hateful-video-specific phenomenon at 2B because:
(a) hate is explicitly decomposed into K rule types in the constitution
    — the decomposition is given to the model, not invented post-hoc;
(b) 2B's reasoning layer is shallow enough that softmax compression on a
    K-way disjunction is meaningfully lossy (larger models can do more
    internal reasoning before the final token, which would reduce but
    not eliminate this gap);
(c) the MHClip label distribution (~30% positive) and the baseline's
    mid-range miss pattern (EN oracle 0.7764, miss rate 22.4%) are
    consistent with a K-way disjunction compression failure rather than
    a systematic blind-spot on any one rule.

---

## 2. Mechanism

Run ONE MLLM forward pass per video with the following prompt template:

    You are a content moderation analyst. Observe this video's visual frames, title, and transcript.

    Title: {title}
    Transcript: {transcript}

    Below are the K community guideline rules for this platform. For each rule,
    decide whether this video violates that specific rule.

    {enumerated rules 1..K}

    For each rule in the order listed, reply with a single character: '1' if
    the video violates that rule, '0' if it does not. Separate the characters
    with single spaces. Do not explain. Output format: "X X X ... X" (K
    characters total, one per rule).

    Answer:

During generation, use vLLM with:

- `temperature=0` (greedy)
- `max_tokens = 2*K - 1` (K binary tokens + K-1 spaces)
- `logprobs=20` to capture the full distribution at each position
- `allowed_token_ids = {id('0'), id('1'), id(' ')}` to force a
  well-formed binary string

At each EVEN decode position i ∈ {0, 2, 4, ..., 2(K-1)}, extract:

    p_i = P('1' at position i) / (P('0' at position i) + P('1' at position i))

yielding a K-vector `p_rules ∈ [0,1]^K`. Positions 1, 3, 5, ... are space
separators and are ignored.

**Aggregation (primary, pre-committed):**

    score = max_{i=1..K} p_i

Max is chosen as the primary aggregator because:
- It is the **exact disjunction operator** for independent Bernoullis
  only in the degenerate limit — but under the constitution's
  non-overlapping-rule design, rules are approximately conditionally
  independent given the video, so max is a reasonable MAP approximation
  to noisy-OR (which would be `1 - prod_i (1 - p_i)`).
- Max is monotone, scale-free, and threshold-family robust under Otsu/GMM.
- Max preserves the rank of a positive that has one strong rule hit, which
  is exactly the baseline's confident-positive case, so we do not risk
  regressing on easy positives.
- Max suppresses spurious low-probability rule fires across many rules,
  which is exactly the compressed-softmax-over-union failure mode that
  the baseline suffers from in reverse.

Ablation D (see §4) will report max, mean, noisy-OR, top-2-mean, and
weighted-mean (weights = train-split positive-rate per rule) as
alternatives. Pre-commit: if any alternative strict-beats max oracle on
both datasets, max is not load-bearing and v5 retires.

**Number of MLLM calls per video: exactly 1.** One prompt, one
forward pass through the vision tower and text encoder, one autoregressive
decode run of 2K-1 tokens (decode is fast and uses the same KV cache
— the expensive part, vision encoding of frames, happens once).

---

## 3. Prediction and pre-commit

**Directional predictions:**

**H1.** Per-rule max score rank-AUC exceeds baseline P(Yes) rank-AUC on
at least one dataset. Specifically:

    AUC(max p_rules) > AUC(baseline P(Yes)) on EN OR ZH

**H2.** At least ONE of {tf_otsu, tf_gmm, tr_otsu, tr_gmm} applied to the
max-score satisfies:

    EN: ACC > 0.7640 AND mF1 ≥ 0.6532
    ZH: ACC > 0.8121 AND mF1 ≥ 0.7871

under a single unified (TR/TF × Otsu/GMM) configuration.

**H3 (oracle pre-commit, binding).**

    EN oracle > 0.7764 (strict)
    ZH oracle > 0.8121 (strict)

If the oracle check fails on EITHER dataset, v5 is an automatic Gate 2
MISS. v5 retires and I will propose v6.

**What would falsify the story?**

- **Null story 1 — softmax already does the disjunction:** if max_p_rules
  oracle ≤ baseline oracle on both datasets, the 2B model is already doing
  the K-way disjunction correctly inside its final softmax, and
  externalizing the aggregation adds no signal. v5 retires.
- **Null story 2 — structured output fails:** if the model does not
  follow the "K binary chars separated by spaces" format and we cannot
  read clean per-rule probabilities (e.g., model emits text instead of
  digits, or runs off the K-char budget), the mechanism is unimplementable
  at 2B. Report as retirement with an implementation note.
- **Null story 3 — per-rule signal is uniform:** if `mean_i corr(p_i, y)`
  is approximately equal across all K rules and matches the baseline's
  correlation to y, then no single rule carries more signal than the
  scalar Yes/No. The rule decomposition does not concentrate the signal,
  and max adds nothing. v5 retires.

Pre-pilot belief: I cannot numerically pre-pilot v5 because the prompt
and readout are new — there is no existing scoring artifact on MHClip
for a per-rule binary string. But the following indirect evidence
supports H1:

(a) The baseline's mid-range miss pattern (EN oracle 0.7764, and 27 FN
    at the TF-Otsu threshold) is consistent with rule-compound borderline
    positives being bottlenecked by softmax compression.
(b) v3 p_evidence — the SAME BINARY_PROMPT with a slightly different
    logprob extraction — produced ZH oracle 0.8188 vs baseline 0.8121
    (+0.0067), showing that even a minor readout perturbation can shift
    the oracle by 0.5-1% on 2B. A substantive readout change (scalar →
    K-vector max) should plausibly move it by 1-3%.
(c) The Qwen3-VL-2B chat template allows single-character constrained
    decoding (we use it for Yes/No baseline already), so the 2*K-1 token
    structured output is within vLLM's and the model's capability. Risk
    is moderate, not extreme.

**If the Gate 2 oracle fails on either dataset, v5 retires. I will not
rescue post-hoc.**

---

## 4. Counterfactual ablations (reported at Gate 2 even if main passes)

Each targets the STORY, not just the component.

- **A. Per-rule-max vs baseline P(Yes).** Score baseline BINARY_PROMPT
  P(Yes) on the same test split (copy from existing baseline file — no
  rescoring needed). If baseline oracle ≥ v5 max oracle on either
  dataset, the externalization added nothing and v5 retires.

- **B. Per-rule readout coverage.** Correlation between p_i and label y
  for each rule i ∈ 1..K. If no single rule's correlation exceeds the
  baseline's scalar P(Yes) correlation, the rule decomposition is
  spreading the signal rather than concentrating it, and the "compound
  borderline" story fails.

- **C. Aggregator robustness (AP1 self-binding).** Report max, mean,
  noisy-OR (`1 - prod_i (1 - p_i)`), top-2-mean, and per-rule weighted
  mean. If ANY alternative matches or strict-beats max oracle on both
  datasets, the specific max operator is not load-bearing — it is
  post-hoc selection from a family of aggregators — and v5 retires.

- **D. Prior-art self-check (AP2).** Compare v5 max oracle against:
  - baseline binary_nodef oracle (EN 0.7764, ZH 0.8121)
  - v3 p_evidence oracle (EN 0.7702, ZH 0.8188)
  - v2 factored oracle (if relevant)
  - v4 nor oracle (EN 0.7640, ZH 0.7987)
  v5 max must strict-beat all four on both datasets. If v5 is tied with
  or beaten by any existing scoring artefact on either dataset, v5 is
  not adding new information beyond the existing 2-call probe family
  and retires.

- **E. Structured-output compliance.** Fraction of test-set videos for
  which the model emits a clean K-char binary string (0/1/space only,
  exact K characters). If < 95%, constrained decoding failed and the
  readout is unreliable on the failures — report with per-rule
  probability fallbacks (take the first K decoded positions that are
  0/1). If < 80%, v5 retires.

- **F. Per-rule positivity distribution sanity check.** Per-rule
  positive rate on train split (`mean_i[p_i]` for each i) should not be
  constant across rules (otherwise the model is not actually
  distinguishing rules). Report rule-wise mean_p across train split; a
  reasonable distribution of per-rule positive rates (variance > 0.01)
  is required as evidence that the K-way decomposition is functional.

---

## 5. Binding Gate 2 clauses (carry forward)

v5's Gate 2 report must include, as binding pre-commit:

1. **Oracle-first (H3).** Strict beat on BOTH datasets, else MISS.
2. **Macro-F1 non-regression.** Unified cell's mF1 ≥ baseline mF1 on
   both datasets.
3. **Ablation A load-bearing (AP1 analogue).** Baseline P(Yes) oracle
   must be < v5 max oracle on BOTH datasets. Else v5 retires.
4. **Ablation C aggregator robustness.** Max must strict-beat all
   alternative aggregators (mean, noisy-OR, top-2-mean, weighted-mean)
   on at least one dataset, and not be tied/beaten on the other. Same
   "not-the-specific-technique" rule as v3 AP1 and v4 clause 5.
5. **Ablation D prior-art self-check.** Max must strict-beat baseline,
   v3 p_evidence, v4 nor, v2 factored oracle on BOTH datasets. Tying
   any prior artefact on either dataset → retirement.
6. **Ablation E structured-output compliance ≥ 80%** on both datasets.
   Else retirement with implementation note.
7. **Ablation F per-rule variance > 0.01** on train split. Else
   retirement — the K-way decomposition isn't happening in the model.
8. **N_test reconciliation.** EN N=161, ZH N=149 (same as v2/v3/v4).

---

## 6. How v5 addresses each prior failure mode

- **v1 — text-only Judge calibration drift.** v5 has no Call 2, no text
  cascade, no Judge. The readout is perceptual, same single forward
  pass as the baseline, just with K decode positions instead of 1.
- **v2 — multiplicative AND-gate compression.** v5 uses max, not
  product. A positive only needs one rule to fire. Max is the most
  permissive aggregator that is still rank-preserving — it cannot
  compress the signal via intersection.
- **v3 — equal-weight logit fusion over polarity-flipped probes.** v5
  has one probe, one polarity, one readout. No fusion operator between
  separate calls.
- **v4 — disjoint-support union fusion beaten by joint prompt.** v5
  reuses the joint prompt (visuals + title + transcript all in one call,
  same as baseline). The narrowing was exactly what hurt v4; v5 does
  not narrow the input. The novelty is the READOUT, not the input.

All four prior mechanisms (text-cascade, AND-gate, polarity-flip
logit fusion, modality split) are structurally excluded from v5 by
design. **v5 is the first method in this track that does not perturb
the number of calls or the input support.**

---

## 7. Anti-pattern self-audit

- **AP1 (no ensembling).** v5 makes exactly 1 MLLM call per video. No
  temperature sampling, no K-best reranking, no multi-prompt. The K
  decode positions are NOT K i.i.d. samples — each is the model's
  answer to one NAMED rule, in a fixed order, from one forward pass.
  Ablation C tests whether the max operator is load-bearing; if any
  simple alternative matches it, v5 auto-retires. This is the strongest
  AP1 binding of any method in this track so far, because the call
  count (exactly 1) is not even in dispute.

- **AP2 (engineering trick).** The mechanism is motivated by a concrete
  information-theoretic mismatch: softmax-over-first-token cannot
  faithfully compute a K-way disjunction, but an external max over K
  per-rule probabilities can. This is a property of how final-layer
  readout works, not a trick borrowed from another paper. Ablation B
  and F test whether the rule decomposition actually concentrates
  signal (per-rule correlation variance > 0, per-rule mean variance >
  0.01). If either fails, the decomposition is imaginary and v5
  retires.

- **AP3 (external datasets).** v5 uses only MHClip_EN / MHClip_ZH data.
  No auxiliary datasets, no retrieval, no external lexicons. The
  constitution rules are part of the target benchmark's own
  specification — they are not external knowledge.

---

## 8. Why I cannot give a numeric pre-pilot

v5 requires a new scoring run. The existing baseline
`test_binary.jsonl` contains only the scalar P(Yes) readout; there is
no per-rule probability artefact anywhere in `results/` that I can
rank-aggregate post-hoc into max_p_rules without actually running the
model. I cannot mimic v5 from v3 data or baseline data — the decoder
has to actually emit and score the K-char string.

The 8-point story above is my pre-experiment belief; H1/H2/H3 are my
falsifiable predictions; §3 lists the three null stories that would
falsify the mechanism; §4 and §5 list the 8 binding Gate 2 clauses. If
any of clauses 1-7 fails, v5 retires.

**I am committing in writing:** if v5 Gate 2 oracle fails on either
dataset, or if v5 max is tied/beaten by any aggregator-free prior
artefact (baseline, v3 p_evidence, etc.), I will retire v5 at Gate 2
and not attempt post-hoc rescue.

---

## 9. Resources

- **1 MLLM call per video**, single forward pass, ~(2K-1) decode tokens
  instead of 1. Vision encoding is the expensive part (~90% of latency
  per video); decode scales ~linearly with token count but each token
  is cheap. Expected runtime per split is **similar to baseline**
  (~10-25 min test, ~45-60 min train), because vision encoding dominates.
- Concurrent 2-GPU pairing: EN-test + ZH-test on wave 1, EN-train +
  ZH-train on wave 2, CPU eval on wave 3. Same pattern as v2/v3/v4.
  Total wall-clock: ~2-2.5 hours end-to-end.
- New files:
  - `src/prompt_paradigm/per_rule_readout.py` — scorer, outputs
    `{video_id, p_rules: [p1,...,pK], score: max(p_rules)}` per split.
  - `src/prompt_paradigm/eval_per_rule.py` — evaluator with the 8
    binding clauses above. Writes
    `results/prompt_paradigm/report_v5.json` and updates
    `results/analysis/prompt_paradigm_report.md`.
  - **`src/prompt_paradigm/run_v5_pipeline.sh`** — pipeline runner
    script (see §10 below).
- No edits to any frozen file. Rule text copied verbatim from
  `src/score_holistic_2b.py:32-49`. The prompt is NEW text (the
  structured-output instruction); verbatim copies apply only to the
  rule list. BINARY_PROMPT itself is not used in v5.
- Standing rules honored: Slurm discipline (no `--dependency`, no `&`,
  no chained job submission — the runner script polls with
  `squeue -j <own_ids>`), 2-GPU cap, active monitoring. Script records
  every submitted job ID to the run log immediately on submission.

---

## 10. Pipeline runner script (mandatory for v5 per team-lead ruling)

New file: `src/prompt_paradigm/run_v5_pipeline.sh`. Responsibilities:

1. **Wave 1 (test scoring):**
   - Submit `per_rule_readout.py --dataset MHClip_EN --split test` via
     `sbatch --gres=gpu:1 --wrap "..."`, capture job ID.
   - Submit `per_rule_readout.py --dataset MHClip_ZH --split test` via
     `sbatch --gres=gpu:1 --wrap "..."`, capture job ID.
   - Append both IDs + timestamps to `docs/experiments/prompt_paradigm_runs.md`.
   - Poll `squeue -u $USER -h -j <id1>,<id2>` every 30 seconds until
     both IDs disappear from the queue.
   - Verify output files `results/prompt_paradigm/MHClip_EN/test_per_rule.jsonl`
     and `results/prompt_paradigm/MHClip_ZH/test_per_rule.jsonl` exist
     and have the expected line counts (161 and 149 respectively, or
     157 before harmonization — script tolerates either).

2. **Wave 2 (train scoring):**
   - Immediately on wave 1 completion, submit train-split scoring jobs
     concurrently. Capture + append + poll identically to wave 1.
   - Verify output files exist.

3. **Wave 3 (CPU eval):**
   - Immediately on wave 2 completion, submit
     `sbatch --cpus-per-task=2 --mem=4G --wrap "... python
     src/prompt_paradigm/eval_per_rule.py"`, capture + append + poll.
   - Verify `results/prompt_paradigm/report_v5.json` exists.

4. **Exit cleanly** after the eval, print a one-line summary of each
   clause's pass/fail to stdout and append final status to the run log.

**Slurm discipline constraints (respected by runner):**
- No `--dependency` (not allowed).
- No `&` backgrounding of sbatch calls.
- No `scancel` of foreign jobs.
- Polls only the runner's own job IDs, recorded in a local variable.
- Polling interval = 30 seconds.
- The runner itself runs on the login node as a foreground bash
  script (this is explicitly allowed; only *Python* is forbidden on
  the login node, and the runner submits all Python via sbatch).

**Script is standalone and auditable.** Readable, commented, ~100
lines of bash. Approval of the runner script is part of Gate 1 for
v5; I will submit the script as a separate patch alongside the
proposal for pre-submission code review.

---

## 11. Open question for team-lead

One open design question that I cannot decide without your guidance:

**Should v5 emit per-rule probabilities via constrained decoding
(`allowed_token_ids={'0','1',' '}` plus length cap 2K-1), or via
unconstrained decoding with parsing?**

Constrained decoding is cleaner and guarantees the structured output
format, but some vLLM deployments have had issues with constrained
decoding + bf16 + Qwen3-VL; if constrained decoding fails silently
(e.g., emits the wrong token because the logit mask bypasses the
model's native Yes-token preference), the per-rule probabilities will
be garbage. Unconstrained decoding requires robust parsing and has a
per-video failure rate — Ablation E tests exactly this — but is
mechanically simpler.

My current plan: **try constrained decoding first**, and fall back to
unconstrained + parsing if the constrained-decoding implementation
shows > 10% garbage rate on the test split. This is a pilot I can do
on 10 videos before the full scoring run, entirely during Gate 1
prep, without touching the train split.

If you want me to pre-commit to unconstrained decoding (which gives a
cleaner AP2 story — "the model chose its own format") or to
constrained decoding (which gives a cleaner AP1 story — "no parser
tricks"), state which. Otherwise I proceed with the constrained-first,
unconstrained-fallback plan.

Awaiting Gate 1 approval.
