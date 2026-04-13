# prompt_paradigm v6 — design scratch notes (2026-04-13)

Author: prompt-paradigm (Teammate A)
Status: DRAFTING. Not a proposal. Gate 1 proposal will be `docs/proposals/prompt_paradigm_v6.md`.

## Starting premises (from Gate 2 v5 post-mortem)

- 1-call hard constraint carried forward from v5.
- Five mechanisms falsified: text-cascade drift (v1), AND-gate (v2), polarity-flip (v3), disjoint-support nor (v4), per-rule readout (v5).
- All five manipulated the OUTPUT (call structure, polarity, readout). They were output-side levers.
- Strongest 1-call prior art: v3 p_evidence, EN oracle 0.7702, ZH oracle 0.8188. Still misses EN baseline 0.7764 by 0.0062.
- Director accepted "change what the model is conditioned on" as the v6 direction. Narrowing: INPUT-side manipulation.
- Must strict-beat baseline (EN 0.7640 / ZH 0.8121) under one unified TR/TF × Otsu/GMM cell.
- Must beat v3 p_evidence on Ablation D (prior-art) on BOTH datasets.
- Oracle-first clause carries forward.
- Anti-patterns still binding: AP1 (no ensembling), AP2 (no engineering tricks), AP3 (no external hate data).

## What is the INPUT to a single 2B call?

A single call to Qwen3-VL-2B-Instruct has these input-side levers:
1. **Frames** — which/how-many frames, what resolution, what sampling pattern.
2. **Title** — present/absent, verbatim/paraphrased/summarized.
3. **Transcript** — present/absent, truncation length, preprocessing (cleaning, chunking).
4. **Rules text** — which rules, in what order, how worded, how grouped.
5. **Task framing** — system prompt, role, question wording, output format instruction.
6. **Examples / demonstrations** — none currently (label-free forbids labeled in-context).
7. **Context augmentation from the MEDIA ITSELF** — structured summaries, captions, OCR, ASR — derived *from* the video, not from external data.

v1-v5 all used the DEFAULT input support (frames + title + transcript + full rules + holistic binary question), varying only v4 which zeroed out one modality each call. None of v1-v5 manipulated the content of what was shown.

## Scientific question for v6

"If the single binary call's output structure cannot be improved further (five failures), can we improve the single call's *conditioning* — what it reads before deciding — in a way that is (a) derived only from the video itself and (b) changes the binary question's decision boundary in a content-specific way?"

## Candidate directions I want to evaluate

### D1. Frame curation / keyframe selection
**Phenomenon claim:** hate signal in video is temporally sparse and localized; uniformly-sampled 8 frames dilute the signal.
**Input-side lever:** select frames by some label-free saliency criterion (e.g., CLIP similarity to platform policy text, face presence via a frozen detector, scene change, motion energy) and pass those to the one call.
**Issue:** would need an external model for the saliency criterion. CLIP is fine (general-purpose pretrained), but any "hate-specific" scoring model would be AP3-adjacent.
**Concern:** this is a "pick which 8 frames" trick, not a scientific story about hateful video specifically. The phenomenon ("hate is sparse") would predict EN and ZH gains together. But this is close to an engineering trick unless the selection mechanism has a deeper reason.

### D2. Transcript-sensitive decoding — context-length scan
**Phenomenon claim:** the default 300-char transcript truncation drops essential target-identification cues in hateful ZH content that carries the target mention late in the utterance.
**Input-side lever:** extend/reduce transcript length, or apply a label-free "informative segment" selection (e.g., first/last/longest/most-repeated-tokens).
**Issue:** this is HP tuning dressed as a method. AP2 red flag.

### D3. Question-conditioning reformulation
**Phenomenon claim:** the current question "does this contain content that violates any of the above rules" forces the model into a binary moderation judgment that triggers safety-aligned hedging (we already saw this on 8B in iteration 2). A more descriptive ("what group, if any, does this video target?") conditioning might elicit the hate-specific signal with less hedging.
**Input-side lever:** instead of asking the binary question first, the CONTEXT of the single call asks for a targeting-frame descriptor, and the binary question is decoded OFF the targeting frame.
**Risk:** this is close to observer-then-judge (v1 pattern), which failed.
**Crucial distinction from v1:** in v1 we added a *second call* where an observer wrote free text that a judge then read. In v6 D3 it would be a SINGLE call whose context says "consider whether the video targets a protected group, then answer: does this violate policy?". One autoregressive decode, no aggregation. The in-context reasoning is part of the model's own forward pass, not a separate call. This is structurally a chain-of-thought variant, which is AP1-adjacent (CoT sampling is forbidden) but single-pass greedy CoT is not ensembling.

### D4. Rule re-grouping / rule-set minimization
**Phenomenon claim:** 9 EN rules / 8 ZH rules include sub-axes that the 2B model cannot discriminate; a smaller collapsed rule set whose axes are separable by 2B's visual+linguistic reasoning would give cleaner Yes/No.
**Input-side lever:** replace the verbatim YouTube/Bilibili rules with a 2-3 axis taxonomy derived from the rules (e.g., "targets a protected group" + "expresses hostility/dehumanization/incitement"). This is NOT relaxing the definition — it's restating the disjunction in a form the model can evaluate.
**Issue:** this is a prompt-engineering trick unless we can say WHY 2B fails on the 8-9-rule version and succeeds on the collapsed version. We have empirical evidence from v5: the per-rule variance is near-zero — the model doesn't actually treat the rules as separable. So collapsing them matches the model's true internal factorization. Phenomenon: "2B internally operates on a coarse binary target+hostility factorization; the verbatim rule list is mis-specified for its granularity".
**This is the one I think has the strongest story.**

### D5. Modality-rebalanced single prompt
**Phenomenon claim:** in the default holistic prompt, the MLLM's attention is dominated by the longer textual rules block, suppressing frame contribution. Rearranging the prompt to place frames AFTER the rules and BEFORE the question shifts attention back toward visual evidence.
**Input-side lever:** prompt order manipulation. One call, one autoregressive decode.
**Issue:** this is a prompt-template trick. AP2 red flag unless we can argue why POSITIONAL order matters for hate specifically.

### D6. Self-contextualization from the video's own caption
**Phenomenon claim:** hateful video often depends on the VIEWER's identification of the target group; the model needs to first "name" what it sees before it can evaluate policy compliance. A one-call prompt that asks the model to describe what the video shows AND then answer yes/no in the same decode is a greedy-single-pass CoT.
**Input-side lever:** prepend a "first describe the video in one sentence, then answer yes/no" instruction. One call, one decode. Extract the Yes/No probability at the forced answer position (after the description).
**Closeness to v1:** v1 had two separate MLLM calls (observer then judge). v6 D6 has one MLLM call whose decode happens to include intermediate tokens before the Yes/No. The 1-call constraint binds at the CALL level, not the token level. Each call still has "one role". The description+binary decode is one role: self-conditioned binary classification.
**Issue:** we need the Yes/No probability at a DETERMINED position. Forced answer format can do this: "Description: <free text>. Answer: Yes or No." and read logprob at the position after "Answer:" — or use structured output with a separator token.

## Which direction has the strongest story?

Ranking by story strength (phenomenon → mechanism → prediction → falsifiable ablation):

1. **D4 rule-taxonomy collapse**: has a SPECIFIC empirical anchor (v5 per-rule variance ~ 0 proves the model doesn't factor the rules). The phenomenon is "2B's internal policy-evaluation circuit is binary in target×hostility axes, not 9-way in rule axes". Mechanism: restating the rules at 2B's native granularity reduces prompt noise and increases the signal at the one binary position. Prediction: both datasets should gain, because the re-grouping is not language-specific — it reflects 2B's shared latent policy representation.
2. **D6 self-contextualization / single-pass CoT**: has a reasonable phenomenon ("target identification must precede policy judgment") but the mechanism is shared with CoT literature in general, not hateful-video-specific. This is AP2 adjacent.
3. **D3 question reformulation**: close to D6, weaker specificity.
4. **D1 keyframe selection**: engineering trick without a story.
5. **D2 transcript length scan**: HP tuning.
6. **D5 prompt order**: prompt-engineering trick.

D4 is my current favorite. Let me stress-test it.

## D4 stress-test

**Empirical anchor:** v5 clause 7 shows train per-rule variance is 8.87e-5 (EN) / 9.53e-5 (ZH). The decoder's per-position probability does not vary by rule content. This says the 2B model does NOT internally discriminate the 9/8 rules — it has a coarser representation.

**Testable hypothesis:** if we collapse the rules to a 2-axis taxonomy ("group_targeting" + "hostility_intensity"), the holistic binary question will be easier for the model to answer because the prompt no longer asks for discrimination the model cannot perform.

**Mechanism:** under the verbatim rules, the model sees 9 fine distinctions, expends attention on trying to match content to the most-specific rule, and confuses its binary vote. Under a 2-axis collapse, the model evaluates two coarse questions its latent circuit can handle, and the binary output is cleaner.

**Prediction (what should happen if D4 is right):**
- EN oracle and ZH oracle both strict-beat v3 p_evidence (EN > 0.7702, ZH > 0.8188).
- Unified cell should strict-beat baseline (EN > 0.7640, ZH > 0.8121).
- The gain should be LARGER on EN (where rules are 9, more collapse) than ZH (where rules are 8).
- If we do an ablation where we keep the verbatim rules but prepend a 2-line "axes to consider" summary, we should get MOST of the gain (showing the axes themselves are the load-bearing bit, not the removal).

**Falsifying prediction:** if D4 is NOT a hate-specific phenomenon but just a "shorter prompt helps" effect, then replacing the 9 rules with ANY 2-line text of similar length (e.g., a legal-disclaimer paragraph) would give similar gains. So the v6 ablation must include a "prompt-length control" clause: the 2-axis collapse must strictly beat a 2-line non-taxonomic control.

**1-call compliance:** D4 is literally one binary call with a different system/user block. Cleanest possible 1-call design.

**AP compliance:**
- AP1: one call, one forward pass, one Yes/No logprob read. No ensembling.
- AP2: there IS a hateful-video-specific phenomenon — 2B's latent policy representation is coarser than the verbatim rule list. Anchored in v5 empirical evidence.
- AP3: the 2-axis taxonomy is DERIVED from the existing YouTube/Bilibili rules. No external dataset. No new labels. The collapse map is a frozen constant, authored by me from re-reading the current rules.

**Frozen-file compliance:** D4 does NOT edit `src/score_holistic_2b.py`. It creates a new scorer `src/prompt_paradigm/coarse_axes_prompt.py` that reuses the message-building pattern from `modality_split.py` but with a new rules block.

## Refinement: what exactly are the 2 axes?

I need to define the collapsed taxonomy from the existing rules without introducing my own definition of hate.

**EN (YouTube) rules:**
1. Encourage violence against groups based on protected status
2. Incite hatred against groups based on protected status
3. Dehumanize groups by comparing to non-human entities
4. Praise or glorify violence against protected groups
5. Use slurs and stereotypes that incite hatred
6. Claim groups are inferior based on protected status
7. Promote hateful supremacism
8. Make conspiratorial claims about groups based on protected status
9. Deny or minimize major violent events

Every EN rule says "against groups based on protected status / against protected groups / about groups based on protected status / (implicitly) about groups". Every rule decomposes into `[TARGET: protected group]` + `[ACTION: violence / hatred / dehumanization / inferiority claim / supremacism / conspiracy / denial]`. So the 2 axes are: **(A) does the content target an identifiable protected group; (B) does it carry one of {violence, hatred, dehumanization, inferiority, supremacism, conspiracy, denial} toward that target**.

**ZH (Bilibili) rules:**
1. Discriminatory content (violence, slurs, exclusion)
2. Attacks, disparagement, belittlement
3. Verbal abuse, personal attacks
4. Displaying/promoting hateful ideologies/symbols
5. Mocking death, sickness, disability
6. Jokes about disasters or tragedies
7. Gender-based hate speech
8. Inciting confrontation/conflict between groups

ZH 1, 2, 4, 7, 8 fit the same target×hostility schema. ZH 3, 5, 6 are slightly different — they include content that may target individuals or circumstances, not protected groups. So the ZH 2-axis collapse needs a looser TARGET axis: "an identifiable person, group, or circumstance (death/illness/disability/disaster)" — not just "protected group".

This looseness on ZH is a worry: it makes the collapse dataset-specific. The cleaner story is one shared taxonomy. Let me re-examine.

Actually, the shared schema is: **(A) the content identifies a TARGET whose harm is covered by platform policy; (B) the content expresses a HOSTILE attitude toward that target**. The interpretation of "target" (protected group vs broader) is platform-specific, which is FINE because the rules text itself is already platform-specific — I am just collapsing each platform's own rules into that platform's own 2-axis form.

So D4 has TWO taxonomies — one per platform — each derived from that platform's own rules. That's defensible.

## v6 design crystallizing (Coarse Axes Prompt — CAP)

Structure:
- ONE call to Qwen3-VL-2B-Instruct, bf16, vLLM, temperature=0.
- System prompt: content moderation analyst (same as v4/v5).
- User content: frames + title + transcript + 2-axis COLLAPSED rules block (NOT the verbatim 9/8 rules) + binary question.
- Output: Yes/No at position 0, extracted via same `extract_binary_score` pattern as `score_holistic_2b.py`.
- Fusion: none. The score is just `p_yes / (p_yes + p_no)`.

**Key change from baseline:** the rules block is rewritten as a 2-axis statement derived from the platform's own rules. Everything else is identical to `score_holistic_2b.py` defaults.

**Ablation matrix for Gate 2:**
- **A (baseline load-bearing):** v6 vs 2B `binary_nodef` baseline — must strict-beat on BOTH.
- **B (prior-art self-check):** v6 vs v3 p_evidence oracle (EN 0.7702, ZH 0.8188) — must strict-beat on BOTH.
- **C (axes-content load-bearing — the critical control):** v6 vs a 2-axis NON-taxonomic control with matched length. The control is 2 lines of platform-flavored safety language that do NOT encode the target×hostility schema. If v6 only wins because the prompt got shorter, the control wins too. Falsifying prediction: v6 must strict-beat the control.
- **D (axes alone vs axes PLUS verbatim rules):** prepend the 2-axis axes before the verbatim 9/8 rules. Does the hybrid recover the same gain as pure collapse? If yes, the axes themselves are the load-bearing bit. If no, collapse is load-bearing via pruning.
- **E (axes-only vs baseline with prompt-length equal):** make the baseline prompt the same length by padding/cropping — isolates the length effect.
- **F (unified TR/TF × Otsu/GMM cell):** one cell must strict-beat.
- **G (format compliance / decode sanity):** should be automatic since output is 1 token.
- **H (n_test reconciliation):** standard.

## Open questions I need to answer before Gate 1 submission

1. **Does 2B actually have this coarser internal representation, or am I post-hoc rationalizing from v5 variance?** The v5 per-rule variance is suggestive but not proof. Counter-possibility: the constrained decode's position bias is an ARTIFACT of forcing '0'/'1' tokens at 9 consecutive positions, not evidence of the model's internal factorization. In that case, asking the binary question with a 2-axis prompt would not help — the binary question under verbatim rules is already at the top level of the model's computation, not per-rule.
2. **Is there prior literature on rule-set collapse for moderation that I should cite?** I know Perspective API / Jigsaw uses fine categories; Llama Guard collapses to 13 categories. I should look for any paper that shows prompt-length or category-count affects moderation accuracy.
3. **What does "strict-beat v3 p_evidence on EN by 0.0062" actually require?** v3 p_evidence EN oracle 0.7702. Baseline EN oracle 0.7764. I need v6 EN oracle > 0.7764 AND unified cell mF1 non-regression under one TR/TF × Otsu/GMM label. That's the bar.
4. **Null story for D4:** the null for D4 is "rewriting the rules has no effect; the model reads the question and ignores the rules block". If I implement D4 and get v6 oracle ≈ baseline ≈ v3, that's the null confirming. The ablation clause C (length control) must ALSO be ≈ baseline to conclude that D4 specifically fails (not that "no prompt change matters").

## Next steps

1. Read any arXiv paper on moderation rule collapse / prompt taxonomy if time permits.
2. Flesh out the coarse taxonomy strings for EN and ZH — draft exact text.
3. Write `docs/proposals/prompt_paradigm_v6.md` with full 4-point story + 8 binding clauses.
4. Send to team-lead for Gate 1 approval.
