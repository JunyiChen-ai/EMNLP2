# Idea Discovery Report

**Direction**: Social/Psychology Theory for Hateful Video Detection
**Date**: 2026-03-27
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → experiment-plan

## Executive Summary

We surveyed 15 social/psychology theories and generated 11 initial ideas for theory-grounded hateful video detection. After two rounds of novelty verification and external critical review (GPT-5.4 as EMNLP senior reviewer), **4 ideas survived** for implementation. The strategy is phased: first screen all 4 theories by their MLLM rationale quality, then implement full theory-aware architectures for the top performers.

**All 4 ideas share the same paradigm**: psychology theory → MLLM structured rationale → BERT-encode fields → theory-aware fusion → theory-motivated middle module → classifier.

---

## Literature Landscape

### Central Gap
Rich social science theories exist for explaining hate, but **zero** have been applied to video-based hate detection. Only 3 theories have text-only computational work:
- Moral Foundations Theory (MFTCXplain, EMNLP 2025 Findings)
- Dehumanization (Saffari et al., EMNLP 2025 Main)
- Social Norm Violation (Scientific Reports 2023)

### Latest Video Hate Detection Methods (all theory-free)
- ARCADE (2026): Judicial debate for intent shifts
- TANDEM (2026): Tandem RL
- MultiHateLoc (WWW 2026): Temporal localization
- RAMF (2025): Reasoning-aware multimodal fusion
- SCANNER (2025): Test-time adaptation
- ImpliHateVid (ACL 2025): Implicit hate in video
- DeHate (ACM MM 2025): 6,689-video dataset
- MultiHateGNN (BMVC 2025): Graph neural network for video hate
- HVGuard (EMNLP 2025): Chain-of-thought for hate video

### Key Insight for Method Design
SOTA pattern in 2025-2026: **MLLM generates structured rationale → encode → fuse as additional modalities**. The theory's value is in structuring what the LLM reasons about, producing more discriminative embeddings.

---

## Ranked Ideas

### 🏆 Idea 1: ITT — Threat-to-Hostility Pipeline (Integrated Threat Theory)
- **Theory**: Stephan & Stephan (2000). Four threat types: realistic, symbolic, intergroup anxiety, negative stereotypes.
- **Reviewer Score**: 8/10
- **Novelty**: CONFIRMED NOVEL
  - Closest: Alorainy et al. (2019, ACM TWeb) — text-only, loose ITT motivation, no 4-type decomposition
  - RAMF shares paradigm (structured LLM reasoning + fusion) but has no theory
- **Why #1**: Best match to real-world hateful rhetoric (invasion, contamination, crime, replacement). Broadest coverage. Best transfer potential. Highest ceiling per reviewer.
- **LLM Fields**: target_salience, realistic_threat, symbolic_threat, anxiety_discomfort, stereotype_support, hostility_prescription (6 fields)
- **Fusion**: 4 parallel threat channels → Threat Accumulator
- **Middle Module**: Threat Moderation Block — target_salience × stereotype_fit moderates threat effect
- **Key Risk**: Four threat types may bleed into each other; needs human annotation to validate separability

### 🥈 Idea 2: IET — Intergroup Emotion Pipeline
- **Theory**: Mackie, Devos, Smith (2000); Smith, Seger, Mackie (2007). Chain: group categorization → group-based appraisal → group-based emotion → action tendency.
- **Reviewer Score**: 7/10
- **Novelty**: CONFIRMED NOVEL
  - Closest: Govindarajan et al. (EACL 2023) — text-only, IGR prediction, no pipeline
  - IET has NEVER been computationally operationalized
- **Why #2**: Most natural reasoning chain. Safest paper. Sequential fusion matches theory well. But emotion labels may be noisy.
- **LLM Fields**: group_framing, appraisal_evidence, emotion_inference, action_tendency, endorsement_stance (5 fields)
- **Fusion**: 3-stage sequential: Appraisal → Emotion → Action (each stage gates next)
- **Middle Module**: Appraisal-to-Emotion Transport — emotion simplex projection + coherence score
- **Key Risk**: Emotion inference from short clips is noisy; sequential constraint may be too rigid

### 🥉 Idea 3: Attribution — Responsibility-Blame-Punishment Pipeline
- **Theory**: Weiner (1985, 1995). Chain: negative outcome → causal attribution → controllability → blame → punishment.
- **Reviewer Score**: 6.5/10
- **Novelty**: CONFIRMED NOVEL
  - Closest: HARE (EMNLP 2023 Findings) — descriptive CoT, no theory
  - Weiner has NEVER been applied to hate detection in NLP
- **Why #3**: Sharpest theory-method bridge. Counterfactual Blame Filter is most falsifiable middle module. Best for scapegoating/conspiracy hate. But narrower coverage.
- **LLM Fields**: negative_outcome, causal_attribution, controllability, responsibility_blame, punitive_tendency (5 fields)
- **Fusion**: Strict causal chain: Outcome → Attribution → Blame → Punishment (later stages only attend through earlier)
- **Middle Module**: Counterfactual Blame Filter — if controllability is low, blame signal suppressed
- **Key Risk**: Too narrow — much hate has no explicit blame chain

### Idea 4: SCM+BIAS — Warmth-Competence Harm Pipeline
- **Theory**: Fiske et al. (2002) + BIAS Map (Cuddy, Fiske, Glick 2007). Two dimensions: warmth, competence → four quadrants → different harm behaviors.
- **Reviewer Score**: 5.5/10
- **Novelty**: NOVEL (no prior work applies SCM to hate detection)
  - Closest: Fraser et al. (ACL 2021) — computational SCM for stereotype understanding, not detection
  - Soral et al. (2024) — empirically validated SCM→hate link, but not computational
- **Why #4**: Elegant 2D geometry. Good interpretability. But weakest theory-content fit to hateful video.
- **LLM Fields**: target_group, warmth_evidence, competence_evidence, social_perception, behavioral_tendency (5 fields)
- **Fusion**: Dual-stream (Warmth + Competence) → Quadrant Composer
- **Middle Module**: Quadrant Attractor — 4 learned prototypes for contempt/envy/pity/admiration
- **Key Risk**: Four quadrants may be too coarse; warmth/competence mapping culturally unstable across EN/ZH

---

## Eliminated Ideas (from initial 11)

| Idea | Theory | Eliminated At | Reason |
|------|--------|:---:|--------|
| DEHUMANIZE-V | Haslam (2006) | Phase 2 | Dual-route dehumanization — feasibility concerns, unclear video advantage |
| HIERARCHY-LENS | Sidanius & Pratto (1999) | Phase 2 | Social dominance — hard to operationalize hierarchy pseudo-labels |
| FRAMECONFLICT | Goffman (1974) | Phase 2 | Framing incongruity — too broad, hard to define hate-specific framing |
| MOBILIZE-HATE | van Zomeren et al. (2008) | Phase 2 | SIMCA collective action — datasets label hate, not mobilization |
| PARASOCIAL HATE | Horton & Wohl (1956) | Phase 2 | Parasocial interaction — interesting but more about persuasion than detection |
| MORTALITY-CUE | Greenberg et al. (1986) | Phase 2 | Terror management — hard to validate, overlaps with legitimate news |
| DISENGAGE (original) | Bandura (1999) | Phase 2→3 | Moral disengagement graph — redesigned as prompting pipeline; graph structure dropped in favor of MLLM rationale approach |

---

## Key References

### Social Psychology Theories
- Stephan, W.G. & Stephan, C.W. (2000). Integrated Threat Theory. *Int. J. Intercultural Relations*.
- Mackie, D.M., Devos, T., & Smith, E.R. (2000). Intergroup emotions. *J. Personality & Social Psychology*.
- Smith, E.R., Seger, C.R., & Mackie, D.M. (2007). Can emotions be truly group level? *J. Personality & Social Psychology*.
- Weiner, B. (1985). An attributional theory of motivation and emotion. *Psychological Review*.
- Fiske, S.T., Cuddy, A.J.C., Glick, P., & Xu, J. (2002). A model of (often mixed) stereotype content. *J. Personality & Social Psychology*.
- Cuddy, A.J.C., Fiske, S.T., & Glick, P. (2007). The BIAS Map. *J. Personality & Social Psychology*.
- Bandura, A. (1999). Moral disengagement in the perpetration of inhumanities.

### Closest Computational Prior Art
- Alorainy, W. et al. (2019). The enemy among us: Detecting cyber hate speech with threats-based othering. *ACM Trans. Web*.
- Fraser, K.C. et al. (2021). Understanding and countering stereotypes: A computational approach to the SCM. *ACL 2021*.
- Govindarajan, V. et al. (2023). Modeling generalized intergroup bias and emotion. *EACL 2023*.
- Friedman, S. et al. (2021). Toward transformer-based NLP for extracting psychosocial indicators of moral disengagement. *CogSci Workshop*.
- Vargas, F. et al. (2025/2026). Self-explaining hate speech detection with moral rationales. *EMNLP 2025 Findings*.

### Hateful Video Detection SOTA
- RAMF (2025): Reasoning-aware multimodal fusion. arXiv:2512.02743.
- MARS (2026): Training-free multi-stage adversarial reasoning. arXiv:2601.15115.
- TANDEM (2026): Temporal-aware neural detection. arXiv:2601.11178.
- MultiHateLoc (WWW 2026): Temporal localization. arXiv:2512.10408.
- ARCADE (2026): Intent shifts in multimodal hate. arXiv:2603.21298.
- ImpliHateVid (ACL 2025). aclanthology.org/2025.acl-long.842.
- DeHate (ACM MM 2025). doi:10.1145/3746027.3758272.
- MultiHateGNN (BMVC 2025). arXiv:2509.13515.
- HVGuard (EMNLP 2025). aclanthology.org/2025.emnlp-main.456.

---

## Next Steps
- [ ] Phase A: Run all 4 LLM prompts → encode → quick screening with baseline fusion
- [ ] Phase B: Best theory → full architecture + seed search
- [ ] Phase C: Second-best theory → full architecture + seed search
- [ ] Phase D: If time, third/fourth
- [ ] Cross-dataset transfer evaluation
- [ ] Ablation study
- [ ] Paper writing: /paper-writing pipeline
