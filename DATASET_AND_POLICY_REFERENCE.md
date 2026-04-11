# Dataset Annotation Guidelines & Platform Hate Speech Policies

**Date**: 2026-04-10

---

## 1. HateMM

### Paper
- **Title**: "HateMM: A Multi-Modal Dataset for Hate Video Classification"
- **Authors**: Mithun Das, Rohit Raj, Punyajoy Saha, Binny Mathew, Manish Gupta, Animesh Mukherjee (IIT Kharagpur + Microsoft India)
- **Venue**: ICWSM 2023
- **Paper**: https://arxiv.org/abs/2305.03915
- **Code**: https://github.com/hate-alert/HateMM
- **Data**: https://zenodo.org/records/7799469 (CC-BY 4.0)

### Source Platform
**BitChute** — alt-right video platform with minimal moderation; chosen for high prevalence of hateful content.

### Definition of "Hateful" (Codebook)
Based on **YouTube's hate speech policy** (https://support.google.com/youtube/answer/2801939):

> *"It promotes discrimination or disparages or humiliates an individual or group of people on the basis of the race, ethnicity, or ethnic origin, nationality, religion, disability, age, veteran status, sexual orientation, gender identity etc."*

**Binary label**: Hate vs Non-Hate. No separate "offensive" category.

### Annotation Process
- **Annotators**: 2 PhD experts + 4 undergrads
- **Training**: 30-video pilot with gold labels, then calibration discussions
- **Process**: Dual annotation; expert tie-breaker on disagreement
- **Workload**: Max 10 videos/day, 10-min breaks between videos
- **Additional annotations**: Frame spans (time-stamped hateful segments), target communities (Blacks, Jews, Whites, Asians, LGBTQ, Others)

### Inter-Annotator Agreement
**Cohen's kappa = 0.625** ("substantial" but at low end)

### Statistics
| | Hate | Non-Hate | Total |
|--|------|----------|-------|
| Count | 431 (39.8%) | 652 (60.2%) | 1,083 |
| Mean length (min) | 2.56 | 2.28 | 2.40 |

**Note**: Transcripts via Vosk (offline ASR), ~22% OOV rate.

### What is NOT provided
- No detailed codebook document published separately
- No fine-grained hate taxonomy (no implicit vs explicit, no severity)
- No offensive-but-not-hateful category

---

## 2. MultiHateClip (MHClip_EN / MHClip_ZH)

### Paper
- **Title**: "MultiHateClip: A Multilingual Benchmark Dataset for Hateful Video Detection on YouTube and Bilibili"
- **Authors**: Han Wang, Tan Rui Yang, Usman Naseem, Roy Ka-Wei Lee
- **Venue**: ACM Multimedia 2024
- **Paper**: https://arxiv.org/abs/2408.03468
- **Code**: https://github.com/Social-AI-Studio/MultiHateClip

### Source Platforms
- **MHClip_EN**: YouTube (English videos, ≤60s)
- **MHClip_ZH**: Bilibili (Chinese videos, ≤60s)

### 3-Class Taxonomy (follows Davidson et al., 2017)

1. **Hateful**: *"Videos that incite discrimination or demean individuals or groups based on attributes such as race, ethnicity, nationality, religion, disability, age, veteran status, sexual orientation, gender identity, etc."*

2. **Offensive**: *"Videos that may cause discomfort or distress, yet do not qualify as hateful under the criteria defined above."*

3. **Normal**: *"Content devoid of hatefulness or offensiveness."*

**Key distinction**: Hateful requires **targeting a protected group** with discrimination/demeaning. Offensive causes distress but does NOT target a protected group or does not rise to that threshold.

### Annotation Process
- **Annotators**: 2 PhD experts + 18 undergrads (all Asian, 18-24, balanced gender)
- **Training**: 3 rounds of calibrating on 30-video test sets
- **Process**: Dual annotation; 3rd annotator on disagreement; experts for persistent conflicts
- **Four questions per video**: (1) Category, (2) Start/end times of H/O segments, (3) Target victim, (4) Contributing modality

### Inter-Annotator Agreement
| Setting | English | Chinese |
|---------|---------|---------|
| 3-class (kappa) | 0.62 | 0.51 |
| Binary H+O vs N (kappa) | 0.72 | 0.66 |

**Hateful/Offensive confusion** is the primary source of disagreement: 24% of EN inconsistencies, 40% of ZH inconsistencies.

### Statistics
| | EN (YouTube) | ZH (Bilibili) |
|--|-------------|----------------|
| Hateful | 82 (8.2%) | 128 (12.8%) |
| Offensive | 256 (25.6%) | 194 (19.4%) |
| Normal | 662 (66.2%) | 678 (67.8%) |
| **Total** | **1,000** | **1,000** |

### What is NOT provided
- No separate annotation codebook published
- No fine-grained hate sub-taxonomy

### Related Follow-up
**HateClipSeg** (arXiv 2508.01712) extends this with segment-level annotations and 5-subcategory offensive taxonomy, Krippendorff's alpha = 0.817.

---

## 3. Platform Hate Speech Policies

### YouTube (source for HateMM definition + MHClip_EN)

**Policy**: https://support.google.com/youtube/answer/2801939

**Definition**: Content that "incites hatred or violence against groups based on protected attributes."

**Protected attributes**: Age, Caste, Disability, Gender, Nationality, Race, Religion, Sex, Sexual orientation, Veteran status.

**Prohibited content types**:
1. Promoting violence/hatred against protected groups
2. Dehumanization (subhuman comparisons)
3. Slurs and stereotypes that incite hatred
4. Claims of inferiority
5. Hateful supremacism
6. Conspiracy theories about protected groups
7. Denial of violent events
8. Promoting hateful ideologies (e.g., Nazism)

**Exceptions**: EDSA (Educational, Documentary, Scientific, Artistic) context.

**Hateful vs offensive**: Separate policies — hate speech targets *groups* based on protected attributes; harassment targets *individuals*; vulgar language is a different policy.

### Bilibili (source for MHClip_ZH)

**Policy**: https://www.bilibili.tv/marketing/protocal/communityrules_en.html

**Definition**: *"Any content you publish must not include discriminatory content or disparage any religion, belief system, race, sexual orientation, or protected group."* Also: *"Attacks, disparagement, or belittlement that target a specific individual or group based on race, religion, gender, age, nationality, disability, sexual orientation, etc."*

**Protected attributes**: Race, Religion, Gender, Age, Nationality, Disability, Sexual orientation.

**Additional prohibited categories**:
- Mocking death, sickness, and disability
- Jokes about disasters and tragic social events
- Hateful content directed toward gender groups (misogynistic/misandrist content specifically named)

**Hateful vs offensive**: Treated as continuum under Chinese regulatory framework, no explicit Western-style distinction.

**Context**: Moderation shaped by China's "Nine Prohibitions" (CAC regulation). Content categorization differs from Western taxonomies.

### BitChute (source for HateMM data)

**Policy**: https://support.bitchute.com/policy/guidelines

**Term used**: "Incitement to Hatred" (not "hate speech")

**Definition**: *"Material likely to incite hatred against a group of persons or a member of a group of persons"* based on UK Communications Act 2003, referencing EU Charter Article 21.

**Critical threshold**: *"There should be a reasonable probability that the content would succeed in inciting actual action against the target."* This is a much higher bar than YouTube/TikTok.

**Protected grounds (14 categories from EU Charter)**: Sex, Race, Colour, Ethnic/social origin, Genetic features, Language, Religion/belief, Political opinion, National minority, Property, Birth, Disability, Age, Sexual orientation.

**Hateful vs offensive**: Explicit distinction via sensitivity tiers — discriminatory language = NSFW (allowed, age-restricted); incitement = prohibited (geo-blocked in UK/EU/EEA only).

**Enforcement**: Geo-restriction, not global removal. Positioning as "free speech" platform.

---

## 4. Cross-Platform Comparison

| Dimension | YouTube | Bilibili | BitChute |
|-----------|---------|----------|----------|
| **Term** | Hate speech | Discriminatory content | Incitement to Hatred |
| **Threshold** | Promotes hatred/violence | Attacks, disparages | Must have reasonable probability of inciting action |
| **# Protected attributes** | ~10 | ~7 | 14 (EU Charter) |
| **Hateful vs Offensive** | Separate policies | Continuum | Sensitivity tiers (NSFW vs prohibited) |
| **Exceptions** | EDSA | Not explicit | 8 categories (press, satire, etc.) |
| **Enforcement** | Global removal | Deletion + suspension | Geo-restriction only |

---

## 5. Implications for Our Project

### The definition problem
Both datasets derive their definitions from **YouTube's policy**: HateMM explicitly cites it; MultiHateClip follows Davidson et al. (2017), which aligns closely. The core definition across both is:

> **Content that promotes discrimination, disparages, or demeans individuals/groups based on protected attributes (race, ethnicity, religion, gender, sexual orientation, etc.)**

### Key observations for method design

1. **The Hateful/Offensive boundary is the bottleneck.** MHClip's IAA drops from 0.72 (binary) to 0.62/0.51 (3-class). 24-40% of disagreements are on this boundary. Any method claiming to distinguish hateful from offensive needs to model "targeting a protected group" — the only definitional difference.

2. **HateMM has no offensive category.** Offensive content in HateMM is folded into Non-Hate. This means the binary collapse in our pipeline (`offensive → 0` for HateMM, `offensive → 1` for MHClip) is correct but reflects genuinely different annotation schemes, not just a mapping choice.

3. **Platform policies focus on group-targeting.** All three platforms define hate via targeting protected groups. This supports a Concept Bottleneck approach where "target group identified" is a core precondition.

4. **BitChute's high threshold matters.** BitChute allows discriminatory language (NSFW tier) and only blocks active incitement. Videos in HateMM were collected from BitChute but annotated using YouTube's lower threshold. Some HateMM "hate" videos might be permitted content on BitChute itself.

5. **Bilibili's regulatory context differs.** Chinese hate speech norms are shaped by CAC regulation, not Western frameworks. Gender-based hate (misogyny/misandry) is specifically named by Bilibili — relevant for MHClip_ZH where women are the most targeted group.
