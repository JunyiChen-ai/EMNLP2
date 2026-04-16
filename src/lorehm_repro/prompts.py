"""LoReHM prompts — upstream strings with meme → video rewrite
(rework 2026-04-15: drop the 2x4 grid framing, describe the prompt
as "the 16 frames of the video").

Upstream: `external_repos/lorehm/utils/constants.py:1-5`.

Substring rewrites applied to upstream BASIC_PROMPT / RSA_PROMPT:
  * `"meme"` → `"video"`
  * `"embedded in the image"` → `"over the 16 frames of the video"`
  * upstream single-image framing -> "the 16 frames of the video"

Everything else — including the `Thought:` / `Answer:` output
format, the `harmful / harmless` label vocabulary, and the
"classifier has labeled ... please review the classifier's
judgment" RSA framing — is preserved byte-for-byte.

MIA insights are rewritten per-dataset to reference video content
instead of meme content; unchanged from the prior delivery.
"""

# Substring rewrite of upstream `constants.py:1` BASIC_PROMPT:
#   meme -> video (x2)
#   "the Text: \"{}\" embedded in the image" ->
#       "the transcript: \"{}\" over the 16 frames of the video"
# Everything else byte-for-byte.
BASIC_PROMPT = (
    'Given the video, with the transcript: "{}" over the 16 frames of '
    'the video, your task is to determine whether this video is harmful '
    'or not, in order to maintain the harmlessness and integrity of '
    'information on the Internet. Please leverage your extensive '
    'knowledge to deeply analyze and understand this video, and give '
    'your final judgment.\n'
    'Your output should stricly follow the format: "Thought: [Your '
    'analysis]\\nAnswer: [harmful/harmless].'
)

# Substring rewrite of upstream `constants.py:4-5` RSA_PROMPT with
# the same meme->video / "embedded in the image"->"over the 16 frames
# of the video" substitutions. The second `{}` slot is filled with
# `"harmful"` or `"harmless"` (upstream `main.py:61, 85`,
# `labels_str = ['harmless', 'harmful']`). The "classifier has
# labeled / please review / disagree / thorough and persuasive
# reasons" framing is byte-for-byte upstream.
RSA_PROMPT = (
    'Given the video, with the transcript: "{}" over the 16 frames of '
    'the video, your task is to determine whether this video is harmful '
    'or not, in order to maintain the harmlessness and integrity of '
    'information on the Internet. A classifier that can identify common '
    'features among multiple videos has labeled this video as {}. '
    'Please review the classifier\'s judgment carefully and use your '
    'extensive knowledge to analyze and understand this video before '
    'providing your final verdict. If you disagree with the classifier\'s '
    'judgment, you must provide exceptionally thorough and persuasive '
    'reasons.\n'
    'Your output should strictly follow the format: "Thought: [Your '
    'analysis]\\nAnswer: [harmful/harmless]."\n'
)

# Upstream label string list (`main.py:61`), verbatim.
LABELS_STR = ["harmless", "harmful"]


# -------------------- MIA insights, rewritten for video --------------------
# Source: `utils/constants.py:7-32` upstream (llava-v1.6-34b lists). We
# rewrite the hand-written strings to reference video content instead
# of meme content and substitute per-dataset domain framings. Each
# list is the same length as its upstream counterpart; the i-th hint
# lines up 1:1 with the upstream i-th hint. Rewrites are conservative
# — meme-specific nouns (`meme`, `image`, `caption`) are replaced
# with video equivalents (`video`, `frames`, `transcript`) and
# otherwise the advice content is preserved.

MHCLIP_EN_INSIGHTS = (
    "1. Always consider the potential for double meanings, innuendos, and cultural stereotypes in the transcript of a video.\n"
    "2. Evaluate the context in which the video could be shared, its intent, and its potential impact on different audiences.\n"
    "3. Be cautious of videos that could be interpreted as promoting or trivializing inappropriate or harmful behavior toward identity groups.\n"
    "4. Consider the broader social and cultural implications of the video's visual content and transcript, while also weighing the intent, humor, and factual accuracy behind it.\n"
    "5. Be sensitive to cultural stereotypes and flag videos that could perpetuate or reinforce negative stereotypes about any group.\n"
    "6. Differentiate between provocative or controversial statements and genuinely harmful content, ensuring that the video's potential to cause real harm is the primary consideration."
)

MHCLIP_ZH_INSIGHTS = (
    "1. Always consider the potential for double meanings, innuendos, and cultural stereotypes in the Chinese transcript of a video, and be aware that Chinese-language slang and slurs may carry region- or community-specific meanings.\n"
    "2. Evaluate the context in which the video could be shared on Chinese-language platforms, its intent, and its potential impact on different audiences.\n"
    "3. Be cautious of videos that could be interpreted as promoting or trivializing inappropriate or harmful behavior toward identity groups, including regional, ethnic, or gender-based targets common in Chinese-language discourse.\n"
    "4. Consider the broader social and cultural implications of the video's visual content and transcript, while also weighing the intent, humor, and factual accuracy behind it.\n"
    "5. Be sensitive to cultural stereotypes and flag videos that could perpetuate or reinforce negative stereotypes about any group.\n"
    "6. Differentiate between provocative or controversial statements and genuinely harmful content, ensuring that the video's potential to cause real harm is the primary consideration."
)

HATEMM_INSIGHTS = (
    "1. Assess the overall impact of the video, including whether it is likely to be perceived as light-hearted or offensive, especially in sensitive contexts such as hate speech, extremist rhetoric, social issues, or activism. Consider whether it trivializes serious situations, promotes discriminatory attitudes, spreads misinformation, or is insensitive to targeted groups, even if the intent is humorous.\n"
    "2. Consider the context and intent of the video, especially if it is meant to be humorous or relatable, but also weigh the potential for it to be perceived as insensitive, harmful, or misleading in serious situations.\n"
    "3. Evaluate whether the video directly or indirectly promotes harmful stereotypes, ideas, or personal attacks against identity groups, trivializes serious issues or individuals' efforts, spreads misinformation, or could be perceived as insensitive or harmful.\n"
    "4. Evaluate whether the video trivializes serious situations, issues, or the efforts of activists, which could be harmful or offensive to those affected, and consider the potential for it to spread misinformation, encourage harmful behavior, or be perceived as insensitive.\n"
    "5. Evaluate whether the video could potentially encourage harmful actions or behaviors against targeted groups. Consider the potential for it to downplay the severity of discrimination, spread misinformation, or be perceived as insensitive, even if the intent is humorous.\n"
    "6. Consider the potential for the video to spread misinformation, harmful stereotypes, or contribute to harmful narratives, even if it is intended as a joke.\n"
    "7. Pay special attention to videos that reference sensitive social, political, or cultural issues, as these are more likely to be harmful.\n"
    "8. Assess the potential for the video to be interpreted in a harmful or misleading way by different audiences, considering the broader social, political, and cultural context, and the potential for it to spread misinformation or harmful narratives.\n"
    "9. Evaluate the potential for the video to be misinterpreted by different audiences, leading to harmful consequences, even if the intent is humorous, and consider the broader social and cultural context.\n"
    "10. Consider the potential for the video to be interpreted as promoting or endorsing harmful actions or behaviors against identity groups."
)

IMPLIHATEVID_INSIGHTS = (
    "1. Always consider the potential for double meanings, implied insults, coded language, and cultural stereotypes in the transcript of a video — implicit hate is the dominant pattern in this dataset.\n"
    "2. Evaluate the context in which the video could be shared, its intent, and its potential impact on different audiences, keeping in mind that implicit hateful content often relies on irony, sarcasm, or dog-whistles rather than explicit slurs.\n"
    "3. Be cautious of videos that could be interpreted as promoting or trivializing harmful behavior toward identity groups through indirect means.\n"
    "4. Consider the broader social and cultural implications of the video's visual content and transcript, while also weighing the intent, humor, and factual accuracy behind it.\n"
    "5. Be sensitive to cultural stereotypes and flag videos that could perpetuate or reinforce negative stereotypes about any group, even when the language is superficially neutral.\n"
    "6. Differentiate between provocative or controversial statements and genuinely harmful content, ensuring that the video's potential to cause real harm is the primary consideration."
)


# v2 brief uses the upstream name `MEME_INSIGHTS` (even though the
# content is video-domain) so the dict key matches upstream's
# `utils/constants.py:MEME_INSIGHTS` exactly. `VIDEO_INSIGHTS` is
# kept as an alias for backwards compatibility with the v1 delivery.
MEME_INSIGHTS = {
    "llava-v1.6-34b": {
        "MHClip_EN": MHCLIP_EN_INSIGHTS,
        "MHClip_ZH": MHCLIP_ZH_INSIGHTS,
        "HateMM": HATEMM_INSIGHTS,
        "ImpliHateVid": IMPLIHATEVID_INSIGHTS,
    }
}
VIDEO_INSIGHTS = MEME_INSIGHTS  # v1 alias, prefer MEME_INSIGHTS going forward
