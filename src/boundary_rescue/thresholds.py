"""
Threshold criteria used by the boundary-rescue pipeline.

Otsu and GMM are re-exported from `src/our_method/quick_eval_all.py`.
li_lee is VENDORED from
  archive/post_shutdown_probes/probe_selector_scanfold.py:129-157
(archive scripts are not runnable in place, so we keep one copy here).
"""

import math
import os
import sys

import numpy as np

_OUR_METHOD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "our_method"
)
sys.path.insert(0, _OUR_METHOD)
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file  # noqa: E402,F401


def li_lee_threshold(scores, nbins=64):
    """Li-Lee minimum cross-entropy thresholding.

    Vendored byte-for-byte from
    archive/post_shutdown_probes/probe_selector_scanfold.py:129-157.
    """
    s = np.asarray(scores)
    hist, edges = np.histogram(s, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    hist = hist.astype(float)
    best_c = float("inf")
    best_b = nbins // 2
    for i in range(1, nbins - 1):
        m0_den = np.sum(hist[: i + 1])
        m1_den = np.sum(hist[i + 1 :])
        if m0_den < 1e-12 or m1_den < 1e-12:
            continue
        m0 = np.sum(hist[: i + 1] * centers[: i + 1]) / m0_den
        m1 = np.sum(hist[i + 1 :] * centers[i + 1 :]) / m1_den
        if m0 < 1e-12 or m1 < 1e-12:
            continue
        c = 0.0
        for k in range(nbins):
            if hist[k] < 1e-12 or centers[k] < 1e-12:
                continue
            if k <= i:
                c += hist[k] * centers[k] * math.log(centers[k] / m0)
            else:
                c += hist[k] * centers[k] * math.log(centers[k] / m1)
        if c < best_c:
            best_c = c
            best_b = i
    return float((edges[best_b] + edges[best_b + 1]) / 2)


CRITERION = {
    "MHClip_EN": ("otsu", otsu_threshold),
    "MHClip_ZH": ("gmm", gmm_threshold),
    "HateMM": ("li_lee", li_lee_threshold),
}
