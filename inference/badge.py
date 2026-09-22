"""Badge/logo evidence channel.

Gate/barrier and radar frames are usually tight front close-ups where the
manufacturer emblem is clearly visible. CLIP model-name prompts
("Changan UNI-K") are near-OOV, but logo graphics ("Changan car logo")
are well represented in CLIP training data — so a dedicated logo read
is much stronger make evidence than a full-car zero-shot guess.
"""
from __future__ import annotations
import numpy as np

def badge_regions(crop_bgr: np.ndarray) -> list[np.ndarray]:
    """Candidate emblem areas for a FRONT-ish vehicle crop.

    Emblem sits near horizontal center, upper-middle band of the nose.
    Returns [wide_strip, tight_center] — caller averages both opinions.
    """
    h, w = crop_bgr.shape[:2]
    wide = crop_bgr[int(h * 0.25):int(h * 0.75), int(w * 0.30):int(w * 0.70)]
    tight = crop_bgr[int(h * 0.35):int(h * 0.65), int(w * 0.40):int(w * 0.60)]
    out = [c for c in (wide, tight) if c.size > 0 and min(c.shape[:2]) >= 24]
    return out or [crop_bgr]

def logo_prompts(makes: list[str]) -> list[str]:
    return [f"the {m} logo emblem on the front of a car" for m in makes]

class BadgeReader:
    def __init__(self, zero_shot):
        self.zs = zero_shot

    def read_make(self, crop_bgr: np.ndarray, makes: list[str],
                  k: int = 5) -> list[tuple[str, float]]:
        """Average logo-ranking over badge regions → [(make, prob)]."""
        prompts = logo_prompts(makes)
        acc: dict[str, float] = {}
        n = 0
        for region in badge_regions(crop_bgr):
            try:
                feat = self.zs.encode_image(region)
                top = self.zs.rank_with(feat, prompts, makes, k=len(makes))
            except Exception:
                continue
            n += 1
            for label, p in top:
                acc[label] = acc.get(label, 0.0) + p
        if not n:
            return [("Unknown", 0.0)]
        avg = sorted(((m, v / n) for m, v in acc.items()), key=lambda x: -x[1])
        return avg[:k]
