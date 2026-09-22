"""Image-quality analysis — runs BEFORE recognition, gates confidence."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import cv2
import numpy as np

@dataclass
class QualityProfile:
    width: int
    height: int
    resolution_label: str
    blur_variance: float
    is_blurred: bool
    brightness: float
    contrast: float
    exposure: str            # UNDER | NORMAL | OVER
    noise_estimate: float
    lighting: str            # DAY | DUSK | NIGHT | LOW_LIGHT | ARTIFICIAL_LIGHT | HEADLIGHT_DOMINATED | BACKLIT
    quality: str             # Good | Acceptable | Poor
    recognition_strength: str  # High | Medium | Low
    reasons: list

def laplacian_blur(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def estimate_noise(gray: np.ndarray) -> float:
    # median-absolute-deviation of high-pass residual, fast ISO-noise proxy
    hp = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(hp)))

def classify_lighting(mean: float, contrast: float, bright_frac: float, dark_frac: float) -> str:
    if mean < 35 and bright_frac > 0.02:
        return "HEADLIGHT_DOMINATED"
    if mean < 45:
        return "NIGHT"
    if mean < 70:
        return "LOW_LIGHT"
    if contrast > 75 and dark_frac > 0.25 and bright_frac > 0.15:
        return "BACKLIT"
    if 70 <= mean <= 170:
        return "DAY"
    return "ARTIFICIAL_LIGHT"

def analyze_quality(bgr: np.ndarray, blur_thr: float = 90.0) -> QualityProfile:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = laplacian_blur(gray)
    mean = float(gray.mean())
    contrast = float(gray.std())
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum() + 1e-6
    dark_frac = float(hist[:30].sum() / total)
    bright_frac = float(hist[225:].sum() / total)
    noise = estimate_noise(gray)
    lighting = classify_lighting(mean, contrast, bright_frac, dark_frac)
    if mean < 25:
        exposure = "UNDER"
    elif mean > 200:
        exposure = "OVER"
    else:
        exposure = "NORMAL"
    min_side = min(h, w)
    resolution_label = "high" if min_side >= 600 else ("medium" if min_side >= 224 else "low")
    reasons: list[str] = []
    if blur < blur_thr:
        reasons.append(f"blur var {blur:.1f} < {blur_thr}")
    if resolution_label == "low":
        reasons.append(f"low resolution {w}x{h}")
    if lighting in ("NIGHT", "LOW_LIGHT", "HEADLIGHT_DOMINATED"):
        reasons.append(f"lighting={lighting}")
    if exposure != "NORMAL":
        reasons.append(f"exposure={exposure}")
    if noise > 12:
        reasons.append(f"high sensor noise {noise:.1f}")
    if not reasons:
        quality, strength = "Good", "High"
    elif len(reasons) == 1 and blur >= blur_thr * 0.6 and resolution_label != "low":
        quality, strength = "Acceptable", "Medium"
    else:
        quality, strength = "Poor", "Low"
    return QualityProfile(w, h, resolution_label, blur, blur < blur_thr,
                          mean, contrast, exposure, noise, lighting,
                          quality, strength, reasons)

def to_dict(p: QualityProfile) -> dict:
    return asdict(p)
