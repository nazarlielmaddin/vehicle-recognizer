"""Aspect-ratio-aware crop extraction. Never squash geometry."""
from __future__ import annotations
import cv2
import numpy as np
from .detector import Detection

def expand_box(d: Detection, w: int, h: int, ctx: float = 0.12) -> tuple[int,int,int,int]:
    bw, bh = d.x2 - d.x1, d.y2 - d.y1
    x1 = max(0, int(d.x1 - bw * ctx)); y1 = max(0, int(d.y1 - bh * ctx))
    x2 = min(w, int(d.x2 + bw * ctx)); y2 = min(h, int(d.y2 + bh * ctx))
    return x1, y1, x2, y2

def letterbox(bgr: np.ndarray, size: tuple[int,int] = (224,224),
              color: tuple[int,int,int] = (114,114,114)) -> np.ndarray:
    """Resize preserving aspect ratio + pad (YOLO-style letterbox)."""
    th, tw = size[1], size[0]
    h, w = bgr.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(w * s), int(h * s)
    r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    canvas[(th-nh)//2:(th-nh)//2+nh, (tw-nw)//2:(tw-nw)//2+nw] = r
    return canvas

def extract_crops(bgr: np.ndarray, dets: list[Detection], ctx: float = 0.12) -> list[dict]:
    """Returns full + tight + detail crops per detection."""
    h, w = bgr.shape[:2]
    crops = []
    for d in dets:
        x1, y1, x2, y2 = expand_box(d, w, h, ctx)
        full = bgr[y1:y2, x1:x2]
        if full.size == 0:
            continue
        bw, bh = x2 - x1, y2 - y1
        # tight crop (no context) for detail, full crop for silhouette
        tx1, ty1 = int(d.x1), int(d.y1)
        tight = bgr[max(0,ty1):int(d.y2), max(0,tx1):int(d.x2)]
        crops.append({
            "box": [x1, y1, x2, y2], "conf_det": d.conf, "label_det": d.label,
            "full": full, "tight": tight if tight.size else full,
            "area_frac": float(bw * bh / (w * h)),
        })
    return crops
