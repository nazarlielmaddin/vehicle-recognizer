"""Fusion + temperature calibration + UNKNOWN abstention logic."""
from __future__ import annotations
import math
import numpy as np

def softmax_with_temperature(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / max(T, 1e-3)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()

def entropy(proba: np.ndarray) -> float:
    p = np.clip(proba, 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())

def fuse_classifier_retrieval(cls_top: list[tuple[str, float]],
                              nn: list[tuple[str, float, dict]],
                              w_cls: float = 0.6, w_ret: float = 0.4) -> list[tuple[str, float]]:
    """Late fusion: classifier mass + retrieval votes mapped to class labels."""
    scores: dict[str, float] = {}
    for label, p in cls_top:
        scores[label] = scores.get(label, 0.0) + w_cls * p
    for _id, sim, meta in nn:
        label = meta.get("class", _id)
        scores[label] = scores.get(label, 0.0) + w_ret * max(0.0, sim) / max(len(nn), 1)
    tot = sum(scores.values()) + 1e-9
    return sorted(((k, v / tot) for k, v in scores.items()), key=lambda x: -x[1])

def decide_unknown(top1_conf: float, top2_conf: float, ent: float,
                   nn_sim: float, quality: str,
                   min_conf=0.55, min_margin=0.15, max_ent=1.6,
                   min_nn=0.45, poor_abstain=True) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if top1_conf < min_conf:
        reasons.append(f"conf {top1_conf:.2f} < {min_conf}")
    if (top1_conf - top2_conf) < min_margin:
        reasons.append(f"margin {(top1_conf-top2_conf):.2f} < {min_margin}")
    if ent > max_ent:
        reasons.append(f"entropy {ent:.2f} > {max_ent}")
    if nn_sim < min_nn:
        reasons.append(f"nn_sim {nn_sim:.2f} < {min_nn}")
    if poor_abstain and quality == "Poor" and top1_conf < 0.8:
        reasons.append("poor image quality")
    return (len(reasons) > 0, reasons)

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    acc = (pred == labels)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)
