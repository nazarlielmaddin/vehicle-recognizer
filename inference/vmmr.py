"""VMMR expert head — EfficientNet-B4, 8,949 Make/Model/Year classes.

One-time download (models/vmmr/vehicle_classifier.pth), then fully local.
torch + timm only — no extra runtime. Complements our trained make head:
strong on 94 Western/JP makes incl. YEAR granularity; knows NO Chinese
brands (verified) so it defers to our head there.
"""
from __future__ import annotations
import re
from pathlib import Path
import cv2
import numpy as np

IMG = 380


def parse_mmy(label: str, known_makes: list[str]) -> tuple[str, str, str]:
    """'BMW 3 Series 2012' -> (BMW, 3 Series, 2012). Multiword makes handled."""
    parts = label.strip().split()
    year = ""
    if parts and re.fullmatch(r"(19|20)\d{2}", parts[-1]):
        year = parts.pop()
    for mk in sorted(known_makes, key=len, reverse=True):
        if " ".join(parts[:len(mk.split())]) == mk:
            rest = " ".join(parts[len(mk.split()):])
            return mk, rest, year
    if not parts:
        return "Unknown", "Unknown", year
    return parts[0], " ".join(parts[1:]), year


class VMMRExpert:
    def __init__(self, weights: str = "models/vmmr/vehicle_classifier.pth",
                 device: str = "cpu"):
        self.weights = Path(weights)
        self.device = device
        self.available = self.weights.exists()
        self._model = None
        self.mapping: dict = {}

    def _load(self):
        import torch
        import timm
        if self._model is None:
            ckpt = torch.load(self.weights, map_location="cpu")
            n = 8949
            mapping = ckpt.get("class_mapping", {})
            if isinstance(mapping, dict) and mapping:
                try:
                    n = max(int(k) for k in mapping.keys()) + 1
                except Exception:
                    pass
            m = timm.create_model("efficientnet_b4", pretrained=False, num_classes=n)
            m.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
            m.eval()
            self._model = m
            self.mapping = mapping or {}
        return self._model

    def predict(self, crop_bgr: np.ndarray, known_makes: list[str],
                k: int = 5) -> list[tuple[str, str, str, float]]:
        """-> [(make, model, year, prob)] top-k."""
        import torch
        if not self.available:
            return []
        m = self._load()
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (IMG, IMG),
                                      interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        t = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)[None]).float()
        with torch.no_grad():
            proba = torch.softmax(m(t), 1)[0].cpu().numpy()
        idx = np.argsort(proba)[::-1][:k]
        out = []
        try:
            from training.taxonomy import normalize_make
        except Exception:
            normalize_make = lambda s: s  # noqa: E731 (fallback: raw labels)
        for i in idx:
            raw = self.mapping.get(int(i), self.mapping.get(str(i), ""))
            if not raw:
                continue
            mk, md, yr = parse_mmy(str(raw), known_makes)
            out.append((normalize_make(mk), md, yr, float(proba[i])))
        return out
