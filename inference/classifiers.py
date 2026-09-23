"""Generic timm classifier wrapper with honest 'untrained' state.

Never fakes predictions: if checkpoint is missing the classifier reports
available=False and the pipeline abstains / marks UNKNOWN instead of
inventing a confidence number.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

class TimmClassifier:
    def __init__(self, weights: str, classes: list[str], backbone: str = "convnext_tiny.fb_in1k",
                 img_size: int = 224, device: str = "auto"):
        self.weights = Path(weights)
        self.classes = classes
        self.backbone = backbone
        self.img_size = img_size
        self.device = self._resolve(device)
        self._model = None
        self.available = self.weights.exists()

    @staticmethod
    def _resolve(device: str):
        try:
            import torch
        except ImportError:
            return "cpu"  # torch not installed → classifiers stay unavailable, pipeline abstains honestly
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load(self):
        import torch, timm
        if self._model is None:
            ckpt = {}
            if self.available:
                ckpt = torch.load(self.weights, map_location="cpu")
                # adopt the checkpoint's own class list (train-time order);
                # taxonomy order may differ or be a superset
                if isinstance(ckpt, dict) and ckpt.get("classes"):
                    self.classes = list(ckpt["classes"])
                if isinstance(ckpt, dict) and ckpt.get("backbone"):
                    self.backbone = ckpt["backbone"]
            m = timm.create_model(self.backbone, pretrained=not self.available,
                                  num_classes=len(self.classes))
            if self.available:
                m.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
            m.eval().to(self.device)
            self._model = m
        return self._model

    def predict_proba(self, rgb_batch: np.ndarray) -> np.ndarray:
        """rgb_batch: float32 NCHW in [0,1]. Returns NxC probabilities."""
        import torch
        if not self.available:
            raise FileNotFoundError(f"weights not trained yet: {self.weights}")
        m = self._load()
        with torch.no_grad():
            t = torch.from_numpy(rgb_batch).to(self.device)
            return torch.softmax(m(t), dim=1).cpu().numpy()

    @staticmethod
    def topk(proba: np.ndarray, classes: list[str], k: int = 5) -> list[tuple[str, float]]:
        idx = np.argsort(proba)[::-1][:k]
        return [(classes[i], float(proba[i])) for i in idx]
