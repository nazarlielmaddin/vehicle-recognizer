"""Embedding extractor + cosine retrieval over reference index."""
from __future__ import annotations
from pathlib import Path
import numpy as np

class EmbeddingStore:
    """Thin wrapper: FAISS if installed else brute-force numpy."""
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.ids: list[str] = []
        self.metas: list[dict] = []
        self.vecs: list[np.ndarray] = []
        self._faiss = None

    def add(self, vec: np.ndarray, ref_id: str, meta: dict):
        v = vec.astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        self.ids.append(ref_id); self.metas.append(meta); self.vecs.append(v)

    def _matrix(self) -> np.ndarray:
        return np.stack(self.vecs) if self.vecs else np.zeros((0, self.dim), np.float32)

    def search(self, q: np.ndarray, k: int = 5) -> list[tuple[str, float, dict]]:
        if not self.vecs:
            return []
        q = q.astype(np.float32); q /= (np.linalg.norm(q) + 1e-9)
        M = self._matrix()
        sims = M @ q
        idx = np.argsort(sims)[::-1][:k]
        return [(self.ids[i], float(sims[i]), self.metas[i]) for i in idx]

    def save(self, path: str):
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"ids": self.ids, "metas": self.metas, "vecs": self.vecs}, f)

    @classmethod
    def load(cls, path: str, dim: int = 512) -> "EmbeddingStore":
        import pickle
        s = cls(dim)
        p = Path(path)
        if p.exists():
            d = pickle.load(open(p, "rb"))
            s.ids, s.metas, s.vecs = d["ids"], d["metas"], d["vecs"]
        return s


class EmbeddingExtractor:
    """timm backbone with classification head removed → L2-normalized vector."""
    def __init__(self, backbone: str = "convnext_tiny.fb_in1k", weights: str = "models/embedding/best.pt",
                 dim: int = 512, device: str = "auto"):
        try:
            import torch
            self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        except ImportError:
            self.device = "cpu"  # torch missing → extract() raises honestly, pipeline continues with nn=[]
        self.backbone_name = backbone
        self.weights = Path(weights)
        self.available = self.weights.exists()
        self._model = None

    def _load(self):
        import torch, timm
        if self._model is None:
            m = timm.create_model(self.backbone_name, pretrained=True, num_classes=0)
            if self.available:
                sd = torch.load(self.weights, map_location="cpu")
                m.load_state_dict(sd.get("state_dict", sd), strict=False)
            m.eval().to(self.device)
            self._model = m
        return self._model

    def extract(self, rgb_nchw: np.ndarray) -> np.ndarray:
        import torch
        m = self._load()
        with torch.no_grad():
            t = torch.from_numpy(rgb_nchw).to(self.device)
            v = m(t).cpu().numpy()[0].astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v
