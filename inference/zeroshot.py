"""Zero-shot fallback (OpenCLIP) — used ONLY when trained classifiers are absent.

Honesty contract:
- every result is tagged evidence_source="zero-shot-clip";
- pipeline caps status at UNCERTAIN for zero-shot-only evidence (never CONFIDENT);
- as soon as trained weights exist, TimmClassifier takes precedence automatically.
"""
from __future__ import annotations
import cv2
import numpy as np

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

PROMPT_T = "a photo of a {} car"
VIEW_PROMPTS = {
    "FRONT": "front view of a car, headlights and grille visible",
    "REAR": "rear view of a car, taillights and trunk visible",
    "LEFT": "left side view of a car",
    "RIGHT": "right side view of a car",
    "FRONT_LEFT": "front-left three-quarter view of a car",
    "FRONT_RIGHT": "front-right three-quarter view of a car",
    "REAR_LEFT": "rear-left three-quarter view of a car",
    "REAR_RIGHT": "rear-right three-quarter view of a car",
    "UNKNOWN": "a car",
}
BODY_PROMPTS = {
    "Sedan": "a sedan car", "Hatchback": "a hatchback car", "Coupe": "a coupe car",
    "Convertible": "a convertible cabriolet car", "Wagon": "a station wagon estate car",
    "SUV": "an SUV car", "Crossover": "a crossover car", "MPV": "a minivan MPV car",
    "Pickup": "a pickup truck", "Van": "a van", "PanelVan": "a panel van",
    "LightTruck": "a light truck", "HeavyTruck": "a heavy truck",
    "Bus": "a bus", "Minibus": "a minibus", "Other": "a vehicle",
    "Unknown": "a vehicle",
}

class ClipZeroShot:
    def __init__(self, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED,
                 device: str = "cpu"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model = None
        self._tok = None
        self._pre = None

    def _load(self):
        if self._model is None:
            import open_clip, torch
            m, _, pre = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained, device=self.device)
            m.eval()
            self._model, self._pre, self._tok = m, pre, open_clip.get_tokenizer(self.model_name)
        return self._model

    def encode_image(self, bgr: np.ndarray):
        """Encode once, reuse across heads (4x faster than per-head encode)."""
        import torch
        m = self._load()
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self._pre(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = m.encode_image(img)
            f /= f.norm(dim=-1, keepdim=True)
        return f

    def rank_with(self, feat, prompts: list[str], labels: list[str],
                  k: int = 5) -> list[tuple[str, float]]:
        """Rank precomputed image features against prompt list."""
        import torch
        m = self._load()
        toks = self._tok(prompts).to(self.device)
        with torch.no_grad():
            f_t = m.encode_text(toks)
            f_t /= f_t.norm(dim=-1, keepdim=True)
            sim = (100.0 * feat @ f_t.T).softmax(-1)[0].cpu().numpy()
        idx = np.argsort(sim)[::-1][:k]
        return [(labels[i], float(sim[i])) for i in idx]

    def rank(self, bgr: np.ndarray, labels: list[str],
             prompt_t: str = PROMPT_T, k: int = 5) -> list[tuple[str, float]]:
        """Cosine-similarity softmax over candidate label prompts."""
        import torch
        m = self._load()
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self._pre(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        toks = self._tok([prompt_t.format(l) for l in labels]).to(self.device)
        with torch.no_grad():
            f_i = m.encode_image(img)
            f_t = m.encode_text(toks)
            f_i /= f_i.norm(dim=-1, keepdim=True)
            f_t /= f_t.norm(dim=-1, keepdim=True)
            sim = (100.0 * f_i @ f_t.T).softmax(-1)[0].cpu().numpy()
        idx = np.argsort(sim)[::-1][:k]
        return [(labels[i], float(sim[i])) for i in idx]

    def rank_viewpoint(self, bgr: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        labels = list(VIEW_PROMPTS)
        prompts = [VIEW_PROMPTS[l] for l in labels]
        import torch
        m = self._load()
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self._pre(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        toks = self._tok(prompts).to(self.device)
        with torch.no_grad():
            f_i = m.encode_image(img)
            f_t = m.encode_text(toks)
            f_i /= f_i.norm(dim=-1, keepdim=True)
            f_t /= f_t.norm(dim=-1, keepdim=True)
            sim = (100.0 * f_i @ f_t.T).softmax(-1)[0].cpu().numpy()
        idx = np.argsort(sim)[::-1][:k]
        return [(labels[i], float(sim[i])) for i in idx]

    def rank_body(self, bgr: np.ndarray, labels: list[str], k: int = 5) -> list[tuple[str, float]]:
        import torch
        m = self._load()
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self._pre(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        toks = self._tok([BODY_PROMPTS.get(l, f"a {l}") for l in labels]).to(self.device)
        with torch.no_grad():
            f_i = m.encode_image(img)
            f_t = m.encode_text(toks)
            f_i /= f_i.norm(dim=-1, keepdim=True)
            f_t /= f_t.norm(dim=-1, keepdim=True)
            sim = (100.0 * f_i @ f_t.T).softmax(-1)[0].cpu().numpy()
        idx = np.argsort(sim)[::-1][:k]
        return [(labels[i], float(sim[i])) for i in idx]
