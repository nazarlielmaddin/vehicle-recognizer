"""SmolVLM review channel (HuggingFaceTB/SmolVLM-500M-Instruct, Apache-2.0).

Measured on i3 CPU: load ~7s, infer 22-43s/image. Knows Western makes,
does NOT know Chinese brands (says Toyota for Changan — verified).
Therefore REVIEW-ONLY: never overrides the tuned head; shown for the
operator + agreement micro-boost when it independently agrees.
Requires transformers==4.49.0 (v5 breaks SmolVLM input format — verified).
"""
from __future__ import annotations
import re
import numpy as np

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


class SmolVLMExpert:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._model = None
        self._proc = None
        self.available = True  # lazy; False only if load fails

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            try:
                # offline-first: cached weights must never hit the network
                self._proc = AutoProcessor.from_pretrained(self.model_id,
                                                           local_files_only=True)
                self._model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id, torch_dtype=torch.float32, local_files_only=True)
            except Exception:
                self._proc = AutoProcessor.from_pretrained(self.model_id)
                self._model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id, torch_dtype=torch.float32)
            self._model.eval()
        return self._model

    def review(self, crop_bgr: np.ndarray) -> dict | None:
        """Returns {make, model, raw} or None on any failure."""
        try:
            import torch
            from PIL import Image
            m = self._load()
            rgb = cv2_to_pil(crop_bgr)
            prompt = ("What is the make (brand) and model of the car? "
                      "Reply exactly: Make: <brand>, Model: <model>")
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self._proc.apply_chat_template(messages, add_generation_prompt=True)
            inp = self._proc(text=text, images=[rgb], return_tensors="pt")
            with torch.no_grad():
                out = m.generate(**inp, max_new_tokens=30)
            raw = self._proc.batch_decode(out, skip_special_tokens=True)[0]
            tail = raw.split("Assistant:")[-1].strip()
            mk = re.search(r"Make:\s*([^,\n]+)", tail)
            md = re.search(r"Model:\s*([^\n]+)", tail)
            make = mk.group(1).strip(" .") if mk else ""
            model = md.group(1).strip(" .") if md else ""
            if not make:
                # fallback: short free-form answer ("Toyota.") -> first words
                cand = re.sub(r"[^A-Za-z &\-]", "", tail).strip().split()
                if 1 <= len(cand) <= 2:
                    make = " ".join(cand)
            return {"make": make, "model": model, "raw": tail[:200]}
        except Exception as e:
            return {"make": "", "model": "", "raw": f"vlm failed: {str(e)[:100]}"}


def cv2_to_pil(bgr: np.ndarray):
    import cv2
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
