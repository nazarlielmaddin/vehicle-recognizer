"""Qwen2.5-VL expert — server-side (no CPU/RAM limits assumed there).

QEYD: Qwen2.5-VL "Qwen License" altındadır (Apache deyil) — dövlət
istifadəsindən əvvəl hüquqi rəy lazımdır. Bu, istifadəçinin açıq
göstərişi ilə inteqrasiya olunub.

Modes (configs/default.yaml -> vlm_server):
  local  - transformers direct. RAM: 3B~10GB / 7B~18GB / 32B~75GB (fp16).
           NOT for the 8GB notebook — server only.
  remote - OpenAI-compatible endpoint (vLLM):
             vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8001
Returns {make, model, year, body, raw, engine} or None. Never raises.
"""
from __future__ import annotations
import re
import numpy as np

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT = ("Look at the vehicle in this image. Identify it. Reply in EXACTLY "
          "this format on one line: Make: <brand>, Model: <model>, "
          "Year: <year or ?>, Body: <body type or ?>")


class QwenVLExpert:
    def __init__(self, model: str = DEFAULT_MODEL, mode: str = "local",
                 endpoint: str = "http://127.0.0.1:8001/v1",
                 max_tokens: int = 80):
        self.model = model
        self.mode = mode
        self.endpoint = endpoint.rstrip("/")
        self.max_tokens = max_tokens
        self._model = None
        self._proc = None

    # ---------- local ----------
    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            self._proc = AutoProcessor.from_pretrained(self.model)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model, dtype=torch.float16, device_map="auto")
            self._model.eval()
        return self._model

    def _ask_local(self, crop_bgr: np.ndarray) -> str:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        m = self._load()
        rgb = Image.fromarray(crop_bgr[:, :, ::-1])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": rgb}, {"type": "text", "text": PROMPT}]}]
        text = self._proc.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        images, _ = process_vision_info(messages)
        inp = self._proc(text=[text], images=images, padding=True,
                         return_tensors="pt").to(m.device, m.dtype)
        with torch.no_grad():
            out = m.generate(**inp, max_new_tokens=self.max_tokens)
        trimmed = out[:, inp.input_ids.shape[1]:]
        return self._proc.batch_decode(trimmed, skip_special_tokens=True)[0]

    # ---------- remote (vLLM OpenAI-compatible) ----------
    def _ask_remote(self, crop_bgr: np.ndarray) -> str:
        import base64
        import httpx
        _, buf = __import__("cv2").imencode(".jpg", crop_bgr,
                                            [int(__import__("cv2").IMWRITE_JPEG_QUALITY), 85])
        b64 = base64.b64encode(buf.tobytes()).decode()
        r = httpx.post(f"{self.endpoint}/chat/completions", timeout=300, json={
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            "max_tokens": self.max_tokens, "temperature": 0.1})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # ---------- public ----------
    def review(self, crop_bgr: np.ndarray) -> dict | None:
        try:
            raw = self._ask_remote(crop_bgr) if self.mode == "remote" \
                else self._ask_local(crop_bgr)
        except Exception as e:
            return {"make": "", "model": "", "year": "", "body": "",
                    "raw": f"qwen-vl failed: {str(e)[:120]}", "engine": "qwen2.5-vl"}
        tail = raw.split("Assistant:")[-1] if "Assistant:" in raw else raw
        out = {"make": "", "model": "", "year": "", "body": "",
               "raw": tail.strip()[:250], "engine": "qwen2.5-vl"}
        for key in ("make", "model", "year", "body"):
            m = re.search(rf"{key}:\s*([^\n,]+)", tail, re.IGNORECASE)
            if m:
                out[key] = m.group(1).strip(" .?")
        if not out["make"]:
            cand = re.sub(r"[^A-Za-z &\-]", "", tail).strip().split()
            if 1 <= len(cand) <= 2:
                out["make"] = " ".join(cand)
        return out
