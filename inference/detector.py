"""Vehicle detection wrapper (Ultralytics YOLO). Lazy-load, cached, CPU/GPU."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Detection:
    x1: float; y1: float; x2: float; y2: float
    conf: float; cls: int; label: str

_VEHICLE_NAMES = {"car", "bus", "truck"}

class VehicleDetector:
    def __init__(self, weights: str = "models/detector/yolo11m.pt",
                 conf: float = 0.35, iou: float = 0.5,
                 imgsz: int = 640, device: str = "auto"):
        self.weights = weights
        self.conf = conf; self.iou = iou; self.imgsz = imgsz
        self.device = None if device == "auto" else device
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            p = Path(self.weights)
            # fall back to COCO nano if custom weights not yet trained
            self._model = YOLO(str(p) if p.exists() else "yolo11m.pt")
        return self._model

    @property
    def ready(self) -> bool:
        try:
            self._load(); return True
        except Exception:
            return False

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        try:
            model = self._load()
        except Exception as e:
            raise RuntimeError(
                "vehicle-detector backend not installed. Run: "
                "pip install ultralytics torch torchvision --index-url https://download.pytorch.org/whl/cpu "
                f"(detail: {e})"
            )
        res = model.predict(bgr, conf=self.conf, iou=self.iou,
                            imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out: list[Detection] = []
        names = res.names
        for b in res.boxes or []:
            cls = int(b.cls[0]); label = str(names.get(cls, cls))
            if label not in _VEHICLE_NAMES:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append(Detection(x1, y1, x2, y2, float(b.conf[0]), cls, label))
        # largest-first (parking cams: closest vehicle usually most relevant)
        out.sort(key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1), reverse=True)
        return out
