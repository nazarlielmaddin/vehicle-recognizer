"""Pre-download + verify ML weights (YOLO detector, CLIP zero-shot). Run once per fresh machine."""
from __future__ import annotations
import numpy as np

def main() -> None:
    from inference.detector import VehicleDetector
    d = VehicleDetector()
    n = len(d.detect(np.zeros((640, 640, 3), np.uint8)))
    print(f"detector OK (blank-image boxes={n})")
    from inference.zeroshot import ClipZeroShot
    ClipZeroShot()._load()
    print("clip OK")

if __name__ == "__main__":
    main()
