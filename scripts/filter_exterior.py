"""Exterior QC: keep frames where YOLO finds a vehicle covering enough area.
Saves the vehicle CROP (what the classifier sees at inference), drops
interiors/details/documents/empty shots. Usage:
  $env:PYTHONPATH='.'; python scripts/filter_exterior.py
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import typer
import cv2
import numpy as np
from PIL import Image

app = typer.Typer()
MOTO_DIRS = {"Royal", "Can-Am", "Canam", "Benda", "Cfmoto", "Voge", "Yadea"}

@app.command()
def main(raw: str = "data/raw", out: str = "data/interim/crops",
         min_area: float = 0.25, min_side: int = 160, ctx: float = 0.12):
    from inference.detector import VehicleDetector
    det = VehicleDetector(conf=0.30, iou=0.5)
    kept = drop = 0
    per: dict[str, int] = {}
    for root in (Path(raw) / "turboaz", Path(raw) / "turboaz_pilot"):
        if not root.exists():
            continue
        for jp in sorted(root.rglob("*.jpg")):
            try:
                mk = jp.parent.parent.name
                md = jp.parent.name
            except IndexError:
                continue
            if mk in MOTO_DIRS:
                continue
            try:
                with Image.open(jp) as im:
                    im.verify()
                with Image.open(jp) as im:
                    if min(im.size) < min_side:
                        drop += 1; continue
            except Exception:
                drop += 1; continue
            blob = np.fromfile(str(jp), dtype=np.uint8)
            bgr = cv2.imdecode(blob, cv2.IMREAD_COLOR)
            if bgr is None:
                drop += 1; continue
            try:
                dets = det.detect(bgr)
            except Exception:
                drop += 1; continue
            if not dets:
                drop += 1; continue
            d = max(dets, key=lambda x: (x.x2 - x.x1) * (x.y2 - x.y1))
            h, w = bgr.shape[:2]
            area = (d.x2 - d.x1) * (d.y2 - d.y1) / (w * h)
            if area < min_area:
                drop += 1; continue
            bw, bh = d.x2 - d.x1, d.y2 - d.y1
            x1, y1 = max(0, int(d.x1 - bw * ctx)), max(0, int(d.y1 - bh * ctx))
            x2, y2 = min(w, int(d.x2 + bw * ctx)), min(h, int(d.y2 + bh * ctx))
            crop = bgr[y1:y2, x1:x2]
            if min(crop.shape[:2]) < 100:
                drop += 1; continue
            dd = Path(out) / mk / md
            dd.mkdir(parents=True, exist_ok=True)
            lid = jp.stem.split("_")[0]
            cv2.imwrite(str(dd / f"{lid}.jpg"), crop)
            (dd / f"{lid}.meta.json").write_text(json.dumps(
                {"make": mk, "model": md, "src": str(jp), "box_area": round(area, 3),
                 "det": d.label, "det_conf": round(d.conf, 3)}, ensure_ascii=False))
            kept += 1
            per[mk] = per.get(mk, 0) + 1
    print(f"kept={kept} dropped={drop}")
    for mk in sorted(per):
        print(f"  {mk}: {per[mk]}")

if __name__ == "__main__":
    app()
