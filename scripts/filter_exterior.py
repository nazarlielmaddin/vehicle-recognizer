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
         min_area: float = 0.25, min_side: int = 160, ctx: float = 0.12,
         progress_every: int = 100):
    import time
    from inference.detector import VehicleDetector
    det = VehicleDetector(conf=0.30, iou=0.5)
    all_jpgs: list[Path] = []
    for root in (Path(raw) / "turboaz", Path(raw) / "turboaz_pilot"):
        if root.exists():
            all_jpgs.extend(sorted(root.rglob("*.jpg")))
    total = len(all_jpgs)
    print(f"Total input: {total}", flush=True)
    kept = drop = 0
    err: dict[str, int] = {"verify": 0, "tiny": 0, "decode": 0,
                           "detector": 0, "no_vehicle": 0, "small_box": 0,
                           "small_crop": 0, "imwrite": 0}
    per: dict[str, int] = {}
    t0 = time.time()
    for idx, jp in enumerate(all_jpgs, 1):
        if idx % progress_every == 0 or idx == total:
            el = time.time() - t0
            rate = idx / max(el, 1e-6)
            eta = (total - idx) / max(rate, 1e-6)
            print(f"{idx} / {total} ({idx/total*100:.2f}%) "
                  f"kept={kept} dropped={drop} ETA={eta/60:.1f}min", flush=True)
        try:
            mk = jp.parent.parent.name
            md = jp.parent.name
        except IndexError:
            err["verify"] += 1; drop += 1; continue
        if mk in MOTO_DIRS:
            continue  # out of scope, not an error
        try:
            with Image.open(jp) as im:
                im.verify()
            with Image.open(jp) as im:
                if min(im.size) < min_side:
                    err["tiny"] += 1; drop += 1; continue
        except Exception:
            err["verify"] += 1; drop += 1; continue
        blob = np.fromfile(str(jp), dtype=np.uint8)
        bgr = cv2.imdecode(blob, cv2.IMREAD_COLOR)
        if bgr is None:
            err["decode"] += 1; drop += 1; continue
        try:
            dets = det.detect(bgr)
        except Exception:
            err["detector"] += 1; drop += 1; continue
        if not dets:
            err["no_vehicle"] += 1; drop += 1; continue
        d = max(dets, key=lambda x: (x.x2 - x.x1) * (x.y2 - x.y1))
        h, w = bgr.shape[:2]
        area = (d.x2 - d.x1) * (d.y2 - d.y1) / (w * h)
        if area < min_area:
            err["small_box"] += 1; drop += 1; continue
        bw, bh = d.x2 - d.x1, d.y2 - d.y1
        x1, y1 = max(0, int(d.x1 - bw * ctx)), max(0, int(d.y1 - bh * ctx))
        x2, y2 = min(w, int(d.x2 + bw * ctx)), min(h, int(d.y2 + bh * ctx))
        crop = bgr[y1:y2, x1:x2]
        if min(crop.shape[:2]) < 100:
            err["small_crop"] += 1; drop += 1; continue
        dd = Path(out) / mk / md
        dd.mkdir(parents=True, exist_ok=True)
        lid = jp.stem.split("_")[0]
        if not cv2.imwrite(str(dd / f"{lid}.jpg"), crop):
            err["imwrite"] += 1; drop += 1; continue
        (dd / f"{lid}.meta.json").write_text(json.dumps(
            {"make": mk, "model": md, "src": str(jp), "box_area": round(area, 3),
             "det": d.label, "det_conf": round(d.conf, 3)}, ensure_ascii=False))
        kept += 1
        per[mk] = per.get(mk, 0) + 1
    print("=== FILTER ===", flush=True)
    print(f"Input: {total}", flush=True)
    print(f"Kept: {kept}", flush=True)
    print(f"Dropped: {drop}", flush=True)
    print(f"Keep rate: {kept/max(total,1)*100:.1f}%", flush=True)
    print(f"Errors: {err}", flush=True)
    for mk in sorted(per):
        print(f"  {mk}: {per[mk]}", flush=True)

if __name__ == "__main__":
    app()
