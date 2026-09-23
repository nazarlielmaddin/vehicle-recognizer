"""Video (dashcam/barrier/phone) -> labeled-dataset staging.

Pipeline: sample frames (default 1 fps) -> YOLO vehicle detect ->
crop -> dHash dedup (video frames repeat!) -> staging/<n>.jpg + meta
(human labels makes/models afterwards via review_labels.py).
Usage: python scripts/extract_frames.py --video FILE.mp4 --out data/video_staging
"""
from __future__ import annotations
import json
from pathlib import Path
import typer
import cv2
import numpy as np
from PIL import Image

app = typer.Typer()


def dhash(bgr: np.ndarray, size: int = 16) -> str:
    g = cv2.cvtColor(cv2.resize(bgr, (size + 1, size)), cv2.COLOR_BGR2GRAY)
    bits = "".join("1" if g[r, c] > g[r, c + 1] else "0"
                   for r in range(size) for c in range(size))
    return f"{int(bits, 2):064x}"


def ham(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


@app.command()
def main(video: str = typer.Option(...), out: str = "data/video_staging",
         fps: float = 1.0, min_area: float = 0.04, ctx: float = 0.12,
         dedup_ham: int = 8):
    from inference.detector import VehicleDetector
    det = VehicleDetector(conf=0.35, iou=0.5)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video}")
        raise typer.Exit(1)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / fps))
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    seen: list[str] = []
    n_frame = n_crop = n_dup = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue
        idx += 1
        n_frame += 1
        try:
            dets = det.detect(frame)
        except Exception:
            continue
        h, w = frame.shape[:2]
        for d in dets:
            if (d.x2 - d.x1) * (d.y2 - d.y1) / (w * h) < min_area:
                continue
            bw, bh = d.x2 - d.x1, d.y2 - d.y1
            x1, y1 = max(0, int(d.x1 - bw * ctx)), max(0, int(d.y1 - bh * ctx))
            x2, y2 = min(w, int(d.x2 + bw * ctx)), min(h, int(d.y2 + bh * ctx))
            crop = frame[y1:y2, x1:x2]
            if min(crop.shape[:2]) < 100:
                continue
            dh = dhash(crop)
            if any(ham(dh, prev) <= dedup_ham for prev in seen):
                n_dup += 1
                continue
            seen.append(dh)
            name = f"{Path(video).stem}_f{idx}"
            cv2.imwrite(str(out_p / f"{name}.jpg"), crop)
            (out_p / f"{name}.meta.json").write_text(json.dumps(
                {"video": video, "frame": idx, "det": d.label,
                 "det_conf": round(d.conf, 3)}, ensure_ascii=False))
            n_crop += 1
        if n_frame % 300 == 0:
            print(f"frames={n_frame} crops={n_crop} dups={n_dup}", flush=True)
    cap.release()
    print(f"DONE frames={n_frame} crops={n_crop} dups_skipped={n_dup} -> {out}", flush=True)


if __name__ == "__main__":
    app()
