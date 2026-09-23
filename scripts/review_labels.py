"""Human-in-the-loop labeling (fast Windows flow).

Step 1 (suggest): run trained make head on staging crops ->
  review/<MAKE>/*.jpg (model pre-sorts, human fixes by drag-drop in Explorer).
Step 2 (finalize): move review/<MAKE>/*.jpg ->
  data/interim/crops/<MAKE>/video/ (ready for filter/prepare).
Usage:
  python scripts/review_labels.py suggest --staging data/video_staging
  # ... fix folders in Explorer ...
  python scripts/review_labels.py finalize
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import typer
import cv2
import numpy as np

app = typer.Typer()


@app.command()
def suggest(staging: str = "data/video_staging", out: str = "data/review"):
    from inference.pipeline import VehiclePipeline
    p = VehiclePipeline()
    if not p.make_clf.available:
        print("ERROR: no trained make head — train first")
        raise typer.Exit(1)
    from inference.pipeline import _prep_rgb
    n = 0
    for jp in sorted(Path(staging).glob("*.jpg")):
        blob = np.fromfile(str(jp), dtype=np.uint8)
        bgr = cv2.imdecode(blob, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = _prep_rgb(bgr, 192)
        try:
            top, _ = p._predict_head(p.make_clf, rgb)
        except Exception as e:
            print(f"predict failed {jp.name}: {str(e)[:80]}")
            continue
        mk = top[0][0].replace(" ", "_")
        d = Path(out) / mk
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(jp, d / jp.name)
        n += 1
    print(f"suggested {n} crops -> {out}/<MAKE>/ (fix mistakes in Explorer, then finalize)")


@app.command()
def finalize(review: str = "data/review"):
    n = 0
    for mkd in sorted(Path(review).iterdir()):
        if not mkd.is_dir():
            continue
        mk = mkd.name
        for jp in sorted(mkd.glob("*.jpg")):
            dd = Path("data/interim/crops") / mk / "video"
            dd.mkdir(parents=True, exist_ok=True)
            shutil.copy2(jp, dd / jp.name)
            (dd / f"{jp.stem}.meta.json").write_text(json.dumps(
                {"make": mk, "model": "Unknown", "src": str(jp),
                 "label": "human-confirmed-make", "model_pending": True},
                ensure_ascii=False))
            n += 1
    print(f"finalized {n} human-confirmed crops (model label pending)")


if __name__ == "__main__":
    app()
