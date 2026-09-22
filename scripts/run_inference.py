"""CLI inference: single image / folder / multi-view fusion."""
from __future__ import annotations
import json
import cv2
import typer
from pathlib import Path

app = typer.Typer()

@app.command()
def single(image: str, config: str = "configs/default.yaml", out: str = ""):
    from inference.pipeline import VehiclePipeline
    p = VehiclePipeline(config)
    bgr = cv2.imread(image)
    assert bgr is not None, f"cannot read {image}"
    r = p.infer_image(bgr)
    s = json.dumps(r, indent=2)
    print(s)
    if out:
        Path(out).write_text(s, encoding="utf-8")

@app.command()
def fused(images: list[str], config: str = "configs/default.yaml"):
    from inference.pipeline import VehiclePipeline
    import cv2 as _cv
    p = VehiclePipeline(config)
    ims = [_cv.imread(i) for i in images]
    print(json.dumps(p.infer_multi_view(ims), indent=2))

@app.command()
def build_index(crops: str = "data/processed/crops/train", out: str = "embeddings/vehicle_index.faiss",
                config: str = "configs/default.yaml"):
    """Extract reference embeddings for the retrieval index."""
    import numpy as np
    from inference.pipeline import VehiclePipeline, _prep_rgb
    p = VehiclePipeline(config)
    from inference.retrieval import EmbeddingStore
    store = EmbeddingStore()
    for cls_dir in sorted(Path(crops).iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_p in list(cls_dir.glob("*.jpg"))[:50]:
            bgr = cv2.imread(str(img_p))
            if bgr is None:
                continue
            v = p.embed.extract(_prep_rgb(bgr))
            store.add(v, str(img_p), {"class": cls_dir.name})
    store.save(out)
    print(f"index: {len(store.ids)} vectors -> {out}")

if __name__ == "__main__":
    app()
