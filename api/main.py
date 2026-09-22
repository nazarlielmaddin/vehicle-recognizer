"""FastAPI service: upload → pipeline → structured JSON + annotated images.

NOTE: endpoints are sync `def` (not async) on purpose — YOLO/CLIP inference is
CPU-blocking, and sync endpoints run in FastAPI's threadpool so /health stays
responsive while a prediction is being computed.
"""
from __future__ import annotations
import io, time, uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference.pipeline import VehiclePipeline

MAX_MB = 15
ALLOWED = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGES = 8

app = FastAPI(title="Avtomobil Tanıma", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pipe: VehiclePipeline | None = None

def get_pipe() -> VehiclePipeline:
    global pipe
    if pipe is None:
        pipe = VehiclePipeline()
    return pipe

def read_image(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"korlanmış/oxunmaz şəkil: {e}")
    if img.width < 32 or img.height < 32:
        raise HTTPException(400, "şəkil çox kiçikdir")
    if img.width > 8000 or img.height > 8000:
        raise HTTPException(400, "şəkil ölçüsü çox böyükdür")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

@app.get("/health")
def health():
    p = get_pipe()
    return {"status": "ok",
            "detector": p.detector.ready,
            "make_trained": p.make_clf.available,
            "model_trained": p.model_clf.available,
            "body_trained": p.body_clf.available,
            "index_size": len(p.store.ids)}

def _predict_one(p: VehiclePipeline, filename: str, content_type: str, data: bytes) -> dict:
    if content_type not in ALLOWED:
        raise HTTPException(400, f"dəstəklənməyən format {content_type} ({filename})")
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"{filename} {MAX_MB} MB-dan böyükdür")
    if len(data) == 0:
        raise HTTPException(400, f"{filename} boşdur")
    bgr = read_image(data)
    t0 = time.time()
    try:
        r = p.infer_image(bgr)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    for v in r.get("vehicles", []):
        x1, y1, x2, y2 = map(int, v["box"])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(bgr, f"#{v['id']} {v['model']} {v['confidence']:.2f}",
                    (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    _, buf = cv2.imencode(".jpg", bgr)
    import base64
    r["annotated_jpeg_b64"] = base64.b64encode(buf.tobytes()).decode()
    r["filename"] = filename
    r["latency_ms"] = round((time.time() - t0) * 1000, 1)
    r["request_id"] = str(uuid.uuid4())
    return r

@app.post("/predict")
def predict(files: list[UploadFile] = File(...)):
    if len(files) > MAX_IMAGES:
        raise HTTPException(400, f"maksimum {MAX_IMAGES} şəkil")
    p = get_pipe()
    return JSONResponse({"results": [
        _predict_one(p, f.filename, f.content_type, f.file.read()) for f in files
    ]})

@app.post("/predict-fused")
def predict_fused(files: list[UploadFile] = File(...)):
    """Eyni avtomobilin bir neçə görünüşü → vahid birləşmiş nəticə."""
    if len(files) < 2:
        raise HTTPException(400, "birləşdirmə üçün eyni avtomobilin 2+ şəklini yükləyin")
    if len(files) > MAX_IMAGES:
        raise HTTPException(400, f"maksimum {MAX_IMAGES}")
    p = get_pipe()
    try:
        imgs = [read_image(f.file.read()) for f in files]
        return JSONResponse(p.infer_multi_view(imgs))
    except RuntimeError as e:
        raise HTTPException(503, str(e))

app.mount("/", StaticFiles(directory="web", html=True), name="web")
