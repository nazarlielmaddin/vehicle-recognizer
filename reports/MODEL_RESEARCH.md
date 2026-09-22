# Research: model & dataset landscape (summary — full report in reports/).
# See reports/MODEL_RESEARCH.md and reports/DATASET_RESEARCH.md for the full tables.

## Detector
YOLO11 (Ultralytics, AGPL-3.0 / Enterprise) — recommended default `yolo11m.pt`.
Why: actively maintained, strong small-object mAP, `model.export(format="onnx")`
for ONNX/TensorRT, Python API stable. Alternatives evaluated: RT-DETR (accurate
but heavier), YOLOv8 (previous gen, same API). COCO classes car/bus/truck reused;
fine-tune on traffic data (BoxCars/VeRi boxes) for parking domain.

## Classifiers / backbones (timm)
Benchmark with `training/benchmark.py`; default `convnext_tiny.fb_in1k`:
best accuracy/latency/VRAM trade-off for 224px fine-grained vehicle ID.
Candidates: efficientnet_b3, convnext_small, swin_tiny, vit_small, resnet50.
All support ONNX export via torch.onnx. License: Apache-2.0 (timm) + per-weight
licenses (mostly MIT/Apache; verify each checkpoint card).

## Embeddings / metric learning
ArcFace head on timm backbone → 512-d L2 embeddings; FAISS (MIT) or numpy
fallback in `inference/retrieval.py`. ReID inspiration: VeRi-776 baselines
(AlignedReID, TransReID) — reuse ideas, not weights, unless license allows.

## Zero-shot / open-vocabulary fallback
OpenCLIP (MIT) `ViT-B-32/laion2b` as OPTIONAL review aid only — never as primary
evidence (too weak for fine-grained model separation + plate/background leakage).
CLIP text prompts: "a photo of a {make} {model} car". Disabled by default.

## Quality / OCR
Classical CV (Laplacian blur, brightness/contrast stats) in `inference/quality.py`
— no learned IQA dependency needed for v1. PaddleOCR (Apache-2.0) OPTIONAL for
future plate-region masking experiments only; identity must never depend on plates.

## Export / serving
PyTorch → ONNX (onnxruntime, CPU+GPU) → TensorRT (GPU) for detector;
classifiers export via `torch.onnx.export` (dynamic batch). All behind
`TimmClassifier`/`VehicleDetector` interfaces so backends are swappable.
