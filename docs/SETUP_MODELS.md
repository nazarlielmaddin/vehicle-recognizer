# docs — SETUP_MODELS: external weights isolation
All pretrained checkpoints are EXTERNAL dependencies (never committed).

1. Detector (auto): first `VehicleDetector.detect()` downloads `yolo11m.pt`
   via Ultralytics. Or manually: `yolo download yolo11m.pt` → `models/detector/`.
   Custom fine-tune: `python training/train_detector.py` → `models/detector/best.pt`,
   update `configs/default.yaml:paths.detector_weights`.
2. Classifiers/embeddings: train with scripts in Training order (README) →
   `models/{make,model,body_type,viewpoint,embedding}/best.pt` + registry entry.
3. Verify: `GET /health` shows trained flags + index size.
4. Licenses: Ultralytics AGPL-3.0 (use Enterprise for closed commercial);
   timm backbones per-checkpoint (mostly MIT/Apache — check model card);
   datasets research-only unless re-licensed (see docs/DATASETS.md).
