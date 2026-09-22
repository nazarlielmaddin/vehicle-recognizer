# Vehicle Recognizer — parking/security-camera robust, production-oriented

> Answers **"What vehicle is this?"** from difficult real-world images, and answers
> **"I don't have enough evidence"** when it can't. Accuracy > forced predictions.

## Architecture

```
upload → quality analysis → YOLO detection → aspect-aware crops → viewpoint
→ timm make/model/body classifiers → embedding retrieval → late fusion
→ temperature calibration → UNKNOWN abstention → JSON + annotated image
```

Modular: every stage (`inference/*.py`) is independently replaceable/retrainable.
Single generic classifier is never blindly trusted — fusion + calibration +
embedding-distance + entropy + margin + quality gates decide CONFIDENT /
UNCERTAIN / UNKNOWN.

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
# 1) detector weights (auto-downloads yolo11m.pt on first run)
# 2) build taxonomy + (after training) retrieval index
python scripts/discover_taxonomy.py --out data/metadata/taxonomy.json
uvicorn api.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000  → drag & drop 1–8 images
```

CLI:
```bash
python scripts/run_inference.py single photo.jpg
python scripts/run_inference.py fused front.jpg rear.jpg side.jpg
python scripts/run_inference.py build-index
```

## Training order (PHASE 6–17)

```bash
python training/benchmark.py                       # pick backbone empirically
python training/train_detector.py --data data/processed/det
python training/train_make.py                      # make classifier
python training/train_finegrained.py               # model classifier (focal + balanced sampler)
python training/train_body.py                      # body-type classifier
python training/train_embedding.py                 # ArcFace/triplet embeddings
python scripts/run_inference.py build-index
python scripts/mine_hard_negatives.py
python scripts/calibrate_confidence.py
python -m evaluation.evaluate
```

See `docs/TRAINING.md`, `docs/DATASETS.md`, `docs/OPERATIONS.md`.

## Honest limitations

- Classifiers ship **untrained** until you run the training pipeline; until then
  the API returns `UNKNOWN` with reason `classifiers untrained` — never fake confidence.
- Supported taxonomy grows via `data/metadata/taxonomy.json` — no app rebuild needed.
- Real surveillance/night data is **required** for night robustness; daylight-only
  training will not generalize (see `configs/augmentation.yaml` night simulation
  as a supplement, not a substitute).
- Per-condition metrics (day/night/viewpoint/occlusion/blur) in `reports/eval.json`;
  a single overall accuracy is never the only metric.
