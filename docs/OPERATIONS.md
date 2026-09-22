# docs — OPERATIONS: run, update, troubleshoot
Run: `uvicorn api.main:app --port 8000` → http://127.0.0.1:8000.
GPU: torch CUDA build; `system.device: cuda`. CPU fallback automatic.
Perf: models lazy-load once and cache; batch via /predict (≤8 imgs); index cached.
Security: MIME allowlist, 15MB cap, dimension/corruption checks, no exec of uploads,
temp files only, no path leakage in responses.
Add model: taxonomy entry → acquire (licensed) → validate → dedup → fine-tune →
eval old-vs-new → publish `models/*/<version>.pt` + registry (never overwrite prod).
Registry: `models/registry.json` {version, date, dataset_version, taxonomy_version,
hyperparams, metrics, checkpoint, git_commit, preprocessing}.
Troubleshoot: NO_VEHICLE → lower `detection.conf`; all-UNKNOWN → train classifiers
+ build index; night fails → add real night data; slow → imgsz 640→480, batch.
Limits: no guaranteed all-model accuracy; plates unused; studio-only training ≠ surveillance.
