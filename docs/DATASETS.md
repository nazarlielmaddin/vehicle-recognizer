# docs — DATASETS: multi-source strategy + provenance + splits
Combine STUDIO + ROAD + SURVEILLANCE + PARKING + NIGHT + MULTI-VIEW.
Seed/benchmark: StanfordCars (16,185 imgs / 196 classes — NOT sole source).
Surveillance core: BoxCars116k (116k, traffic cams, 3D bbox + viewpoint),
VeRi-776 (50k, multi-cam ReID,Attrs), CompCars (214k, parts/viewpoints),
Visual Census (712k, large supplement). See `data/sources.py` + `reports/dataset_sources.json`.

Rules:
- Every sample: provenance {source, url, dataset, license, acquired, make/model/
  generation/year/body/viewpoint/lighting/quality/camera, preprocessing[]}.
- Normalize makes/models via `training/taxonomy.py` (Mercedes/VW aliases etc.).
- QC: `scripts/deduplicate_dataset.py` (pHash), `scripts/validate_dataset.py`
  (corrupt/tiny/leak checks). Near-dups stay in ONE split (identity-aware,
  `training/common.py:identity_aware_split` — same vehicle/video/family).
- Viewpoint coverage report per class (front/rear/side/diagonal %); pipeline
  warns on severe imbalance. Clean val/test = REAL images, never synthetic augments.
- `data/processed/dataset_version.json` pins sources/counts/mapping/aug/splits
  for every training run. Never scrape against ToS/robots.txt/rate limits.
