# DATASET RESEARCH — verified online 2026-09-23. Nothing assumed.

## Per-source verdicts

### CompCars — BENCHMARK ONLY
- URL: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html (alive, fetched)
- 163 makes / 1,716 models / 136,726 web + 27,618 parts + 50,000 surveillance
- Annotations: bbox, viewpoint, 5 attributes, train/test splits included
- Chinese brands: YES (best content fit)
- License: **non-commercial research only; no redistribution; no commercial use**
- Download: instruction file on page, author agreement expected, tens of GB
- Taxonomy fit: high — mapping work only
- Verdict: license blocks production/state use. Benchmark-only candidate.

### Stanford Cars/Cars196 — DIVERSITY SUPPLEMENT (gated)
- Original URL DEAD (404 confirmed 2026-09-23; torchvision docs confirm)
- Mirrors: huggingface.co/datasets/tanganke/stanford_cars (6.02 GB, 72k rows
  incl. robustness variants, 17k pulls/month — ACTIVE), GitHub jhpohovey mirror,
  Kaggle copies (no local creds)
- 16,185 images / ~50 makes / 196 classes (Make Model Year), train 8144/test 8041
- Chinese brands: NO. Surveillance: NO.
- License: original research use; mirrors carry NO clear license
- Verdict: optional Western-make diversity; production use needs legal review.
  Handler implemented: scripts/download_dataset.py --source stanford-hf
  (requires `pip install datasets pyarrow` + --confirm-license).

### BoxCars116k — UNAVAILABLE
- Repo alive (JakubSochor/BoxCars, 209 stars, research-only code+data, TF .h5 weights)
- 116,826 surveillance images / ~45 makes / 693 models, 3D boxes, tracks, splits
- Data server https://medusa.fit.vutbr.cz/.../BoxCars116k.zip: **DNS-dead (verified)**
- Chinese brands: NO (Czech traffic). TF weights not directly integrable.
- Verdict: revisit if server returns.

### VMMRdb — SKIP
- 9,170 classes / 291,752 images, make/model/year (repo faezetta/VMMRdb, 174 stars)
- Verified against class_mapping.csv: 94 makes, **0 Changan/BYD/Haval/Lada/Moskvich**
- Download via Dropbox (unverified); no explicit commercial license (academic)
- Old Torch .t7 reference models
- Verdict: wrong market + murky license. Not integrated.

### VeRi-776 — SKIP (classification)
- 50k multi-camera images, 776 vehicles, ~10 makes; research-only, author request
- ReID task, not fine-grained make/model. Noted for future ReID stage only.

### HF mirrors (CompCars etc.) — SAME BLOCK
- JorgeLlorente/CompCars-Repository (cc-by-nc-4.0): underlying data still
  CompCars research-only. Benchmark only.

### Kaggle sets — SKIP (unreviewed)
- No local credentials; re-uploads with murky per-dataset licensing.
- Any future candidate needs individual license review first.

### turbo.az collection (OURS) — PRODUCTION SOURCE
- Working polite collector (robots-checked, delays, provenance per image)
- ~1,350 raw frames → YOLO exterior QC → 773 verified crops, 66 makes
- Native fit to 77/380 taxonomy; parking-like + dealership views (DAY mostly —
  night/surveillance gap documented for next collection round)
- License: listing images (c) owners/turbo.az; training use with provenance;
  **state deployment needs legal review** (stated openly, not hidden)

## Conclusion
No legally-usable external dataset beats our first-party collection for this
market. External sources stay BENCHMARK/diversity-gated behind license flags.
The integration pipeline (discover → download → normalize → map → dedup →
prepare → audit) is implemented and tested so any cleared source plugs in.
