"""Dataset source registry — every fact below was verified online 2026-09-23.
Statuses: available | dead | gated (agreement/form) | mirror-only.
License verdicts are explicit: NOTHING here is assumed commercial-safe.
Usage: python scripts/discover_datasets.py [--json]
"""
from __future__ import annotations
import json
import typer

app = typer.Typer()

SOURCES: list[dict] = [
    {
        "name": "CompCars", "kind": "academic",
        "url": "http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html",
        "status": "available-page / gated-download",
        "images": 136726, "parts": 27618, "surveillance": 50000,
        "makes": 163, "models": 1716,
        "annotations": ["bbox", "viewpoint", "5 attributes", "train/test splits"],
        "chinese_brands": True, "surveillance": True,
        "license": "non-commercial research only; no redistribution; no commercial use",
        "commercial": False,
        "download": "instruction file on page; author agreement expected; ~tens of GB",
        "taxonomy_fit": "high (163 makes incl. Chinese), needs mapping",
        "verdict": "BENCHMARK ONLY — license blocks production/state use",
    },
    {
        "name": "StanfordCars/Cars196", "kind": "academic",
        "url": "https://ai.stanford.edu/~jkrause/cars/car_dataset.html",
        "status": "original DEAD (404, confirmed); mirrors exist",
        "images": 16185, "makes": "~50", "models": 196,
        "annotations": ["make/model/year labels", "train(8144)/test(8041) split"],
        "chinese_brands": False, "surveillance": False,
        "license": "research use (original); mirrors carry no clear license",
        "commercial": False,
        "download": "mirror: huggingface.co/datasets/tanganke/stanford_cars (6GB, active); "
                    "github.com/jhpohovey/StanfordCars-Dataset; kaggle copies (need account)",
        "taxonomy_fit": "partial (Western makes overlap ours); zero Chinese brands",
        "verdict": "OPTIONAL diversity supplement for Western makes; needs license review before production",
    },
    {
        "name": "BoxCars116k", "kind": "academic",
        "url": "https://github.com/JakubSochor/BoxCars",
        "status": "repo alive (209 stars); DATA SERVER DEAD (DNS fail, confirmed)",
        "images": 116826, "makes": "~45", "models": 693,
        "annotations": ["3D boxes", "tracks/multi-view", "splits", "TF .h5 checkpoints"],
        "chinese_brands": False, "surveillance": True,
        "license": "research only (code + data)",
        "commercial": False,
        "download": "https://medusa.fit.vutbr.cz/traffic/data/BoxCars116k.zip — UNREACHABLE",
        "taxonomy_fit": "medium (EU/Czech market); TF weights not directly integrable",
        "verdict": "UNAVAILABLE — revisit if server returns; TF weights need conversion anyway",
    },
    {
        "name": "VMMRdb", "kind": "academic",
        "url": "https://github.com/faezetta/VMMRdb",
        "status": "repo alive (174 stars); data via Dropbox link (unverified here)",
        "images": 291752, "makes": "~100", "models": 9170,
        "annotations": ["make/model/year labels"],
        "chinese_brands": False, "surveillance": False,
        "license": "no explicit commercial license (academic paper) → treat as research-only",
        "commercial": False,
        "download": "dropbox (see repo README); old Torch .t7 reference models",
        "taxonomy_fit": "low for our market (verified: 0 Changan/BYD/Haval/Lada classes)",
        "verdict": "SKIP — wrong market + murky license",
    },
    {
        "name": "VeRi-776", "kind": "academic",
        "url": "https://github.com/JingdongDeployment/VeRi",
        "status": "known-good (not re-fetched this round)",
        "images": 50000, "makes": "~10", "models": 776,
        "annotations": ["multi-camera tracks", "reid splits"],
        "chinese_brands": "partial", "surveillance": True,
        "license": "research only",
        "commercial": False,
        "download": "author request",
        "taxonomy_fit": "low (reid task, few makes)",
        "verdict": "SKIP for classification; reviewer-noted for future ReID stage",
    },
    {
        "name": "HF CompCars mirrors", "kind": "mirror",
        "url": "https://huggingface.co/datasets (JorgeLlorente/CompCars-Repository, ...)",
        "status": "listed, tiny usage",
        "images": "same as CompCars", "makes": 163, "models": 1716,
        "annotations": ["inherits CompCars"],
        "chinese_brands": True, "surveillance": True,
        "license": "cc-by-nc-4.0 on mirror (underlying data still CompCars research-only)",
        "commercial": False,
        "download": "huggingface_hub snapshot (needs agreement in spirit)",
        "taxonomy_fit": "high",
        "verdict": "SAME license block as CompCars — benchmark only",
    },
    {
        "name": "Kaggle vehicle sets", "kind": "mirror",
        "url": "kaggle.com (stanford-cars copies, VMMRdb copies, misc galleries)",
        "status": "not attempted (no local kaggle credentials)",
        "images": "varies", "makes": "varies", "models": "varies",
        "annotations": ["varies"],
        "chinese_brands": "mostly no", "surveillance": False,
        "license": "murky re-uploads — per-dataset review required",
        "commercial": False,
        "download": "kaggle API (needs account)",
        "taxonomy_fit": "unknown until reviewed",
        "verdict": "SKIP until a specific set is reviewed + licensed",
    },
    {
        "name": "turbo.az collection (OURS)", "kind": "first-party",
        "url": "local pipeline scripts/collect_turboaz.py",
        "status": "WORKING (1000+ images collected, provenance per image)",
        "images": "~1100+", "makes": "~65", "models": "~150+",
        "annotations": ["make/model parsed", "provenance", "YOLO exterior QC"],
        "chinese_brands": True, "surveillance": "parking-like",
        "license": "listing images (c) owners/turbo.az — training use with provenance; "
                   "state deployment needs legal review (reported honestly)",
        "commercial": "pending legal review",
        "download": "polite collector in-repo",
        "taxonomy_fit": "native (built for our 77/380 taxonomy)",
        "verdict": "PRODUCTION SOURCE — the only market-correct dataset available",
    },
]


@app.command()
def main(as_json: bool = False):
    if as_json:
        print(json.dumps(SOURCES, ensure_ascii=False, indent=2))
        return
    for s in SOURCES:
        print(f"- {s['name']} [{s['status']}] img={s['images']} "
              f"makes={s['makes']} commercial={s['commercial']}\n  -> {s['verdict']}")


if __name__ == "__main__":
    app()
