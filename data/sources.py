"""Dataset registry + provenance. Every sample carries source/license metadata."""
from __future__ import annotations
import json
from pathlib import Path

# Candidate sources — inspect before use; respect license/robots/ToS.
SOURCES = [
    {"name": "StanfordCars/Cars196", "url": "https://ai.stanford.edu/~jkrause/cars/car_dataset.html",
     "images": 16185, "classes": 196, "bbox": False, "make_model": True, "body": False,
     "license": "research-only", "commercial": False, "role": "benchmark/seed"},
    {"name": "CompCars", "url": "http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/",
     "images": 214345, "classes": 1716, "bbox": True, "make_model": True, "body": False,
     "license": "research-only", "commercial": False, "role": "viewpoint-rich"},
    {"name": "VeRi-776", "url": "https://github.com/JingdongDeployment/VeRi",
     "images": 50000, "classes": 776, "bbox": True, "make_model": True, "body": True,
     "license": "research-only", "commercial": False, "role": "surveillance ReID/multi-view"},
    {"name": "BoxCars116k", "url": "https://github.com/JakubSochor/BoxCars",
     "images": 116826, "classes": 693, "bbox": True, "make_model": True, "body": True,
     "license": "research (CC-BY-NC-like, verify)", "commercial": False, "role": "surveillance fine-grained"},
    {"name": "Stanford Visual Census", "url": "https://github.com/serre-lab/visual_census",
     "images": 712430, "classes": 2657, "bbox": True, "make_model": True, "body": False,
     "license": "verify per-release", "commercial": False, "role": "large-scale supplement"},
    {"name": "Manufacturer press kits (licensed)", "url": "OEM media sites",
     "images": "varies", "classes": "varies", "bbox": False, "make_model": True, "body": True,
     "license": "per-OEM media license", "commercial": "only with permission", "role": "clean identity"},
]

def provenance(source: str, url: str, dataset: str, license: str, **kw) -> dict:
    from datetime import date
    rec = {"source": source, "source_url": url, "dataset": dataset, "license": license,
           "acquired": str(date.today()), "preprocessing": []}
    rec.update(kw)
    return rec

if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    Path("reports/dataset_sources.json").write_text(json.dumps(SOURCES, indent=2))
    print("wrote reports/dataset_sources.json — review licenses before downloading")
