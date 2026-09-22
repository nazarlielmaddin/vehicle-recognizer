"""Prepare identity-aware train/val/test crops + dataset_version.json."""
from __future__ import annotations
import json
from pathlib import Path
import typer
app = typer.Typer()

@app.command()
def main(raw: str = "data/interim", out: str = "data/processed",
         version: str = "0.1.0"):
    Path(f"{out}/crops/train").mkdir(parents=True, exist_ok=True)
    dv = {"version": version, "sources": [], "image_count": 0, "class_count": 0,
          "preprocessing": "letterbox-224", "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
          "augmentation": "configs/augmentation.yaml", "note": "populate via detector crops"}
    Path(f"{out}/dataset_version.json").write_text(json.dumps(dv, indent=2))
    print(f"scaffolded {out}/crops + dataset_version.json")

if __name__ == "__main__":
    app()
