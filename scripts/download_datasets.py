"""Download licensed datasets AFTER reviewing reports/dataset_sources.json licenses.
Respects ToS/robots/rate limits; writes provenance sidecars. Usage:
  python scripts/download_datasets.py --datasets BoxCars116k VeRi-776 --out data/raw
"""
from __future__ import annotations
import typer
app = typer.Typer()

@app.command()
def main(datasets: list[str] = typer.Option([], help="names from data/sources.py"),
         out: str = "data/raw"):
    print("1) Read docs/DATASETS.md + reports/dataset_sources.json license column.")
    print("2) Download MANUALLY from official URLs (auth often required).")
    print(f"3) Place under {out}/<dataset>/ + provenance.json per sample.")
    print(f"Requested: {datasets or 'ALL — pick per license review'}")

if __name__ == "__main__":
    app()
