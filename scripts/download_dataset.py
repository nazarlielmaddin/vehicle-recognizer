"""Download external datasets into data/external/<source>/ — LICENSE-GATED.

Nothing downloads without an explicit flag combo:
  --source NAME --confirm-license="research-only"   (academic sets)
Academic sets NEVER enter production training without legal review;
the script prints the verdict from discover_datasets.py and aborts otherwise.

Handlers:
  stanford-hf  HF mirror tanganke/stanford_cars (needs `datasets` package)
  compcars     BLOCKED (needs author agreement) — prints instructions
  boxcars      BLOCKED (server dead) — prints status
  vmmrdb       BLOCKED (wrong market + murky license) — prints status
Usage: python scripts/download_dataset.py --source stanford-hf --confirm-license research-only
"""
from __future__ import annotations
from pathlib import Path
import typer

app = typer.Typer()

GATES = {
    "stanford-hf": ("research-only", "tanganke/stanford_cars (HF mirror, ~6GB)"),
    "compcars": ("agreement", None),
    "boxcars": ("dead", None),
    "vmmrdb": ("skip", None),
}


@app.command()
def main(source: str = typer.Option(..., help="stanford-hf|compcars|boxcars|vmmrdb"),
         out: str = "data/external",
         confirm_license: str = typer.Option("", help="required: research-only")):
    if source == "compcars":
        print("BLOCKED: CompCars needs a signed author agreement "
              "(http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) "
              "and is non-commercial research-only. Complete it manually, then re-run.")
        raise typer.Exit(2)
    if source == "boxcars":
        print("BLOCKED: BoxCars server DNS-dead (verified 2026-09-23). Nothing to download.")
        raise typer.Exit(2)
    if source == "vmmrdb":
        print("SKIP: wrong market (0 Chinese-brand classes) + murky license. Not integrated.")
        raise typer.Exit(2)
    if source != "stanford-hf":
        print(f"Unknown source: {source}")
        raise typer.Exit(2)
    if confirm_license != "research-only":
        print('REFUSED: pass --confirm-license="research-only" (Stanford mirror: research use, '
              "no clear commercial license; benchmark/diversity only).")
        raise typer.Exit(2)
    try:
        from datasets import load_dataset
    except ImportError:
        print("Need `pip install datasets pyarrow` first (heavy). Aborting without changes.")
        raise typer.Exit(2)
    dest = Path(out) / "stanford-hf"
    dest.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("tanganke/stanford_cars", split="train")
    n = 0
    for i, row in enumerate(ds):
        img = row.get("image")
        label = row.get("label", row.get("class", "unknown"))
        if img is None:
            continue
        d = dest / "raw" / str(label)
        d.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(d / f"{i}.jpg")
        n += 1
    (dest / "PROVENANCE.txt").write_text(
        "source: huggingface.co/datasets/tanganke/stanford_cars (mirror of Stanford Cars)\n"
        "license: research use; no clear commercial license — NOT for production training\n"
        "without legal review. Benchmark/diversity only.\n")
    print(f"downloaded {n} images -> {dest}/raw")


if __name__ == "__main__":
    app()
