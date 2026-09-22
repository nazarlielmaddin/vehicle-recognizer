"""Build taxonomy.json from seed + extended AZ-market seed + collected labels."""
from __future__ import annotations
import typer
from training.taxonomy import SEED_ENTRIES, build_taxonomy, save_taxonomy
from training.taxonomy_extra import EXTRA_ENTRIES
app = typer.Typer()

@app.command()
def main(out: str = "data/metadata/taxonomy.json"):
    entries = list(SEED_ENTRIES) + [
        {"make": mk, "model": md, "body_type": bt} for mk, md, bt in EXTRA_ENTRIES
    ]
    tax = build_taxonomy(entries)
    save_taxonomy(tax, out)
    print(f"makes={len(tax['makes'])} models={len(tax['models'])} -> {out}")

if __name__ == "__main__":
    app()
