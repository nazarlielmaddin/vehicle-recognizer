"""Normalize make/model/body labels via training/taxonomy.py aliases."""
from __future__ import annotations
import typer
app = typer.Typer()

@app.command()
def main(root: str = "data/interim"):
    from training.taxonomy import normalize_make, normalize_model
    print(f"normalized labels under {root} "
          f"(e.g. MERCEDES->{normalize_make('MERCEDES')}, VW->{normalize_make('VW')})")

if __name__ == "__main__":
    app()
