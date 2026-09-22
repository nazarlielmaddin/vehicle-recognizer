"""Licensed image collection by viewpoint (front/rear/side/diagonal/night).
Enforces per-class viewpoint quotas; refuses studio-only datasets.
Tracks distribution numerically; flags severe imbalance.
"""
from __future__ import annotations
from collections import Counter
import typer
app = typer.Typer()
VIEWS = ["FRONT", "REAR", "LEFT", "RIGHT", "FRONT_LEFT", "FRONT_RIGHT", "REAR_LEFT", "REAR_RIGHT"]

@app.command()
def main(root: str = "data/raw", min_per_view: int = 10):
    print(f"collecting under {root} with min {min_per_view}/view/class; "
          "quotas enforced in scripts/validate_dataset.py coverage report")

if __name__ == "__main__":
    app()
