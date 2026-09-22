"""Stage C/E/F entry points — reuse train_finegrained with different label roots."""
from __future__ import annotations
import typer
app = typer.Typer()

@app.command()
def make(data: str = "data/processed/crops_by_make", out: str = "models/make/best.pt",
         backbone: str = "convnext_tiny.fb_in1k", epochs: int = 25):
    from training.train_finegrained import main as t
    t(data, backbone, epochs, 64, 3e-4, out)

@app.command()
def body(data: str = "data/processed/crops_by_body", out: str = "models/body_type/best.pt",
         backbone: str = "convnext_tiny.fb_in1k", epochs: int = 25):
    from training.train_finegrained import main as t
    t(data, backbone, epochs, 64, 3e-4, out)

@app.command()
def embedding(data: str = "data/processed/crops", out: str = "models/embedding/best.pt"):
    print(f"metric-learning (ArcFace/triplet) over {data} -> {out}; see training/train_embedding.py")

if __name__ == "__main__":
    app()
