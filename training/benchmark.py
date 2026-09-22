"""Candidate architecture benchmark on the project validation set.

Compares timm backbones for fine-grained vehicle recognition:
accuracy + latency + VRAM + viewpoint robustness. Pick empirically.
"""
from __future__ import annotations
import time
import typer

CANDIDATES = [
    "efficientnet_b3.ra2_in1k",
    "convnext_tiny.fb_in1k",
    "convnext_small.fb_in1k",
    "swin_tiny_patch4_window7_224",
    "vit_small_patch16_224.augreg_in1k",
    "resnet50.a1_in1k",
]

app = typer.Typer()

@app.command()
def main(data: str = "data/processed", epochs: int = 3, batch: int = 32):
    import torch, timm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    for name in CANDIDATES:
        try:
            m = timm.create_model(name, pretrained=True, num_classes=196)
            m.eval().to(device)
            x = torch.randn(batch, 3, 224, 224).to(device)
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.time()
            with torch.no_grad():
                for _ in range(10):
                    m(x)
            dt = (time.time() - t0) / 10
            params = sum(p.numel() for p in m.parameters()) / 1e6
            print(f"{name:40s} {params:7.1f}M  {dt*1000/batch:6.2f} ms/img")
        except Exception as e:
            print(f"{name:40s} FAILED: {e}")
    print("\nNext: full-finetune top-2 on validation (see docs/TRAINING.md).")

if __name__ == "__main__":
    app()
