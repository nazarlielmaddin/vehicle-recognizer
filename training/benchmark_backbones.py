"""Backbone benchmark: params + CPU latency + size (factual, no training).
Usage: python training/benchmark_backbones.py
"""
from __future__ import annotations
import io
import sys
import time
import torch
import timm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
torch.set_num_threads(4)

CANDS = ["efficientnet_b0", "efficientnet_b2", "convnext_tiny.fb_in1k"]


def main() -> None:
    print(f"{'backbone':28s} {'params':>9s} {'ms/img CPU':>11s} {'img':>5s}", flush=True)
    for name in CANDS:
        try:
            m = timm.create_model(name, pretrained=False, num_classes=47)
            m.eval()
            n = sum(p.numel() for p in m.parameters()) / 1e6
            x = torch.randn(8, 3, 192, 192)
            with torch.no_grad():
                for _ in range(3):
                    m(x)
                t0 = time.time()
                for _ in range(5):
                    m(x)
                ms = (time.time() - t0) / 40 * 1000
            print(f"{name:28s} {n:8.1f}M {ms:10.1f} {'192':>5s}", flush=True)
        except Exception as e:
            print(f"{name:28s} FAILED {str(e)[:100]}", flush=True)


if __name__ == "__main__":
    main()
