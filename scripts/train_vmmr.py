"""Fine-tune the VMMR EfficientNet-B4 backbone on OUR taxonomy classes.

Starts from models/vmmr/vehicle_classifier.pth (car-adapted features from
280k images) and retrains the head (default: head-only, fast on CPU).
Saves models/vmmr/make_tuned.pt compatible with TimmClassifier
(classes + backbone adopted automatically).
EncyCarpedia note: site blocks all automated access (403 even with browser
UA, verified) — no collector possible; our turbo.az data is the source.
Usage: python scripts/train_vmmr.py --epochs 12
"""
from __future__ import annotations
from pathlib import Path
import typer
import torch
import timm

app = typer.Typer()
SRC = "models/vmmr/vehicle_classifier.pth"


@app.command()
def main(data: str = "data/processed/makes",
         epochs: int = 12, batch: int = 8, lr: float = 1e-3,
         img_size: int = 300, freeze: bool = True, seed: int = 42,
         out: str = "models/vmmr/make_tuned.pt"):
    import random
    import numpy as np
    from training.train_finegrained import _preflight, _validate, build_loaders
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    device = "cpu"
    mapping = _preflight(data)
    tr_ld, va_ld, classes = build_loaders(data, "efficientnet_b4", batch, img_size)
    assert classes == mapping, "loader/mapping mismatch"
    base = torch.load(SRC, map_location="cpu")
    m = timm.create_model("efficientnet_b4", pretrained=False, num_classes=8949)
    missing, unexpected = m.load_state_dict(base.get("model_state", base), strict=False)
    print(f"vmmr backbone loaded (missing={len(missing)} unexpected={len(unexpected)})",
          flush=True)
    m.reset_classifier(num_classes=len(classes))
    if freeze:
        for n, p in m.named_parameters():
            if "classifier" not in n:
                p.requires_grad = False
    m.to(device)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    best = 0.0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        m.train()
        tl, nb = 0.0, 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(m(x), y)
            tl += loss.item(); nb += 1
            loss.backward(); opt.step()
        acc, vl = _validate(m, va_ld, device, crit)
        print(f"Epoch {ep}/{epochs} | Train Loss: {tl/max(nb,1):.4f} | "
              f"Val Loss: {vl:.4f} | Val Acc: {acc*100:.2f}% | "
              f"Best Acc: {best*100:.2f}%", flush=True)
        if acc > best:
            best = acc
            torch.save({"state_dict": m.state_dict(), "classes": classes,
                        "backbone": "efficientnet_b4", "acc": acc, "epoch": ep,
                        "config": {"img_size": img_size, "batch": batch, "lr": lr,
                                   "seed": seed, "frozen": freeze, "src": SRC}}, out)
    print(f"best={best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    app()
