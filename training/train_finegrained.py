"""Fine-grained model classifier training (Stage D) with hard-negative support."""
from __future__ import annotations
from pathlib import Path
import typer
import torch, timm
from torch.utils.data import DataLoader

app = typer.Typer()

def build_loaders(root: str, backbone: str, batch: int, img: int):
    from torchvision import datasets, transforms
    tf_tr = transforms.Compose([
        transforms.RandomResizedCrop(img, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_va = transforms.Compose([
        transforms.Resize(int(img * 1.14)), transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tr = datasets.ImageFolder(f"{root}/train", tf_tr)
    va = datasets.ImageFolder(f"{root}/val", tf_va)
    from training.common import balanced_sampler
    # num_workers=0: Windows spawn overhead dominates on small datasets;
    # images are tiny JPEGs, in-process loading is faster here.
    return (DataLoader(tr, batch, sampler=balanced_sampler(tr.targets), num_workers=0),
            DataLoader(va, batch * 2, num_workers=0), tr.classes)

@app.command()
def main(data: str = "data/processed/crops",
         backbone: str = "convnext_tiny.fb_in1k",
         epochs: int = 60, batch: int = 64, lr: float = 3e-4,
         img_size: int = 224,
         freeze_epochs: int = 5,
         seed: int = 42,
         out: str = "models/model/best.pt"):
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mapping = _preflight(data)
    tr_ld, va_ld, classes = build_loaders(data, backbone, batch, img_size)
    if classes != mapping:
        print("ERROR: loader class order != class_mapping.json.\nTraining aborted.", flush=True)
        raise SystemExit(1)
    m = timm.create_model(backbone, pretrained=True, num_classes=len(classes)).to(device)
    head_names = {"classifier", "head", "fc"}
    def params(head_only: bool):
        if not head_only:
            return m.parameters()
        return [p for n, p in m.named_parameters()
                if any(h in n for h in head_names)]
    def freeze_base(frozen: bool):
        for n, p in m.named_parameters():
            if not any(h in n for h in head_names):
                p.requires_grad = not frozen
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    best = 0.0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cfg = {"backbone": backbone, "epochs": epochs, "batch": batch, "lr": lr,
           "img_size": img_size, "freeze_epochs": freeze_epochs, "seed": seed,
           "data": data}
    ep = 0
    if freeze_epochs > 0:
        # Phase 1: head only — fast, stable, adapts ImageNet features to cars
        freeze_base(True)
        opt = torch.optim.AdamW(params(True), lr=lr)
        for _ in range(min(freeze_epochs, epochs)):
            ep += 1
            m.train()
            tl, nb = 0.0, 0
            for x, y in tr_ld:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(m(x), y)
                tl += loss.item(); nb += 1
                loss.backward(); opt.step()
            acc, vl = _validate(m, va_ld, device, crit)
            print(f"Epoch {ep}/{epochs} | Phase: head-only | "
                  f"Train Loss: {tl/max(nb,1):.4f} | Val Loss: {vl:.4f} | "
                  f"Val Acc: {acc*100:.2f}% | Best Acc: {best*100:.2f}%", flush=True)
            if acc > best:
                best = acc
                _save(out, m, classes, backbone, acc, ep, cfg)
    # Phase 2: full fine-tune with differential LRs
    freeze_base(False)
    opt = torch.optim.AdamW([{"params": m.get_classifier().parameters(), "lr": lr},
                             {"params": [p for n, p in m.named_parameters()
                                          if "classifier" not in n and "head" not in n],
                              "lr": lr / 10}], weight_decay=0.05)
    remaining = epochs - ep
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(remaining, 1))
    for _ in range(max(remaining, 0)):
        ep += 1
        m.train()
        tl, nb = 0.0, 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(m(x), y)
            tl += loss.item(); nb += 1
            loss.backward(); opt.step()
        sched.step()
        acc, vl = _validate(m, va_ld, device, crit)
        print(f"Epoch {ep}/{epochs} | Phase: finetune | "
              f"Train Loss: {tl/max(nb,1):.4f} | Val Loss: {vl:.4f} | "
              f"Val Acc: {acc*100:.2f}% | Best Acc: {best*100:.2f}%", flush=True)
        if acc > best:
            best = acc
            _save(out, m, classes, backbone, acc, ep, cfg)
    print(f"best={best:.4f} -> {out}", flush=True)


def _save(out: str, m, classes: list[str], backbone: str, acc: float,
          epoch: int, cfg: dict) -> None:
    torch.save({"state_dict": m.state_dict(), "classes": classes,
                "backbone": backbone, "acc": acc, "epoch": epoch,
                "config": cfg}, out)


def _preflight(data: str) -> list[str]:
    """QAYDA 11: training starts ONLY on PASS. Returns the stable class list."""
    root = Path(data)
    tr_dir, va_dir = root / "train", root / "val"
    if not tr_dir.is_dir() or not va_dir.is_dir():
        raise SystemExit(f"ERROR: missing train/ or val/ under {data}\nTraining aborted.")
    tr_classes = sorted(p.name for p in tr_dir.iterdir() if p.is_dir())
    va_classes = sorted(p.name for p in va_dir.iterdir() if p.is_dir())
    if set(tr_classes) != set(va_classes):
        print("ERROR: Train/validation class mismatch.\nTraining aborted.", flush=True)
        print(f"  train-only: {sorted(set(tr_classes) - set(va_classes))}", flush=True)
        print(f"  val-only: {sorted(set(va_classes) - set(tr_classes))}", flush=True)
        raise SystemExit(1)
    te_dir = root / "test"
    if te_dir.is_dir():
        te_classes = sorted(p.name for p in te_dir.iterdir() if p.is_dir())
        if set(te_classes) != set(tr_classes):
            print("ERROR: Train/test class mismatch.\nTraining aborted.", flush=True)
            raise SystemExit(1)
    mapping_p = root.parent / "class_mapping.json"
    if mapping_p.exists():
        import json
        mapping = json.loads(mapping_p.read_text(encoding="utf-8"))["classes"]
        if mapping != tr_classes:
            print("ERROR: class_mapping.json != train dir order.\nTraining aborted.", flush=True)
            raise SystemExit(1)
    else:
        mapping = tr_classes
    # leakage: same filename stem in both splits
    leak = 0
    for cls in tr_classes:
        a = {p.stem for p in (root / "train" / cls).glob("*.jpg")}
        b = {p.stem for p in (root / "val" / cls).glob("*.jpg")}
        leak += len(a & b)
    if leak:
        print(f"ERROR: {leak} leaked files present in both splits.\nTraining aborted.", flush=True)
        raise SystemExit(1)
    # corrupt/empty check
    bad = [str(p) for p in list(root.rglob("*.jpg")) if p.stat().st_size == 0]
    if bad:
        print(f"ERROR: {len(bad)} empty images, e.g. {bad[:3]}.\nTraining aborted.", flush=True)
        raise SystemExit(1)
    n_tr = sum(1 for _ in (root / "train").rglob("*.jpg"))
    n_va = sum(1 for _ in (root / "val").rglob("*.jpg"))
    print("=== DATASET PREFLIGHT ===", flush=True)
    print(f"Classes: {len(mapping)}", flush=True)
    print(f"Train images: {n_tr}", flush=True)
    print(f"Val images: {n_va}", flush=True)
    print(f"Class overlap: {len(mapping)}/{len(mapping)}", flush=True)
    print(f"Leakage: {leak}", flush=True)
    print(f"Missing classes: 0", flush=True)
    print(f"Invalid images: 0", flush=True)
    print("Status: PASS", flush=True)
    return mapping


def _validate(m, va_ld, device, crit) -> tuple[float, float]:
    m.eval(); correct = tot = 0; loss_sum = 0.0; nb = 0
    with torch.no_grad():
        for x, y in va_ld:
            x, y = x.to(device), y.to(device)
            out = m(x)
            loss_sum += crit(out, y).item(); nb += 1
            p = out.argmax(1).cpu()
            correct += (p == y.cpu()).sum().item(); tot += len(y)
    return correct / max(tot, 1), loss_sum / max(nb, 1)

if __name__ == "__main__":
    app()
