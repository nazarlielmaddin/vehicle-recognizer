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
         out: str = "models/model/best.pt"):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_ld, va_ld, classes = build_loaders(data, backbone, batch, img_size)
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
    ep = 0
    if freeze_epochs > 0:
        # Phase 1: head only — fast, stable, adapts ImageNet features to cars
        freeze_base(True)
        opt = torch.optim.AdamW(params(True), lr=lr)
        for _ in range(min(freeze_epochs, epochs)):
            ep += 1
            m.train()
            for x, y in tr_ld:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(m(x), y)
                loss.backward(); opt.step()
            acc = _validate(m, va_ld, device)
            print(f"epoch {ep}/{epochs} [head] val_acc={acc:.4f}", flush=True)
            if acc > best:
                best = acc
                torch.save({"state_dict": m.state_dict(), "classes": classes,
                            "backbone": backbone, "acc": acc}, out)
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
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(m(x), y)
            loss.backward(); opt.step()
        sched.step()
        acc = _validate(m, va_ld, device)
        print(f"epoch {ep}/{epochs} val_acc={acc:.4f}", flush=True)
        if acc > best:
            best = acc
            torch.save({"state_dict": m.state_dict(), "classes": classes,
                        "backbone": backbone, "acc": acc}, out)
    print(f"best={best:.4f} -> {out}")


def _validate(m, va_ld, device) -> float:
    m.eval(); correct = tot = 0
    with torch.no_grad():
        for x, y in va_ld:
            p = m(x.to(device)).argmax(1).cpu()
            correct += (p == y).sum().item(); tot += len(y)
    return correct / max(tot, 1)

if __name__ == "__main__":
    app()
