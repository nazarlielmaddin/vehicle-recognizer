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
    return (DataLoader(tr, batch, sampler=balanced_sampler(tr.targets), num_workers=4),
            DataLoader(va, batch * 2, num_workers=4), tr.classes)

@app.command()
def main(data: str = "data/processed/crops",
         backbone: str = "convnext_tiny.fb_in1k",
         epochs: int = 60, batch: int = 64, lr: float = 3e-4,
         out: str = "models/model/best.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_ld, va_ld, classes = build_loaders(data, backbone, batch, 224)
    m = timm.create_model(backbone, pretrained=True, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW([{"params": m.get_classifier().parameters(), "lr": lr},
                             {"params": [p for n, p in m.named_parameters()
                                          if "classifier" not in n and "head" not in n],
                              "lr": lr / 10}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    from training.common import FocalLoss
    crit = FocalLoss(gamma=2.0)
    best = 0.0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        m.train()
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(m(x), y)
            loss.backward(); opt.step()
        sched.step()
        # validate
        m.eval(); correct = tot = 0
        with torch.no_grad():
            for x, y in va_ld:
                p = m(x.to(device)).argmax(1).cpu()
                correct += (p == y).sum().item(); tot += len(y)
        acc = correct / max(tot, 1)
        print(f"epoch {ep+1}/{epochs} val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"state_dict": m.state_dict(), "classes": classes,
                        "backbone": backbone, "acc": acc}, out)
    print(f"best={best:.4f} -> {out}")

if __name__ == "__main__":
    app()
