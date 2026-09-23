"""Evaluate a trained classifier checkpoint on the TEST split.

Metrics: top-1, top-3, top-5, macro-F1, confusion matrix (top confused pairs).
Usage: python scripts/evaluate_classifier.py --weights models/make/best.pt --data data/processed/makes --out reports/eval_make.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import typer
import torch
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, confusion_matrix, top_k_accuracy_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
app = typer.Typer()


@app.command()
def main(weights: str, data: str = "data/processed/makes",
         out: str = "reports/eval_make.json", batch: int = 32):
    ckpt = torch.load(weights, map_location="cpu")
    classes: list[str] = ckpt["classes"]
    img = ckpt.get("config", {}).get("img_size", 192)
    m = timm.create_model(ckpt.get("backbone", "efficientnet_b0"),
                          pretrained=False, num_classes=len(classes))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    tf = transforms.Compose([transforms.Resize(int(img * 1.14)), transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    te = datasets.ImageFolder(f"{data}/test", tf)
    assert te.classes == classes, f"test classes != checkpoint classes"
    ld = DataLoader(te, batch, num_workers=0)
    import numpy as np
    probas, ys = [], []
    with torch.no_grad():
        for x, y in ld:
            probas.append(torch.softmax(m(x), 1).cpu().numpy())
            ys.extend(y.tolist())
    P = np.concatenate(probas)
    Y = np.array(ys)
    all_p = P.argmax(1).tolist()
    all_y = ys
    top1 = float((P.argmax(1) == Y).mean())
    top3 = float(top_k_accuracy_score(Y, P, k=3, labels=list(range(len(classes)))))
    top5 = float(top_k_accuracy_score(Y, P, k=min(5, len(classes)),
                                      labels=list(range(len(classes)))))
    macro_f1 = float(f1_score(all_y, all_p, average="macro", zero_division=0))
    cm = confusion_matrix(all_y, all_p, labels=list(range(len(classes)))).tolist()
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > 0:
                pairs.append({"true": classes[i], "pred": classes[j], "count": cm[i][j]})
    pairs.sort(key=lambda d: -d["count"])
    rep = {"n": len(all_y), "classes": len(classes), "top1": round(top1, 4),
           "top3": round(top3, 4), "top5": round(top5, 4), "macro_f1": round(macro_f1, 4),
           "confusion_top": pairs[:25], "weights": weights}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n", "top1", "top3", "macro_f1")}, indent=2))


if __name__ == "__main__":
    app()
