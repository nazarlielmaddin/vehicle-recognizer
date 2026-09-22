"""Evaluation suite — per-condition slices, calibration, UNKNOWN metrics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import typer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

app = typer.Typer()

CONDITIONS = ["DAY", "NIGHT", "LOW_LIGHT", "FRONT", "REAR", "SIDE", "DIAGONAL",
              "OCCLUDED", "LOW_RESOLUTION", "BLURRED", "MULTI_VEHICLE", "SINGLE_VEHICLE"]

@app.command()
def main(pred_json: str = "reports/predictions.json", out: str = "reports/eval.json"):
    d = json.loads(Path(pred_json).read_text())
    rows = d if isinstance(d, list) else d.get("rows", [])
    rep: dict = {"n": len(rows), "slices": {}}
    if rows:
        y = [r["label"] for r in rows]; p = [r["pred"] for r in rows]
        rep["top1"] = accuracy_score(y, p)
        rep["macro_f1"] = f1_score(y, p, average="macro", zero_division=0)
        rep["confusion"] = confusion_matrix(y, p).tolist()
        # top-5 if logits stored
        rep["unknown_recall"] = sum(1 for r in rows if r["label"] == "UNKNOWN" and r["pred"] == "UNKNOWN") / max(1, sum(1 for r in rows if r["label"] == "UNKNOWN"))
        for c in CONDITIONS:
            sub = [r for r in rows if c in r.get("tags", [])]
            if sub:
                rep["slices"][c] = {
                    "n": len(sub),
                    "acc": accuracy_score([r["label"] for r in sub], [r["pred"] for r in sub]),
                }
    else:
        rep["note"] = "no predictions yet — run scripts/evaluate.py after training"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    app()
