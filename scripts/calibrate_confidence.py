"""Temperature scaling on validation logits → configs/default.yaml."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import typer

app = typer.Typer()

@app.command()
def main(logits_npz: str = "reports/val_logits.npz", config: str = "configs/default.yaml"):
    import yaml
    d = np.load(logits_npz)
    logits, labels = d["logits"], d["labels"]
    best_T, best_nll = 1.0, 1e9
    for T in np.linspace(0.3, 5.0, 50):
        z = logits / T
        z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        nll = -np.log(p[np.arange(len(labels)), labels] + 1e-9).mean()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    cfg = yaml.safe_load(open(config, encoding="utf-8"))
    cfg["calibration"]["temperature"] = round(best_T, 3)
    yaml.safe_dump(cfg, open(config, "w", encoding="utf-8"), sort_keys=False)
    print(f"T={best_T:.3f} nll={best_nll:.4f} written to {config}")

if __name__ == "__main__":
    app()
