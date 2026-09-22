"""Hard-negative mining: confusion pairs → targeted fine-tune lists."""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
import typer

app = typer.Typer()

@app.command()
def main(pred_json: str = "reports/predictions.json",
         out: str = "reports/hard_negatives.json", top: int = 25):
    rows = json.loads(Path(pred_json).read_text())
    rows = rows if isinstance(rows, list) else rows.get("rows", [])
    pairs = Counter((r["label"], r["pred"]) for r in rows if r["label"] != r["pred"])
    rep = [{"true": a, "pred": b, "count": c} for (a, b), c in pairs.most_common(top)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=2))
    print("\n".join(f"{r['true']} -> {r['pred']}  x{r['count']}" for r in rep) or "no errors found")

if __name__ == "__main__":
    app()
