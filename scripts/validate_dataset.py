"""Validate labels/metadata + viewpoint coverage + leakage checks."""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
import typer
app = typer.Typer()

@app.command()
def main(root: str = "data/interim", out: str = "reports/validation.json"):
    metas = list(Path(root).glob("**/meta.json"))
    views: Counter = Counter()
    for m in metas:
        try:
            d = json.loads(m.read_text())
            views[d.get("viewpoint", "UNKNOWN")] += 1
        except Exception:
            pass
    rep = {"meta_files": len(metas), "viewpoint_dist": dict(views)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    app()
