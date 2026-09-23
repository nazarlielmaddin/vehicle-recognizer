"""Map external labels onto the project taxonomy (77/380 source of truth).

Reads data/external/<src>/labels.csv, normalizes make/model via
training/taxonomy.py (aliases, spelling variants) and writes
data/external/<src>/mapping.json + mapping_report.json.

NEVER invents a wrong mapping: uncertain pairs go to UNMAPPED and are
reported — they never enter processed training data.
Usage: python scripts/map_taxonomy.py --source stanford-hf
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import typer

sys.path.insert(0, ".")
from training.taxonomy import normalize_make, normalize_model  # noqa: E402

app = typer.Typer()


@app.command()
def main(source: str = typer.Option(...), root: str = "data/external"):
    src = Path(root) / source
    tax = json.loads(Path("data/metadata/taxonomy.json").read_text(encoding="utf-8"))
    makes: set[str] = set(tax["makes"])
    models: set[str] = set(tax["models"])
    detail: dict = tax.get("detail", {})
    labels_p = src / "labels.csv"
    if not labels_p.exists():
        print(f"ERROR: {labels_p} missing — run normalize_dataset.py first")
        raise typer.Exit(1)
    mapped: dict[str, dict] = {}
    unmapped: dict[str, int] = {}
    with open(labels_p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = f"{row['make']} {row['model']}".strip()
            mk = normalize_make(row["make"] or "Unknown")
            md = normalize_model(mk, row["model"] or "Unknown")
            pair = f"{mk} {md}".strip()
            if mk in makes and (pair in models or md.lower() == "unknown"):
                mapped[raw] = {"make": mk, "model": md if pair in models else "Unknown"}
            elif mk in makes and md != "Unknown":
                # known make, model not in taxonomy: keep make, flag model for review
                mapped[raw] = {"make": mk, "model": "UNMAPPED:" + md}
                unmapped[raw] = unmapped.get(raw, 0) + 1
            else:
                mapped[raw] = {"make": "UNMAPPED", "model": "UNMAPPED"}
                unmapped[raw] = unmapped.get(raw, 0) + 1
    (src / "mapping.json").write_text(json.dumps(mapped, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    n_map = sum(1 for v in mapped.values() if v["make"] != "UNMAPPED")
    (src / "mapping_report.json").write_text(json.dumps(
        {"mapped_labels": n_map, "unmapped_labels": len(mapped) - n_map,
         "unmapped": dict(sorted(unmapped.items(), key=lambda kv: -kv[1])[:50])},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mapped={n_map} unmapped={len(mapped) - n_map} (see mapping_report.json)")


if __name__ == "__main__":
    app()
