"""Normalize an external source into data/external/<src>/unified/.

Input layouts accepted:
  A) <src>/raw/<anything>/**/*.jpg  (+ optional labels.csv with file,label)
  B) ImageFolder-ish <src>/raw/<CLASS>/*.jpg
Output: unified/<MAKE>/<MODEL>/*.jpg + labels.csv columns:
  file,make,model,viewpoint,lighting,source,source_url,license,split_hint
Idempotent: re-runs skip existing outputs (hash check).
Usage: python scripts/normalize_dataset.py --source stanford-hf
"""
from __future__ import annotations
import csv
import hashlib
import shutil
from pathlib import Path
import typer
from PIL import Image

app = typer.Typer()


def _verify(jp: Path, min_side: int = 64) -> bool:
    try:
        with Image.open(jp) as im:
            im.verify()
        with Image.open(jp) as im:
            return min(im.size) >= min_side
    except Exception:
        return False


def _md5(jp: Path) -> str:
    h = hashlib.md5()
    with open(jp, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


@app.command()
def main(source: str = typer.Option(...), root: str = "data/external"):
    src = Path(root) / source
    raw = src / "raw"
    if not raw.is_dir():
        print(f"ERROR: {raw} missing — run download_dataset.py first")
        raise typer.Exit(1)
    uni = src / "unified"
    uni.mkdir(parents=True, exist_ok=True)
    labels_p = src / "labels.csv"
    seen: dict[str, dict] = {}
    if labels_p.exists():
        with open(labels_p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen[row["md5"]] = row
    n_new = n_skip = n_bad = 0
    for jp in sorted(raw.rglob("*.jpg")):
        md5 = _md5(jp)
        if md5 in seen:
            n_skip += 1
            continue
        if not _verify(jp):
            n_bad += 1
            continue
        rel = jp.relative_to(raw).parts
        if len(rel) >= 3:
            mk, md = rel[0], rel[1]
        elif len(rel) == 2:
            mk, md = "Unknown", rel[0]
        else:
            mk, md = "Unknown", "Unknown"
        dest = uni / mk / md
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(jp, dest / f"{md5[:12]}.jpg")
        seen[md5] = {"md5": md5, "file": f"{mk}/{md}/{md5[:12]}.jpg",
                     "make": mk, "model": md, "viewpoint": "UNKNOWN",
                     "lighting": "UNKNOWN", "source": source,
                     "source_url": "", "license": "see source PROVENANCE",
                     "split_hint": ""}
        n_new += 1
    with open(labels_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["md5", "file", "make", "model", "viewpoint",
                                          "lighting", "source", "source_url",
                                          "license", "split_hint"])
        w.writeheader()
        w.writerows(seen.values())
    print(f"new={n_new} skipped={n_skip} bad={n_bad} total={len(seen)} -> {uni}")


if __name__ == "__main__":
    app()
