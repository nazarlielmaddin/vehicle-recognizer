"""Build make-level train/val dataset from turbo.az collection.

- drops two-wheeler makes (Royal Enfield, Can-Am, Benda, CFMoto, Voge)
- verifies images (PIL), drops tiny (<160px side)
- anti-leakage: ONE image per listing id (thumb+full of same photo are near-dups)
- deterministic 85/15 split by listing-id hash
Usage: python scripts/prepare_make_dataset.py
"""
from __future__ import annotations
import hashlib, shutil
from pathlib import Path
import typer
from PIL import Image

app = typer.Typer()
MOTO_DIRS = {"Royal", "Can-Am", "Canam", "Benda", "Cfmoto", "Voge"}

@app.command()
def main(raw: str = "data/raw", out: str = "data/processed/makes",
         min_side: int = 160):
    roots = [Path(raw) / "turboaz", Path(raw) / "turboaz_pilot"]
    files: dict[str, list[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for jp in sorted(root.rglob("*.jpg")):
            try:
                mk = jp.parent.parent.name
            except IndexError:
                continue
            if mk in MOTO_DIRS:
                continue
            try:
                with Image.open(jp) as im:
                    im.verify()
                with Image.open(jp) as im:
                    if min(im.size) < min_side:
                        continue
            except Exception:
                continue
            lid = jp.stem.split("_")[0]
            files.setdefault(f"{mk}|{lid}", []).append(jp)
    tr = va = 0
    per_make: dict[str, int] = {}
    for key, cands in sorted(files.items()):
        mk, lid = key.split("|", 1)
        # prefer full-size over thumb
        cands.sort(key=lambda p: (0 if "_t" not in p.stem else 1, str(p)))
        src = cands[0]
        h = int(hashlib.md5(lid.encode()).hexdigest(), 16) % 100
        split = "val" if h < 15 else "train"
        d = Path(out) / split / mk
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d / f"{lid}.jpg")
        per_make[mk] = per_make.get(mk, 0) + 1
        tr, va = (tr + 1, va) if split == "train" else (tr, va + 1)
    print(f"train={tr} val={va} makes={len(per_make)}")
    for mk in sorted(per_make):
        print(f"  {mk}: {per_make[mk]}")

if __name__ == "__main__":
    app()
