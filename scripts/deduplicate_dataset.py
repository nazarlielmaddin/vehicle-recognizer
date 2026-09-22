"""Dedup + QC: pHash near-duplicates, corrupt/low-res/irrelevant filtering."""
from __future__ import annotations
import typer
from pathlib import Path
from PIL import Image
import imagehash

app = typer.Typer()

@app.command()
def main(root: str = "data/raw", out: str = "data/interim/dedup_report.json",
         hash_size: int = 16, threshold: int = 6, min_side: int = 64):
    import json
    seen: dict[str, str] = {}
    dups, corrupt, tiny, kept = [], [], [], []
    files = [p for p in Path(root).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    for p in files:
        try:
            im = Image.open(p); im.verify()
            im = Image.open(p)
            if min(im.size) < min_side:
                tiny.append(str(p)); continue
        except Exception:
            corrupt.append(str(p)); continue
        try:
            h = str(imagehash.phash(Image.open(p), hash_size=hash_size))
        except Exception:
            corrupt.append(str(p)); continue
        # near-duplicate vs seen hashes (hamming)
        is_dup = False
        for k, q in seen.items():
            if sum(c1 != c2 for c1, c2 in zip(h, q)) <= threshold:
                dups.append({"file": str(p), "near": q}); is_dup = True; break
        if not is_dup:
            seen[h] = str(p); kept.append(str(p))
    rep = {"scanned": len(files), "kept": len(kept), "duplicates": len(dups),
           "corrupt": len(corrupt), "tiny": len(tiny),
           "dups": dups[:200], "corrupt_sample": corrupt[:50], "tiny_n": len(tiny)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=2))
    print(f"scanned={len(files)} kept={len(kept)} dups={len(dups)} corrupt={len(corrupt)} tiny={len(tiny)}")

if __name__ == "__main__":
    app()
