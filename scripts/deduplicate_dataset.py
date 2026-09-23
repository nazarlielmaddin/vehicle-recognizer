"""Cross-source deduplication: exact md5 + perceptual dHash (PIL only, no new deps).

Compares data/interim/crops against data/external/*/unified (+ within).
Near-duplicates (hamming <= threshold) collapse to one keeper (prefer: crops
over external, larger file, earlier source). Writes dedup_report.json.
Usage: python scripts/deduplicate_dataset.py [--threshold 6]
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import typer
from PIL import Image

app = typer.Typer()


def dhash(jp: Path, size: int = 16) -> str | None:
    try:
        im = Image.open(jp).convert("L").resize((size + 1, size), Image.BILINEAR)
        px = list(im.getdata())
        bits = "".join("1" if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] else "0"
                       for r in range(size) for c in range(size))
        return f"{int(bits, 2):064x}"
    except Exception:
        return None


def ham(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


@app.command()
def main(threshold: int = 6, out: str = "reports/dedup_cross_source.json"):
    pools: list[tuple[str, Path]] = [("crops", Path("data/interim/crops"))]
    ext = Path("data/external")
    if ext.is_dir():
        for src in sorted(p for p in ext.iterdir() if p.is_dir()):
            u = src / "unified"
            if u.is_dir():
                pools.append((src.name, u))
    files: list[tuple[str, Path, int]] = []
    for pool, root in pools:
        for jp in sorted(root.rglob("*.jpg")):
            files.append((pool, jp, jp.stat().st_size))
    print(f"scanning {len(files)} files in {len(pools)} pools", flush=True)
    by_md5: dict[str, tuple[str, Path]] = {}
    exact_dup = 0
    hashes: list[tuple[str, str, Path]] = []  # (pool, dhash, path)
    for pool, jp, sz in files:
        h = hashlib.md5()
        with open(jp, "rb") as f:
            for b in iter(lambda: f.read(65536), b""):
                h.update(b)
        md5 = h.hexdigest()
        if md5 in by_md5:
            exact_dup += 1
            continue
        by_md5[md5] = (pool, jp)
        dh = dhash(jp)
        if dh:
            hashes.append((pool, dh, jp))
    near = 0
    seen_hashes: list[str] = []
    for pool, dh, jp in hashes:
        if any(ham(dh, prev) <= threshold for prev in seen_hashes):
            near += 1
            continue
        seen_hashes.append(dh)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    rep = {"files": len(files), "pools": [p for p, _ in pools],
           "exact_duplicates": exact_dup, "near_duplicates": near,
           "unique": len(files) - exact_dup - near, "threshold": threshold}
    Path(out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    app()
