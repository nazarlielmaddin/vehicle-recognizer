"""Build make-level train/val from YOLO-verified crops — CLASS-AWARE split.

Rules enforced (see docs/PRO_PLAN.md + audit):
- split happens WITHIN each class (never class-vs-class);
- 1-image classes -> data/processed/rare/ holdout (never fake validation);
- one file per listing id (near-dup guard);
- writes data/processed/class_mapping.json — the SINGLE stable label mapping;
- deterministic (sorted order, no RNG).
Usage: python scripts/prepare_make_dataset.py
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import typer
from PIL import Image

app = typer.Typer()
MOTO_DIRS = {"Royal", "Can-Am", "Canam", "Benda", "Cfmoto", "Voge", "Yadea"}
VAL_FRAC = 0.15


def _verify(jp: Path, min_side: int) -> bool:
    try:
        with Image.open(jp) as im:
            im.verify()
        with Image.open(jp) as im:
            return min(im.size) >= min_side
    except Exception:
        return False


@app.command()
def main(out: str = "data/processed/makes", rare_out: str = "data/processed/rare",
         min_side: int = 100, val_frac: float = VAL_FRAC):
    crops = Path("data/interim/crops")
    by_make: dict[str, dict[str, Path]] = {}
    bad = 0
    for jp in sorted(crops.rglob("*.jpg")):
        rel = jp.relative_to(crops).parts
        if len(rel) != 3:
            continue
        mk = rel[0]
        if mk in MOTO_DIRS:
            continue
        if not _verify(jp, min_side):
            bad += 1
            continue
        lid = jp.stem
        # one file per listing id (filter already dedups; keep first deterministically)
        by_make.setdefault(mk, {}).setdefault(lid, jp)
    out_p, rare_p = Path(out), Path(rare_out)
    for d in (out_p / "train", out_p / "val", rare_p):
        if d.exists():
            shutil.rmtree(d)
    mapping: list[str] = []
    rare: dict[str, int] = {}
    n_tr = n_va = 0
    for mk in sorted(by_make):
        lids = sorted(by_make[mk])
        if len(lids) < 2:
            # QAYDA 5: no fake validation — holdout for future collection rounds
            dd = rare_p / mk
            dd.mkdir(parents=True, exist_ok=True)
            for lid in lids:
                shutil.copy2(by_make[mk][lid], dd / f"{lid}.jpg")
            rare[mk] = len(lids)
            continue
        mapping.append(mk)
        n_val = max(1, round(len(lids) * val_frac))
        val_ids, tr_ids = set(lids[:n_val]), set(lids[n_val:])
        for lid in tr_ids:
            dd = out_p / "train" / mk
            dd.mkdir(parents=True, exist_ok=True)
            shutil.copy2(by_make[mk][lid], dd / f"{lid}.jpg")
            n_tr += 1
        for lid in val_ids:
            dd = out_p / "val" / mk
            dd.mkdir(parents=True, exist_ok=True)
            shutil.copy2(by_make[mk][lid], dd / f"{lid}.jpg")
            n_va += 1
    (Path(out).parent / "class_mapping.json").write_text(
        json.dumps({"classes": mapping, "val_frac": val_frac,
                    "rule": "stratified-within-class, min-2-images"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train={n_tr} val={n_va} classes={len(mapping)}", flush=True)
    print(f"rare holdout makes ({len(rare)}): {sorted(rare)}", flush=True)
    print(f"invalid skipped: {bad}", flush=True)
    print(f"mapping -> data/processed/class_mapping.json", flush=True)


if __name__ == "__main__":
    app()
