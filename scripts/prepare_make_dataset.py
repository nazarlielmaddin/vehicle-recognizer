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
TEST_FRAC = 0.15


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
         min_side: int = 100, val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC,
         mix_full: float = 0.0):
    """mix_full: also include the source full frame when its YOLO box area >=
    threshold (car-dominant frames only). Same split as its crop (no leakage).
    For full-frame inference robustness (vmmr-only mode)."""
    crops = Path("data/interim/crops")
    by_make: dict[str, dict[str, tuple]] = {}
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
        if lid in by_make.setdefault(mk, {}):
            continue
        src, area = "", 0.0
        mp = jp.parent / f"{lid}.meta.json"
        if mp.exists():
            try:
                import json as _js
                mm = _js.loads(mp.read_text(encoding="utf-8"))
                src, area = mm.get("src", ""), float(mm.get("box_area", 0.0))
            except Exception:
                pass
        by_make[mk][lid] = (jp, src, area)
    out_p, rare_p = Path(out), Path(rare_out)
    for d in (out_p / "train", out_p / "val", out_p / "test", rare_p):
        if d.exists():
            shutil.rmtree(d)
    mapping: list[str] = []
    rare: dict[str, int] = {}
    n_tr = n_va = n_te = 0
    for mk in sorted(by_make):
        lids = sorted(by_make[mk])
        if len(lids) < 3:
            # QAYDA 5 + 3-way split integrity: <3 images cannot cover
            # train+val+test -> rare holdout for future collection rounds
            dd = rare_p / mk
            dd.mkdir(parents=True, exist_ok=True)
            for lid in lids:
                shutil.copy2(by_make[mk][lid][0], dd / f"{lid}.jpg")
            rare[mk] = len(lids)
            continue
        mapping.append(mk)
        n_val = max(1, round(len(lids) * val_frac))
        n_tst = max(1, round(len(lids) * test_frac)) if len(lids) >= 3 else 0
        val_ids = set(lids[:n_val])
        test_ids = set(lids[n_val:n_val + n_tst])
        tr_ids = [l for l in lids if l not in val_ids and l not in test_ids]

        def _put(lid: str, split: str) -> int:
            dd = out_p / split / mk
            dd.mkdir(parents=True, exist_ok=True)
            jp, src, area = by_make[mk][lid]
            shutil.copy2(jp, dd / f"{lid}.jpg")
            n = 1
            if mix_full > 0 and src and area >= mix_full:
                try:
                    with Image.open(src) as im:
                        im.verify()
                    shutil.copy2(src, dd / f"{lid}_full.jpg")
                    n = 2
                except Exception:
                    pass
            return n

        for lid in tr_ids:
            n_tr += _put(lid, "train")
        for lid in val_ids:
            n_va += _put(lid, "val")
        for lid in test_ids:
            n_te += _put(lid, "test")
    (Path(out).parent / "class_mapping.json").write_text(
        json.dumps({"classes": mapping, "val_frac": val_frac, "test_frac": test_frac,
                    "rule": "stratified-within-class, min-2-images"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train={n_tr} val={n_va} test={n_te} classes={len(mapping)}", flush=True)
    print(f"rare holdout makes ({len(rare)}): {sorted(rare)}", flush=True)
    print(f"invalid skipped: {bad}", flush=True)
    print(f"mapping -> data/processed/class_mapping.json", flush=True)


if __name__ == "__main__":
    app()
