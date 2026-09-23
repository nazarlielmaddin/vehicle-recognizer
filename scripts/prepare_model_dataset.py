"""Build model-level dataset: data/processed/models/{train,val,test}/<make>/<model>/.

Rules:
- only VALID make/model pairs (must exist in taxonomy detail) — never invent pairs;
  invalid pairs -> data/processed/unmapped_models/ + report;
- n>=4: stratified train/val/test (~70/15/15, min 1 val + 1 test);
- n==3: 1/1/1; n==2: train+val; n==1: rare holdout (no fake splits);
- deterministic (sorted listing ids, no RNG).
Usage: python scripts/prepare_model_dataset.py
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
import typer
from PIL import Image

sys.path.insert(0, ".")

app = typer.Typer()
MOTO_DIRS = {"Royal", "Can-Am", "Canam", "Benda", "Cfmoto", "Voge", "Yadea"}


def _verify(jp: Path, min_side: int = 100) -> bool:
    try:
        with Image.open(jp) as im:
            im.verify()
        with Image.open(jp) as im:
            return min(im.size) >= min_side
    except Exception:
        return False


@app.command()
def main(out: str = "data/processed/models", val_frac: float = 0.15,
         test_frac: float = 0.15):
    tax = json.loads(Path("data/metadata/taxonomy.json").read_text(encoding="utf-8"))
    detail: dict = tax.get("detail", {})
    crops = Path("data/interim/crops")
    pairs: dict[tuple[str, str], dict[str, Path]] = {}
    bad = inv = 0
    for jp in sorted(crops.rglob("*.jpg")):
        rel = jp.relative_to(crops).parts
        if len(rel) != 3:
            continue
        mk, md = rel[0], rel[1]
        if mk in MOTO_DIRS or not _verify(jp):
            bad += 1
            continue
        # valid pair? map raw dir name onto an EXISTING taxonomy model.
        # Uncertain input -> UNMAPPED holdout (never invent pairs).
        from training.taxonomy import normalize_make, match_taxonomy_model
        mk_norm = normalize_make(mk)
        hit = match_taxonomy_model(mk_norm, md, detail)
        if hit is None:
            inv += 1
            dd = Path("data/processed/unmapped_models") / mk / md
            dd.mkdir(parents=True, exist_ok=True)
            shutil.copy2(jp, dd / jp.name)
            continue
        mk, md = mk_norm, hit
        pairs.setdefault((mk, md), {})[jp.stem] = jp
    out_p = Path(out)
    for d in (out_p / "train", out_p / "val", out_p / "test",
              Path("data/processed/rare_models"), Path("data/processed/unmapped_models")):
        if d.exists():
            shutil.rmtree(d)
    mapping: list[str] = []
    stats: dict[str, dict] = {}
    n_tr = n_va = n_te = n_rare = 0
    for (mk, md) in sorted(pairs):
        lids = sorted(pairs[(mk, md)])
        label = f"{mk} {md}"
        if len(lids) < 2:
            dd = Path("data/processed/rare_models") / mk / md
            dd.mkdir(parents=True, exist_ok=True)
            for lid in lids:
                shutil.copy2(pairs[(mk, md)][lid], dd / f"{lid}.jpg")
            n_rare += len(lids)
            continue
        mapping.append(label)
        nv = max(1, round(len(lids) * val_frac))
        nt = max(1, round(len(lids) * test_frac)) if len(lids) >= 3 else 0
        val_ids, test_ids = set(lids[:nv]), set(lids[nv:nv + nt])
        tr_ids = [l for l in lids if l not in val_ids and l not in test_ids]
        c_tr = c_va = c_te = 0
        for lid in tr_ids:
            dd = out_p / "train" / mk / md
            dd.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pairs[(mk, md)][lid], dd / f"{lid}.jpg")
            c_tr += 1
        for lid in val_ids:
            dd = out_p / "val" / mk / md
            dd.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pairs[(mk, md)][lid], dd / f"{lid}.jpg")
            c_va += 1
        for lid in test_ids:
            dd = out_p / "test" / mk / md
            dd.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pairs[(mk, md)][lid], dd / f"{lid}.jpg")
            c_te += 1
        stats[label] = {"total": len(lids), "train": c_tr, "val": c_va, "test": c_te,
                        "source": "turboaz"}
        n_tr, n_va, n_te = n_tr + c_tr, n_va + c_va, n_te + c_te
    (Path(out).parent / "model_mapping.json").write_text(
        json.dumps({"classes": mapping, "rule": "valid-pairs-only, min-2-images"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    counts = [s["total"] for s in stats.values()]
    import statistics
    print(f"pairs={len(mapping)} train={n_tr} val={n_va} test={n_te} "
          f"rare={n_rare} invalid_pairs={inv} bad={bad}", flush=True)
    if counts:
        print(f"min={min(counts)} median={statistics.median(counts)} max={max(counts)}", flush=True)
    (Path(out).parent / "model_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
