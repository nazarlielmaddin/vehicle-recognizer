"""Dataset audit: facts only, changes nothing. (QAYDA 2)
Reports taxonomy, source crops, processed splits, integrity.
Usage: python scripts/audit_dataset.py
"""
from __future__ import annotations
import json, sys, io
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

def scan_dir(d: Path, depth: int = 2) -> dict[str, list[str]]:
    """depth=2: Make/Model/*.jpg (crops). depth=1: Make/*.jpg (processed splits)."""
    out: dict[str, list[str]] = defaultdict(list)
    if not d.exists():
        return out
    for jp in sorted(d.rglob("*.jpg")):
        rel = jp.relative_to(d).parts
        if len(rel) < depth + 1:
            continue
        out[rel[0]].append(jp.stem)
    return out

def main() -> None:
    tax = json.loads((ROOT / "data/metadata/taxonomy.json").read_text(encoding="utf-8"))
    print("=== AUDIT ===")
    print(f"Make taxonomy: {len(tax['makes'])}")
    print(f"Model taxonomy: {len(tax['models'])}")

    crops = scan_dir(ROOT / "data/interim/crops")
    n_crop = sum(len(v) for v in crops.values())
    print("\n=== DATASET (source: data/interim/crops) ===")
    print(f"Total images: {n_crop}")
    print(f"Total makes: {len(crops)}")
    per = sorted(crops.items(), key=lambda kv: -len(kv[1]))
    print("Per-make (top 15): " + ", ".join(f"{m}={len(v)}" for m, v in per[:15]))
    print("Per-make (tail): " + ", ".join(f"{m}={len(v)}" for m, v in per[-8:]))
    print(f"Min/max per make: {min(len(v) for v in crops.values())}/{max(len(v) for v in crops.values())}")
    single = [m for m, v in crops.items() if len(v) == 1]
    print(f"Single-image makes ({len(single)}): {sorted(single)}")

    tr = scan_dir(ROOT / "data/processed/makes/train", depth=1)
    va = scan_dir(ROOT / "data/processed/makes/val", depth=1)
    te = scan_dir(ROOT / "data/processed/makes/test", depth=1)
    n_tr = sum(len(v) for v in tr.values())
    n_va = sum(len(v) for v in va.values())
    n_te = sum(len(v) for v in te.values())
    print("\n=== TRAIN ===")
    print(f"Classes: {len(tr)}")
    print(f"Images: {n_tr}")
    print("\n=== VALIDATION ===")
    print(f"Classes: {len(va)}")
    print(f"Images: {n_va}")
    print("\n=== TEST ===")
    print(f"Classes: {len(te)}")
    print(f"Images: {n_te}")
    print("\n=== CLASS BALANCE (train) ===")
    import statistics
    cnts = sorted((len(v) for v in tr.values()))
    if cnts:
        print(f"Min: {cnts[0]}  Median: {statistics.median(cnts)}  Max: {cnts[-1]}")
    print("Per-class (class total train val test source):")
    for mk in sorted(set(tr) | set(va) | set(te)):
        tot = len(tr.get(mk, [])) + len(va.get(mk, [])) + len(te.get(mk, []))
        print(f"  {mk}: total={tot} train={len(tr.get(mk, []))} "
              f"val={len(va.get(mk, []))} test={len(te.get(mk, []))} source=turboaz")
    print("\n=== INTEGRITY ===")
    str_, sva, ste = set(tr), set(va), set(te)
    print(f"Class overlap: {len(str_ & sva & ste)}/{len(str_ | sva | ste)}")
    print(f"Missing train classes (val-only): {sorted(sva - str_)}")
    print(f"Missing val classes (train-only): {sorted(str_ - sva)}")
    print(f"Missing test classes: {sorted(str_ - ste)}")
    leak = 0
    for mk in str_ & sva:
        dup = set(tr[mk]) & set(va[mk])
        leak += len(dup)
        if dup:
            print(f"  LEAK {mk}: {sorted(dup)[:5]}")
    for mk in str_ & ste:
        dup = set(tr[mk]) & set(te[mk])
        leak += len(dup)
        if dup:
            print(f"  LEAK-T {mk}: {sorted(dup)[:5]}")
    print(f"Leakage: {leak}")
    print(f"Duplicate images: {leak} (same stem across splits)")
    print("\n=== MODELS ===")

    def pairsplit(s):
        root = ROOT / f"data/processed/models/{s}"
        out = {}
        if root.is_dir():
            for mkd in sorted(root.iterdir()):
                if not mkd.is_dir():
                    continue
                for mdd in sorted(mkd.iterdir()):
                    if mdd.is_dir():
                        out[f"{mkd.name} {mdd.name}"] = len(list(mdd.glob("*.jpg")))
        return out

    ptr, pva, pte = pairsplit("train"), pairsplit("val"), pairsplit("test")
    ctr = sum(ptr.values())
    print(f"Pairs train/val/test: {len(ptr)}/{len(pva)}/{len(pte)} "
          f"images {ctr}/{sum(pva.values())}/{sum(pte.values())}")
    allp = set(ptr) | set(pva) | set(pte)
    print(f"Pair overlap: {len(set(ptr) & set(pva) & set(pte))}/{len(allp)}")
    cn = sorted(ptr.values())
    if cn:
        print(f"Balance min/median/max: {cn[0]}/{statistics.median(cn)}/{cn[-1]}")
    bad = 0
    for d in (ROOT / "data/processed/makes/train", ROOT / "data/processed/makes/val"):
        for jp in d.rglob("*.jpg"):
            if jp.stat().st_size == 0:
                bad += 1
    print(f"Invalid images: {bad}")

if __name__ == "__main__":
    main()
