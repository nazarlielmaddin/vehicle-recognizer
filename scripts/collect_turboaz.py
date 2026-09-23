"""turbo.az collector — polite, provenance-tracked, ToS-gated.

- Respects https://turbo.az/robots.txt (only /*bookmarks is disallowed; /autos allowed).
- Polite by default: 2.5s delay + jitter, browser User-Agent, per-make quotas.
- Every image gets meta.json provenance (listing URL, title, parsed labels, license note).
- Images belong to listing owners / turbo.az: RESEARCH/BOOTSTRAP use only.
  Do not redistribute. If turbo.az ToS changes, stop using this script.
- Requires --confirm-tos (you accept responsibility for ToS compliance).

Usage:
  python scripts/collect_turboaz.py --pages 5 --per-make 25 --out data/raw/turboaz --confirm-tos
"""
from __future__ import annotations
import json, random, re, time, urllib.robotparser
from html.parser import HTMLParser
from pathlib import Path
from datetime import date
import typer

try:
    from training.taxonomy import normalize_make, normalize_model
except ImportError:  # script run from repo root without package context
    import sys
    sys.path.insert(0, ".")
    from training.taxonomy import normalize_make, normalize_model

app = typer.Typer()
BASE = "https://turbo.az"
UA = {"User-Agent": "vehicle-recognizer-research/0.1 (+contact: repo docs/DATASETS.md)"}

CARD_RE = re.compile(r'<a[^>]+href="(/autos/\d+[^"]*)"[^>]*>(.*?)</a>', re.S)
IMG_RE = re.compile(r'https://turbo\.azstatic\.com/uploads/(?:full|big)/[A-Za-z0-9/_\-.]+\.(?:jpg|jpeg|png|webp)', re.I)
TITLE_CLEAN = re.compile(r"<[^>]+>")

def robots_ok() -> bool:
    """Fetch robots.txt with our real UA (turbo.az blocks urllib's default UA)
    and check /autos manually. Fail closed on any error."""
    import httpx
    try:
        r = httpx.get(BASE + "/robots.txt", headers=UA, timeout=20, follow_redirects=True)
        if r.status_code != 200:
            return False
        disallows = [ln.split(":", 1)[1].strip() for ln in r.text.splitlines()
                     if ln.lower().startswith("disallow:")]
        return not any(d in ("/autos", "/*", "/") for d in disallows)
    except Exception:
        return False

def polite_get(client, url: str, delay: float) -> str:
    time.sleep(delay + random.random())
    import httpx
    r = client.get(url, timeout=30)
    r.raise_for_status()
    return r.text

BLOCK_RE = re.compile(r'<div class="products-i[ "]', re.S)
LINK_RE = re.compile(r'<a class="products-i__link"[^>]+href="(/autos/\d+[^"]*)"')
NAME_RE = re.compile(r'<div class="products-i__name[^"]*">([^<]+)</div>')
THUMB_RE = re.compile(r'<img[^>]+src="(https://turbo\.azstatic\.com/uploads/[^"]+)"')

def parse_cards(html: str) -> list[tuple[str, str, str]]:
    """(detail_url, title_text, thumb_url) — matches real turbo.az markup:
    empty products-i__link anchor + products-i__name div + img thumbnail."""
    out = []
    for block in BLOCK_RE.split(html)[1:]:
        block = block[:6000]  # one card is ~2-4KB; bound the search
        lm = LINK_RE.search(block)
        nm = NAME_RE.search(block)
        if not lm or not nm:
            continue
        tm = THUMB_RE.search(block)
        title = re.sub(r"\s+", " ", nm.group(1)).strip()
        if len(title) > 1:
            out.append((BASE + lm.group(1), title, tm.group(1) if tm else ""))
    seen, res = set(), []
    for u, t, th in out:
        if u not in seen:
            seen.add(u); res.append((u, t, th))
    return res

NON_CAR_MAKES = {"voge", "honda moto", "yamaha", "kawasaki", "suzuki moto",
                 "bmw moto", "harley", "ducati", "ktm", "bajaj", "lifan moto",
                 " Loncin", "zongshen", "aprilia", "vespa", "sym", "kymco",
                 "cfmoto", "cf moto", "royal enfield", "royal", "can-am",
                 "canam", "benda", "harley-davidson", "indian moto", "yadea"}

def parse_make_model(title: str) -> tuple[str, str] | None:
    """Titles look like 'Volkswagen Passat' or 'Mercedes EQS 580 4MATIC SUV'.
    Returns None for motorcycles / unparseable titles."""
    import html as _html
    import re as _re2
    title = _html.unescape(title)  # turbo.az emits &amp; &quot; etc.
    title = _re2.sub(r"\([^)]*\)", "", title)  # drop "(Great Wall Motor)"-style notes
    first = title.split(",")[0].strip()
    parts = [p for p in first.split() if p != "&"]
    if len(parts) < 2:
        # single token like "900" after cleanup is a trim, not a vehicle
        return None
    if parts[0].lower() in NON_CAR_MAKES or " ".join(parts[:2]).lower() in NON_CAR_MAKES:
        return None  # two-wheeler, out of scope
    # try 1-word then 2-word make (e.g. 'Land Rover', 'Iran Khodro', 'Mercedes-Benz')
    # a 2-word candidate wins ONLY if it is a known make (alias table hit)
    import re as _re
    from training.taxonomy import MAKE_ALIASES as _ALIASES
    for n in (2, 1):
        cand = " ".join(parts[:n])
        key = _re.sub(r"\s+", " ", _re.sub(r"[^a-z0-9]+", " ", cand.lower()).strip())
        if n == 2 and key not in _ALIASES:
            continue
        mk = normalize_make(cand)
        rest = " ".join(parts[n:])
        if rest:
            md = normalize_model(mk, rest.split(",")[0].strip())  # keep original case
            return mk, md
    return None

@app.command()
def main(pages: int = 5, per_make: int = 25, imgs_per_listing: int = 3,
         out: str = "data/raw/turboaz", delay: float = 2.5,
         confirm_tos: bool = typer.Option(False, help="Required: you accept turbo.az ToS responsibility")):
    import httpx
    if not confirm_tos:
        raise typer.BadParameter("pass --confirm-tos (see docstring: you are responsible for ToS compliance)")
    if not robots_ok():
        raise typer.Exit("robots.txt disallows /autos — aborting")
    root = Path(out); root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    # resume: seed quotas from already-downloaded provenance (no re-download)
    for mp in root.rglob("*.meta.json"):
        try:
            mk = json.loads(mp.read_text(encoding="utf-8")).get("make", "")
            if mk:
                counts[mk] = counts.get(mk, 0) + 1
        except Exception:
            pass
    print(f"resumed counts: {len(counts)} makes, {sum(counts.values())} images")
    client = httpx.Client(headers=UA, follow_redirects=True)
    for pg in range(1, pages + 1):
        try:
            html = polite_get(client, f"{BASE}/autos?page={pg}", delay)
        except Exception as e:
            print(f"page {pg} failed: {e}"); continue
        for url, title, thumb in parse_cards(html):
            mm = parse_make_model(title)
            if not mm:
                continue
            mk, md = mm
            if counts.get(mk, 0) >= per_make:
                continue
            d = root / mk.replace(" ", "_") / md.replace(" ", "_").replace("/", "-")
            d.mkdir(parents=True, exist_ok=True)
            lid = re.search(r"/autos/(\d+)", url)
            lid = lid.group(1) if lid else "x"
            got = 0
            # 1) thumbnail straight from the listing (1 request saved per card)
            # try full-size by URL rewrite first (f460x343 -> full), fall back to thumb
            cands = []
            if thumb:
                cands.append(re.sub(r"/uploads/f\d+x\d+/", "/uploads/full/", thumb))
                cands.append(thumb)
            for cand_url in cands:
                if counts.get(mk, 0) >= per_make or got >= 1:
                    break
                try:
                    time.sleep(delay)
                    blob = client.get(cand_url, timeout=30).content
                    if len(blob) >= 8000:
                        stem = f"{lid}_t{counts.get(mk,0)}"
                        (d / f"{stem}.jpg").write_bytes(blob)
                        (d / f"{stem}.meta.json").write_text(json.dumps({
                            "source": "turbo.az", "source_url": url, "image_url": cand_url,
                            "dataset": "turboaz-bootstrap", "make": mk, "model": md,
                            "title": title, "viewpoint": "UNKNOWN", "lighting": "UNKNOWN",
                            "resolution": "full-rewrite" if "/full/" in cand_url else "thumb-460",
                            "license": "© listing owner / turbo.az — research/bootstrap only, do not redistribute",
                            "acquired": str(date.today()),
                        }, ensure_ascii=False, indent=2), encoding="utf-8")
                        counts[mk] = counts.get(mk, 0) + 1
                        got += 1
                except Exception as e:
                    print(f"thumb failed: {e}")
            # 2) full-size from detail page only if quota still open
            if got < imgs_per_listing and counts.get(mk, 0) < per_make:
                try:
                    detail = polite_get(client, url, delay)
                except Exception as e:
                    print(f"detail failed {url}: {e}"); continue
                imgs = list(dict.fromkeys(IMG_RE.findall(detail)))[:imgs_per_listing - got]
                for im_url in imgs:
                    if counts.get(mk, 0) >= per_make:
                        break
                    try:
                        time.sleep(delay)
                        blob = client.get(im_url, timeout=30).content
                        if len(blob) < 8000:
                            continue
                        stem = f"{lid}_{counts.get(mk,0)}"
                        (d / f"{stem}.jpg").write_bytes(blob)
                        (d / f"{stem}.meta.json").write_text(json.dumps({
                            "source": "turbo.az", "source_url": url, "image_url": im_url,
                            "dataset": "turboaz-bootstrap", "make": mk, "model": md,
                            "title": title, "viewpoint": "UNKNOWN", "lighting": "UNKNOWN",
                            "resolution": "full",
                            "license": "© listing owner / turbo.az — research/bootstrap only, do not redistribute",
                            "acquired": str(date.today()),
                        }, ensure_ascii=False, indent=2), encoding="utf-8")
                        counts[mk] = counts.get(mk, 0) + 1
                    except Exception as e:
                        print(f"img failed: {e}")
        print(f"page {pg}: {dict(counts)}")
        if all(v >= per_make for v in counts.values()) and counts:
            break
    Path("reports/turboaz_collection.json").write_text(
        json.dumps({"counts": counts, "date": str(date.today())}, ensure_ascii=False, indent=2))
    print("DONE:", dict(counts))

if __name__ == "__main__":
    app()
