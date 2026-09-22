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
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(BASE + "/robots.txt")
        rp.read()
        return rp.can_fetch("*", BASE + "/autos")
    except Exception:
        return False

def polite_get(client, url: str, delay: float) -> str:
    time.sleep(delay + random.random())
    import httpx
    r = client.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def parse_cards(html: str) -> list[tuple[str, str]]:
    """(detail_url, title_text) — defensive, skips unparseable cards."""
    out = []
    for href, inner in CARD_RE.findall(html):
        title = TITLE_CLEAN.sub(" ", inner)
        title = re.sub(r"\s+", " ", title).strip()
        if "/autos/" in href and len(title) > 3:
            out.append((BASE + href.split('"')[0], title))
    # dedupe by url
    seen, res = set(), []
    for u, t in out:
        if u not in seen:
            seen.add(u); res.append((u, t))
    return res

def parse_make_model(title: str) -> tuple[str, str] | None:
    """Titles look like 'Volkswagen Passat, 2.0 L, 2019 il, 145 000 km'."""
    first = title.split(",")[0].strip()
    parts = first.split()
    if len(parts) < 2:
        return None
    # try 1-word then 2-word make (e.g. 'Land Rover', 'Iran Khodro', 'Mercedes-Benz')
    for n in (2, 1):
        cand = " ".join(parts[:n])
        mk = normalize_make(cand)
        rest = " ".join(parts[n:])
        if rest and mk.lower() != cand.lower().replace("  ", " ") or n == 1:
            md = normalize_model(mk, rest.split(",")[0].split("  ")[0].strip().title())
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
    client = httpx.Client(headers=UA, follow_redirects=True)
    for pg in range(1, pages + 1):
        try:
            html = polite_get(client, f"{BASE}/autos?page={pg}", delay)
        except Exception as e:
            print(f"page {pg} failed: {e}"); continue
        for url, title in parse_cards(html):
            mm = parse_make_model(title)
            if not mm:
                continue
            mk, md = mm
            if counts.get(mk, 0) >= per_make:
                continue
            try:
                detail = polite_get(client, url, delay)
            except Exception as e:
                print(f"detail failed {url}: {e}"); continue
            imgs = list(dict.fromkeys(IMG_RE.findall(detail)))[:imgs_per_listing]
            if not imgs:
                continue
            d = root / mk.replace(" ", "_") / md.replace(" ", "_").replace("/", "-")
            d.mkdir(parents=True, exist_ok=True)
            for im_url in imgs:
                if counts.get(mk, 0) >= per_make:
                    break
                try:
                    time.sleep(delay)
                    blob = client.get(im_url, timeout=30).content
                    if len(blob) < 8000:
                        continue
                    lid = re.search(r"/autos/(\d+)", url)
                    stem = f"{lid.group(1) if lid else 'x'}_{counts.get(mk,0)}"
                    (d / f"{stem}.jpg").write_bytes(blob)
                    (d / f"{stem}.meta.json").write_text(json.dumps({
                        "source": "turbo.az", "source_url": url, "image_url": im_url,
                        "dataset": "turboaz-bootstrap", "make": mk, "model": md,
                        "title": title, "viewpoint": "UNKNOWN", "lighting": "UNKNOWN",
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
