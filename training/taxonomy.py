"""Taxonomy: normalized makes/models/generations. Grows without app rebuild."""
from __future__ import annotations
import json, re, unicodedata
from pathlib import Path

MAKE_ALIASES = {
    "mercedes benz": "Mercedes-Benz", "mercedes": "Mercedes-Benz",
    "vw": "Volkswagen", "volkswagen": "Volkswagen",
    "bmw": "BMW", "toyota": "Toyota", "honda": "Honda", "ford": "Ford",
    "hyundai": "Hyundai", "kia": "Kia", "audi": "Audi", "nissan": "Nissan",
    "renault": "Renault", "peugeot": "Peugeot", "skoda": "Skoda",
    "lada": "Lada", "vaz": "Lada",
    "gaz": "GAZ", "uaz": "UAZ", "zaz": "ZAZ",
    "land rover": "Land Rover", "range rover": "Land Rover",
    "iran khodro": "Iran Khodro", "iran khodrou": "Iran Khodro", "ikco": "Iran Khodro",
    "saipa": "Saipa", "daewoo": "Daewoo", "ravon": "Ravon",
    "chevrolet": "Chevrolet", "chevrole": "Chevrolet", "chevy": "Chevrolet",
    "volkswagen": "Volkswagen",
    "great wall": "Great Wall", "haval": "Haval", "geely": "Geely",
    "chery": "Chery", "lifan": "Lifan", "dacia": "Dacia",
    "citroen": "Citroen", "fiat": "Fiat", "jeep": "Jeep",
    "volvo": "Volvo", "porsche": "Porsche", "tesla": "Tesla",
    "infiniti": "Infiniti", "lexus": "Lexus", "mazda": "Mazda",
    "mitsubishi": "Mitsubishi", "subaru": "Subaru", "suzuki": "Suzuki",
    "opel": "Opel",
    "changan": "Changan", "byd": "BYD", "mg": "MG", "morris garages": "MG",
    "jac": "JAC", "jetour": "Jetour", "exeed": "Exeed", "omoda": "Omoda",
    "dongfeng": "Dongfeng", "dfsk": "DFSK", "faw": "FAW", "bestune": "FAW",
    "hongqi": "Hongqi", "baic": "BAIC",
    "zeekr": "Zeekr", "li auto": "Li Auto", "lixiang": "Li Auto", "li": "Li Auto",
    "nio": "NIO", "xpeng": "XPeng", "wuling": "Wuling",
    "moskvich": "Moskvich", "moskvitch": "Moskvich", "azlk": "Moskvich",
    "seres": "Seres", "aito": "Seres",
    "gac": "GAC", "trumpchi": "GAC",
    "cfmoto": "CFMoto", "cf moto": "CFMoto",
    "gwm": "Great Wall", "great wall": "Great Wall",
    "lynk": "Lynk&Co", "lynk&co": "Lynk&Co", "lynkco": "Lynk&Co",
    "jmc": "JMC", "vgv": "VGV", "soueast": "Soueast",
    "royal": "Royal Enfield", "royal enfield": "Royal Enfield",
    "can-am": "Can-Am", "canam": "Can-Am", "benda": "Benda",
    "jaecoo": "Jaecoo", "voyah": "Voyah", "rox": "Rox",
    "seat": "Seat", "mini": "Mini", "gmc": "GMC",
    "cadillac": "Cadillac", "maserati": "Maserati", "ferrari": "Ferrari",
    "bentley": "Bentley", "man": "MAN", "kamaz": "KamAZ", "shacman": "Shacman",
    "radar": "Radar", "mercedes-maybach": "Mercedes-Benz", "maybach": "Mercedes-Benz",
    "yadea": "Yadea",
    "howo": "HOWO", "sinotruk": "HOWO", "sitrak": "SITRAK",
    "voge": "Voge",
}
MODEL_ALIASES = {
    ("mercedes-benz", "c class"): "C-Class", ("mercedes-benz", "c-class"): "C-Class",
    ("mercedes-benz", "e class"): "E-Class", ("mercedes-benz", "s class"): "S-Class",
    ("bmw", "3 series"): "3 Series", ("bmw", "5 series"): "5 Series",
    ("toyota", "land cruiser"): "Land Cruiser", ("toyota", "rav 4"): "RAV4",
}

def normalize_make(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    key = re.sub(r"\s+", " ", key)
    return MAKE_ALIASES.get(key, name.strip().title() if name.isupper() else name.strip())

def normalize_model(make: str, model: str) -> str:
    mk = normalize_make(make)
    key = re.sub(r"[^a-z0-9]+", " ", model.lower()).strip()
    return MODEL_ALIASES.get((mk.lower(), key), model.strip())

def build_taxonomy(entries: list[dict]) -> dict:
    """entries: [{make, model, generation?, year_from?, year_to?, body_type?, aliases?}]"""
    makes: dict[str, dict] = {}
    for e in entries:
        mk = normalize_make(e["make"])
        md = normalize_model(mk, e["model"])
        makes.setdefault(mk, {}).setdefault(md, []).append(e)
    return {
        "makes": sorted(makes),
        "models": sorted(f"{mk} {md}" for mk, v in makes.items() for md in v),
        "detail": makes,
        "version": "0.1.0",
    }

def save_taxonomy(tax: dict, path: str = "data/metadata/taxonomy.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(tax, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

SEED_ENTRIES = [
    {"make": "Toyota", "model": "Camry", "body_type": "Sedan"},
    {"make": "Toyota", "model": "Corolla", "body_type": "Sedan"},
    {"make": "Toyota", "model": "RAV4", "body_type": "SUV"},
    {"make": "Toyota", "model": "Land Cruiser", "body_type": "SUV"},
    {"make": "Toyota", "model": "Hilux", "body_type": "Pickup"},
    {"make": "BMW", "model": "3 Series", "body_type": "Sedan"},
    {"make": "BMW", "model": "5 Series", "body_type": "Sedan"},
    {"make": "BMW", "model": "X3", "body_type": "SUV"},
    {"make": "BMW", "model": "X5", "body_type": "SUV"},
    {"make": "Mercedes-Benz", "model": "C-Class", "body_type": "Sedan"},
    {"make": "Mercedes-Benz", "model": "E-Class", "body_type": "Sedan"},
    {"make": "Mercedes-Benz", "model": "S-Class", "body_type": "Sedan"},
    {"make": "Mercedes-Benz", "model": "GLC", "body_type": "SUV"},
    {"make": "Mercedes-Benz", "model": "Sprinter", "body_type": "Van"},
]
