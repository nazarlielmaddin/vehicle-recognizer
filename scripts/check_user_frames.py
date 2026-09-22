"""Ad-hoc check on user-provided radar frames (not committed as fixtures)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import cv2, json, glob
import numpy as np
from inference.pipeline import VehiclePipeline

def load(path: str):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

p = VehiclePipeline()
files = sorted(glob.glob(r"C:\Users\enaza\OneDrive\Pictures\Screenshots\Ekran şəkli 2026-09-22 17*.png"))
print("files:", files)
for f in files:
    bgr = load(f)
    if bgr is None:
        print("UNREADABLE:", f); continue
    print("=" * 80, "\n", f, bgr.shape)
    r = p.infer_image(bgr)
    print("quality:", r["quality"]["quality"], r["quality"]["lighting"], "| status:", r["status"])
    for v in r.get("vehicles", []):
        print(f"  make={v['make']} ({v['make_conf']}) model={v['model']} ({v['confidence']}) "
              f"body={v['body_type']} view={v['orientation']} status={v['status']} src={v.get('evidence_source')}")
        print("  alt:", [(a["label"], a["confidence"]) for a in v["alternatives"]])
        print("  why:", v["abstain_reasons"][:3])
    if r["status"] in ("NO_VEHICLE", "DETECTOR_UNAVAILABLE"):
        print("  msg:", r.get("message"))
    # badge debug
    try:
        from inference.detector import VehicleDetector
        from inference.crops import extract_crops
        from inference.badge import BadgeReader
        dets = VehicleDetector().detect(bgr)
        crops = extract_crops(bgr, dets, 0.12)
        if crops:
            br = BadgeReader(p.zero_shot).read_make(crops[0]["full"], p.make_clf.classes, k=5)
            print("  badge:", [(m, round(s, 3)) for m, s in br])
    except Exception as e:
        print("  badge ERR:", str(e)[:200])
