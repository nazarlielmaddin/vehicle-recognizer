"""Production inference pipeline — the full required chain.

INPUT → QUALITY → DETECTION → CROP → VIEWPOINT → FEATURES → MAKE → MODEL
→ BODY → FUSION → CALIBRATION → UNKNOWN → FINAL RESULT
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import yaml

from .quality import analyze_quality, to_dict
from .detector import VehicleDetector
from .crops import extract_crops, letterbox
from .classifiers import TimmClassifier
from .retrieval import EmbeddingExtractor, EmbeddingStore
from .fusion import fuse_classifier_retrieval, decide_unknown, entropy, softmax_with_temperature
from .zeroshot import ClipZeroShot, VIEW_PROMPTS, BODY_PROMPTS

VIEWPOINT = ["FRONT","REAR","LEFT","RIGHT","FRONT_LEFT","FRONT_RIGHT","REAR_LEFT","REAR_RIGHT","UNKNOWN"]
BODY = ["Sedan","Hatchback","Coupe","Convertible","Wagon","SUV","Crossover","MPV","Pickup",
        "Van","PanelVan","LightTruck","HeavyTruck","Bus","Minibus","Other","Unknown"]

def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def _prep_rgb(crop: np.ndarray, size: int = 224) -> np.ndarray:
    lb = letterbox(crop, (size, size))
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # ImageNet normalize
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    rgb = (rgb - mean) / std
    return np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)

class VehiclePipeline:
    def __init__(self, config: str = "configs/default.yaml"):
        self.cfg = _load_cfg(config)
        d = self.cfg["detection"]
        self.detector = VehicleDetector(self.cfg["paths"]["detector_weights"],
                                        d["conf"], d["iou"], self.cfg["image"]["detector_imgsz"],
                                        self.cfg["system"]["device"])
        # class lists come from taxonomy if present, else minimal fallback
        tax = Path(self.cfg["paths"]["taxonomy"])
        self._model_body: dict[str, str] = {}
        if tax.exists():
            import json
            t = json.loads(tax.read_text(encoding="utf-8"))
            makes = t.get("makes", ["Unknown"])
            models = t.get("models", ["Unknown"])
            for mk, sub in (t.get("detail") or {}).items():
                for md, entries in (sub or {}).items():
                    bts = [e.get("body_type", "Unknown") for e in entries or []]
                    body = max(set(bts), key=bts.count) if bts else "Unknown"
                    self._model_body[f"{mk} {md}"] = body
        else:
            makes, models = ["Unknown"], ["Unknown"]
        self.full_makes = list(makes)  # full taxonomy coverage (anti-forgetting backstop)
        P = self.cfg["paths"]
        self.make_clf = TimmClassifier(P["make_weights"], makes)
        self.model_clf = TimmClassifier(P["model_weights"], models)
        self.body_clf = TimmClassifier(P["body_weights"], BODY)
        self.view_clf = TimmClassifier(P["viewpoint_weights"], VIEWPOINT)
        self.embed = EmbeddingExtractor(weights=P["embedding_weights"])
        self.store = EmbeddingStore.load(P["vector_index"])
        self.T = float(self.cfg["calibration"].get("temperature", 1.0))
        self._zs = None  # lazy zero-shot fallback, built on first need
        self._last_gate_note = ""

    @property
    def zero_shot(self) -> ClipZeroShot:
        if self._zs is None:
            device = "cuda" if self.make_clf.device == "cuda" else "cpu"
            self._zs = ClipZeroShot(device=device)
        return self._zs

    @staticmethod
    def _allowed_bodies(det_label: str) -> set[str]:
        """YOLO-class gate: a 'car' crop must never yield truck/bus models."""
        if det_label == "truck":
            return {"LightTruck", "HeavyTruck", "Pickup", "PanelVan", "Van"}
        if det_label == "bus":
            return {"Bus", "Minibus"}
        return {"Sedan", "Hatchback", "Coupe", "Convertible", "Wagon", "SUV",
                "Crossover", "MPV", "Pickup", "Van", "PanelVan", "LightTruck",
                "Other", "Unknown"}

    def _gate_models(self, top: list[tuple[str, float]], det_label: str,
                     k: int = 5) -> list[tuple[str, float]]:
        allowed = self._allowed_bodies(det_label)
        kept = [x for x in top if self._model_body.get(x[0], "Unknown") in allowed]
        dropped = len(top) - len(kept)
        self._last_gate_note = (
            f"yolo={det_label}: {dropped} incompatible body types excluded" if dropped else "")
        return kept[:k] if kept else top[:k]

    def _predict_head(self, clf: TimmClassifier, rgb: np.ndarray, k=5):
        if not clf.available:
            return [("Unknown", 0.0)], np.array([1.0])
        proba = clf.predict_proba(rgb)[0]
        return TimmClassifier.topk(proba, clf.classes, k), proba

    def _zs_head(self, crop_bgr: np.ndarray, classes: list[str], kind: str, k=5, feat=None):
        """Zero-shot fallback — returns (topk, proba_like, ok). Reuses image feat when given."""
        try:
            zs = self.zero_shot
            if feat is None:
                feat = zs.encode_image(crop_bgr)
            if kind == "viewpoint":
                # rank only real orientations; UNKNOWN is a fallback, not a candidate
                labels = [v for v in VIEWPOINT if v != "UNKNOWN"]
                top = zs.rank_with(feat, [VIEW_PROMPTS[l] for l in labels], labels, k=min(k, 3))
                if not top or top[0][1] < 0.25:
                    return [("UNKNOWN", 1.0)], np.array([1.0]), True
            elif kind == "body":
                top = zs.rank_with(feat, [BODY_PROMPTS.get(l, f"a {l}") for l in classes], classes, k)
            elif kind == "make":
                top = zs.rank_with(feat, [f"a {m} car" for m in classes], classes, k)
            else:
                top = zs.rank_with(feat, [f"a photo of a {m} car" for m in classes], classes, k)
            p = np.zeros(max(len(classes), 1))
            for label, prob in top:
                if label in classes:
                    p[classes.index(label)] = prob
            return top, p, True
        except Exception:
            return [("Unknown", 0.0)], np.array([1.0]), False

    @staticmethod
    def _blend_make_distributions(trained_top, trained_proba, trained_classes,
                                  zs_top, zs_proba, full_makes) -> tuple[list, bool]:
        """Anti-forgetting backoff: P = lam*P_trained + (1-lam)*P_zs over union,
        lam = trained top1 confidence. Returns (ranked top-5, used_backstop)."""
        lam = float(trained_top[0][1]) if trained_top else 0.0
        if lam >= 0.5:
            return trained_top, False
        blended: dict[str, float] = {}
        for j, cls in enumerate(trained_classes):
            if j < len(trained_proba):
                blended[cls] = blended.get(cls, 0.0) + lam * float(trained_proba[j])
        for j, cls in enumerate(full_makes):
            if j < len(zs_proba):
                blended[cls] = blended.get(cls, 0.0) + (1.0 - lam) * float(zs_proba[j])
        tot = sum(blended.values()) + 1e-9
        ranked = sorted(blended.items(), key=lambda kv: -kv[1])
        return [(k, v / tot) for k, v in ranked[:5]], True

    @staticmethod
    def _constrain_to_make(fused: list, make_name: str,
                           same_min: float) -> tuple[list, str]:
        """Fantasy pairs forbidden: keep same-make candidates, else empty + note."""
        if make_name == "Unknown":
            return fused, ""
        same = [x for x in fused if x[0] == make_name or x[0].startswith(make_name + " ")]
        if same and same[0][1] >= same_min:
            return same + [x for x in fused if x not in same], ""
        dropped = fused[0][0] if fused else "none"
        if same:
            return [], (f"best {make_name} model {same[0][0]} too weak "
                        f"({same[0][1]:.2f}) — model unknown")
        return [], f"model {dropped} excluded (not {make_name}) — model unknown"

    def infer_image(self, bgr: np.ndarray) -> dict:
        q = analyze_quality(bgr)
        try:
            dets = self.detector.detect(bgr)
        except RuntimeError as e:
            # detector backend (ultralytics+torch) not installed → still return
            # quality analysis + honest status so the UI remains testable
            return {"quality": to_dict(q), "vehicles": [],
                    "status": "DETECTOR_UNAVAILABLE", "message": str(e)}
        crops = extract_crops(bgr, dets, self.cfg["detection"]["crop_context"])
        # no vehicle found → honest UNKNOWN, not a forced label
        if not crops:
            return {"quality": to_dict(q), "vehicles": [],
                    "status": "NO_VEHICLE", "message": "No vehicle detected."}
        vehicles = []
        for i, c in enumerate(crops):
            rgb = _prep_rgb(c["full"], self.cfg["image"]["classifier_size"][0])
            zs_used = False
            # encode once for all zero-shot heads (was 4 separate encodes)
            zfeat = None
            if not (self.make_clf.available and self.model_clf.available
                    and self.body_clf.available and self.view_clf.available):
                try:
                    zfeat = self.zero_shot.encode_image(c["full"])
                except Exception:
                    zfeat = None
            # HYBRID MAKE (anti catastrophic forgetting): trained head knows its
            # classes well; zero-shot backstop keeps the other taxonomy makes
            # recognizable. Blend weight follows trained confidence (adaptive):
            # confident-trained → trained dominates; flat-trained → zs dominates.
            make_hybrid = False
            if self.make_clf.available:
                make_top_t, make_p_t = self._predict_head(self.make_clf, rgb)
                make_top_z, make_p_z, ok_z = self._zs_head(
                    c["full"], self.full_makes, "make", feat=zfeat)
                zs_used = zs_used or ok_z
                make_top, make_hybrid = self._blend_make_distributions(
                    make_top_t, make_p_t, self.make_clf.classes,
                    make_top_z, make_p_z, self.full_makes)
                if make_hybrid:
                    zs_used = True
                    make_p = make_p_t
                else:
                    make_top, make_p = make_top_t, make_p_t
            else:
                make_top, make_p, ok = self._zs_head(c["full"], self.make_clf.classes, "make", feat=zfeat)
                zs_used = zs_used or ok
            if self.model_clf.available:
                model_top, model_p = self._predict_head(self.model_clf, rgb)
            else:
                model_top, model_p, ok = self._zs_head(
                    c["full"], self.model_clf.classes, "model", k=15, feat=zfeat)
                zs_used = zs_used or ok
            # hard gate BEFORE fusion: incompatible bodies can never win
            model_top = self._gate_models(model_top, c.get("label_det", "car"), k=5)
            if self.body_clf.available:
                body_top, body_p = self._predict_head(self.body_clf, rgb)
            else:
                body_top, body_p, ok = self._zs_head(c["full"], BODY, "body", feat=zfeat)
                zs_used = zs_used or ok
            if self.view_clf.available:
                view_top, _ = self._predict_head(self.view_clf, rgb, k=3)
            else:
                view_top, _, ok = self._zs_head(c["full"], VIEWPOINT, "viewpoint", k=3, feat=zfeat)
                zs_used = zs_used or ok
            # badge/logo evidence (gate frames: emblem often clearly visible)
            badge_top: list[tuple[str, float]] = []
            if not self.make_clf.available and zfeat is not None:
                try:
                    from .badge import BadgeReader
                    badge_top = BadgeReader(self.zero_shot).read_make(
                        c["full"], self.make_clf.classes, k=5)
                except Exception:
                    badge_top = []
            if badge_top and badge_top[0][0] != "Unknown" and badge_top[0][1] >= 0.40:
                bmake, bprob = badge_top[0]
                head_prob = dict(make_top).get(bmake, 0.0)
                make_top = [(bmake, 0.5 * bprob + 0.5 * head_prob)] + [
                    (l, p) for l, p in make_top if l != bmake][:4]
                reasons_badge = [f"emblem reads {bmake}"]
            else:
                reasons_badge = []
            try:
                vec = self.embed.extract(rgb)
                nn = self.store.search(vec, k=5)
            except Exception as e:  # torch/timm missing or no index → retrieval skipped honestly
                nn = []
                nn_error = str(e)
            else:
                nn_error = ""
            nn_sim = nn[0][1] if nn else 0.0
            fused = fuse_classifier_retrieval(
                model_top, nn,
                self.cfg["fusion"]["w_classifier"], self.cfg["fusion"]["w_retrieval"])
            fused_all = list(fused)
            # HIERARCHICAL CONSISTENCY — fantasy pairs ("Changan NIO ES6",
            # "Lada BYD Seagull") are a hard failure and are forbidden:
            # the shown model must belong to the shown make, always.
            make_name, make_conf = make_top[0]
            same_min = float(self.cfg["unknown"].get("same_make_model_min", 0.10))
            fused, hier_note = self._constrain_to_make(fused, make_name, same_min)
            if fused:
                top1 = fused[0][1]
                top2 = fused[1][1] if len(fused) > 1 else 0.0
                model_label = fused[0][0]
            else:
                top1 = make_conf if make_name != "Unknown" else 0.0
                top2 = 0.0
                model_label = "Unknown"
            ent = entropy(model_p) if (self.model_clf.available or zs_used) else 99.0
            u = self.cfg["unknown"]
            abstain, reasons = decide_unknown(
                top1, top2, ent, nn_sim, q.quality,
                u["min_confidence"], u["min_margin"], u["max_entropy"],
                u["min_nn_similarity"], u["poor_quality_abstain"])
            trained = self.model_clf.available and self.make_clf.available
            if not fused and make_name != "Unknown" and (
                    self.make_clf.available or make_conf >= 0.4):
                # make known, model unknown — review, never a fantasy pair
                status = "UNCERTAIN"
                reasons = [hier_note + " (confidence = make-level)"] + reasons
            elif abstain:
                status = "UNKNOWN"
            elif not trained and zs_used:
                # zero-shot evidence only → real guess, but never CONFIDENT
                status = "UNCERTAIN"
                reasons = ["evidence: zero-shot CLIP (untrained fallback) — train classifiers for CONFIDENT"] + reasons
            elif not trained:
                status = "UNKNOWN"
            else:
                status = "CONFIDENT" if top1 >= 0.75 else "UNCERTAIN"
            if not trained and not zs_used:
                reasons = ["classifiers untrained — see docs/TRAINING.md"] + reasons
            if nn_error:
                reasons = [f"retrieval unavailable: {nn_error}"] + reasons
            reasons = reasons_badge + reasons
            if make_hybrid:
                reasons = [f"hybrid make (trained uncertain → zero-shot backstop blended, "
                           f"top={make_top[0][0]} {make_top[0][1]:.2f})"] + reasons
            if self._last_gate_note:
                reasons = [self._last_gate_note] + reasons
            # alternatives: same-make confusions first (most likely look-alikes);
            # drawn from the pre-constraint list so review keeps full context
            mk_name = make_top[0][0]
            alt_pool = list(fused_all[1:6] if fused else fused_all[:5])
            same = [x for x in alt_pool if x[0] == mk_name or x[0].startswith(mk_name + " ")]
            alt_ordered = (same + [x for x in alt_pool if x not in same])[:3]
            vehicles.append({
                "id": i + 1, "box": c["box"], "det_conf": round(float(c["conf_det"]), 3),
                "evidence_source": ("trained" if trained
                                    else ("trained-make+zero-shot" if self.make_clf.available
                                          else ("zero-shot-clip" if zs_used else "none"))),
                "make": make_top[0][0], "make_conf": round(float(make_top[0][1]), 4),
                "model": model_label,
                "confidence": round(float(top1), 4),
                "body_type": body_top[0][0], "body_conf": round(float(body_top[0][1]), 4),
                "orientation": view_top[0][0],
                "status": status,
                "alternatives": [{"label": l, "confidence": round(float(p), 4)} for l, p in alt_ordered],
                "nn_similarity": round(float(nn_sim), 4),
                "entropy": round(float(ent), 3),
                "abstain_reasons": reasons,
            })
        return {"quality": to_dict(q), "vehicles": vehicles, "status": "OK"}

    def infer_multi_view(self, images: list[np.ndarray]) -> dict:
        """Multi-image fusion: average fused distributions across views."""
        per, agg = [], {}
        for im in images:
            r = self.infer_image(im)
            per.append(r)
            for v in r.get("vehicles", [])[:1]:
                agg[v["model"]] = agg.get(v["model"], 0) + v["confidence"]
        if agg:
            tot = sum(agg.values())
            fused = sorted(((k, v / tot) for k, v in agg.items()), key=lambda x: -x[1])
            return {"per_image": per,
                    "fused_model": fused[0][0], "fused_confidence": round(fused[0][1], 4),
                    "fused_alternatives": [{"label": l, "confidence": round(p, 4)} for l, p in fused[1:4]]}
        return {"per_image": per, "fused_model": "Unknown", "fused_confidence": 0.0, "fused_alternatives": []}
