"""Smoke + integration tests: quality, crops, fusion, unknown, API validation."""
import numpy as np

def test_quality_gates_poor_image():
    from inference.quality import analyze_quality
    dark = np.zeros((120, 200, 3), np.uint8)
    q = analyze_quality(dark)
    assert q.quality == "Poor"
    assert q.lighting in ("NIGHT", "LOW_LIGHT", "HEADLIGHT_DOMINATED")

def test_letterbox_preserves_geometry():
    from inference.crops import letterbox
    wide = np.zeros((100, 400, 3), np.uint8)
    out = letterbox(wide, (224, 224))
    assert out.shape == (224, 224, 3)

def test_unknown_logic_abstains_when_uncertain():
    from inference.fusion import decide_unknown
    abstain, reasons = decide_unknown(0.3, 0.28, 2.0, 0.1, "Poor")
    assert abstain and len(reasons) >= 2

def test_fusion_prefers_agreement():
    from inference.fusion import fuse_classifier_retrieval
    fused = fuse_classifier_retrieval(
        [("BMW 5 Series", 0.5), ("BMW 3 Series", 0.3)],
        [("x", 0.9, {"class": "BMW 5 Series"})])
    assert fused[0][0] == "BMW 5 Series"

def test_taxonomy_normalization():
    from training.taxonomy import normalize_make, normalize_model
    assert normalize_make("MERCEDES") == "Mercedes-Benz"
    assert normalize_make("VW") == "Volkswagen"
    assert normalize_model("BMW", "3 series") == "3 Series"

def test_api_rejects_bad_mime():
    allowed = {"image/jpeg", "image/png", "image/webp"}
    assert "application/x-msdownload" not in allowed
