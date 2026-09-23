"""Anti-forgetting + anti-fantasy regression (QAYDA 17).
Pure-logic tests: no weights, no network, no images needed.
"""
import numpy as np
from inference.pipeline import VehiclePipeline

T47 = ["Changan", "Hyundai", "Kia", "Toyota"]
FULL = ["Changan", "Hyundai", "Kia", "Toyota", "Subaru", "NIO"]

def _vec(order, peaks):
    v = np.full(len(order), 0.01)
    for label, p in peaks.items():
        v[order.index(label)] = p
    return v / v.sum()

def test_trained_confident_wins():
    top_t = [("Changan", 0.8), ("Hyundai", 0.1)]
    pt = _vec(T47, {"Changan": 0.8, "Hyundai": 0.1})
    zt = [("NIO", 0.5)]
    pz = _vec(FULL, {"NIO": 0.5})
    top, hybrid = VehiclePipeline._blend_make_distributions(top_t, pt, T47, zt, pz, FULL)
    assert not hybrid and top[0][0] == "Changan", top

def test_subaru_backstop():
    """User scenario: trained head never saw Subaru (flat), zs knows it."""
    top_t = [("Hyundai", 0.15), ("Kia", 0.12)]
    pt = _vec(T47, {"Hyundai": 0.15, "Kia": 0.12, "Changan": 0.1, "Toyota": 0.1})
    zt = [("Subaru", 0.6), ("Toyota", 0.2)]
    pz = _vec(FULL, {"Subaru": 0.6, "Toyota": 0.2})
    top, hybrid = VehiclePipeline._blend_make_distributions(top_t, pt, T47, zt, pz, FULL)
    assert hybrid and top[0][0] == "Subaru", top

def test_fantasy_forbidden():
    fused = [("NIO ES6", 0.27), ("Tesla Model 3", 0.2)]
    out, note = VehiclePipeline._constrain_to_make(fused, "Changan", 0.10)
    assert out == [] and "not Changan" in note, (out, note)

def test_same_make_kept():
    fused = [("NIO ES6", 0.27), ("Changan UNI-K", 0.20), ("Tesla Model 3", 0.15)]
    out, note = VehiclePipeline._constrain_to_make(fused, "Changan", 0.10)
    assert out[0][0] == "Changan UNI-K" and note == "", (out, note)

def test_unknown_make_passthrough():
    fused = [("NIO ES6", 0.27)]
    out, note = VehiclePipeline._constrain_to_make(fused, "Unknown", 0.10)
    assert out == fused and note == ""
