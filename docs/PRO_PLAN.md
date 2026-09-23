# PRO PLAN — state radar integration readiness

## 1. Dil və kitabxana seçimi (qərar)

**Dil: Python.** Səbəb: kompüter-görmənin bütün ekosistemi Python-dadır
(PyTorch, Ultralytics, timm, ONNX Runtime, scikit-learn). Alternativ yoxdur —
C++ yalnız final deploymanın optimizasiyasında (TensorRT/ONNX) istifadə olunur,
təlim və pipeline Python-da qalır. Radar inteqrasiyası REST API ilə olur
(`api/main.py`), ona görə istehlakçı dil asılı deyil.

**Kitabxanalar:**
| Rol | Seçim | Səbəb |
|---|---|---|
| Detektor | Ultralytics YOLO11 (m → s, CPU üçün) | İşləyir (det 69%), ONNX export hazır |
| Klassifikator | timm `efficientnet_b0` | 4M param, CPU-da ~100ms, transfer-learning üçün ideal |
| Metrik/embedding | timm backbone + ArcFace | Nadir siniflər üçün retrieval |
| Kalibrasiya | scikit-learn (temperature scaling, ECE) | Standart |
| Deploy format | ONNX Runtime | CPU serverdə stabil, GPU-suz işləyir |
| Data | turbo.az kollektoru (hazır, test edilib) | Regional markalar yalnız burdadır |

**Niyə hazır modellər yararsızdır (araşdırma ilə sübut):**
- VMMRdb (8,949 sinif): Changan/BYD/Haval/Lada = 0 sinif. Yoxlanılıb.
- CompCars çəkiləri: heç kim paylaşmayıb + dataset imzalı müqavilə istəyir.
- Stanford-əsaslı: 196 Qərb modeli.
- Nəticə: Changan-ı bilən endirilə bilən klassifikator **mövcud deyil**.
  Yeganə yol regional datada öz təlimimizdir.

## 2. Dərhal fixlər (bu gün — artıq kodda)

1. ✅ YOLO-sinif qapısı: `car` kropuna TIR/avtobus modeli təklif oluna bilməz
   (`_gate_models` + `_allowed_bodies`). SITRAK absurdı kökündən bitdi.
2. ✅ Viewpoint: UNKNOWN yalnız zəif sübutda (0.25 astanası).
3. ✅ Alternativlər: eyni-marka qarışıqlıqları birinci.
4. ✅ Loqo kanalı (`badge.py`) — CLIP-lə tavan 0.13, öyrədilmiş loqo
   klassifikatoru ilə əvəzlənəcək (Faza 2).

## 3. Təlim yol xəritəsi (dövlət qəbulu üçün)

**Faza A — Data (indi işləyir):** turbo.az kolleksiya, 20+ marka × 30+ şəkil,
hər şəkilə provenance. Dedup (pHash) + QC + identity-aware split.
**Faza B — Marka klassifikatoru (ən yüksək təsir):** EfficientNet-B0,
focal loss + balanced sampler, CPU-da ~1 saat. Hədəf: marka top-1 ≥ 85%.
**Faza C — Model klassifikatoru (top markalar):** Changan/BYD/Toyota/Lada
üçün ayrı incə-tənzimləmə + hard-negative mining (UNI-K vs UNI-T kimi).
**Faza D — Kalibrasiya + UNKNOWN:** temperature scaling, ECE < 0.05,
qəbul qadaları (aşağıda).
**Faza E — Qəbul testi (sizin radar kadrlarınızla):** 200+ etiketli radar
şəkli, gündüz/gece, ön/arxa. Keçid şərti:
- marka top-1 ≥ 90%, model top-1 ≥ 75%, top-3 ≥ 90%
- TIR/yük absurdı: 0 (qapı ilə zəmanət)
- UNKNOWN-dən kənar səhv: < 5% (əmin səhvdən imtina üstünlüyü)

## 4. "Bir daha səhv" zəmanətləri (arxitektura)

- Qapılar (YOLO-sinif → kuzov) səhv kateqoriyanı **strukturla** mümkünsüz edir.
- Kalibrasiya olunmamış güvən heç vaxt göstərilmir.
- Sübut zəifdirsə UNKNOWN — səhv markadan üstündür.
- Hər proqnozda `abstain_reasons` + `evidence_source` audit izi var.
- Model registry: köhnə prod modeli heç vaxt silinmir, rollback hazırdır.
