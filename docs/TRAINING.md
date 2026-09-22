# docs — TRAINING: staged transfer learning
A: ImageNet/pretrained (timm) → B: vehicle-domain adapt → C: make → D: model
(focal + class-balanced sampler + hard-negative rounds) → E: body-type →
F: embedding (ArcFace/triplet) → G: hard negatives (`mine_hard_negatives.py`) →
H: night/surveillance fine-tune (augmentation.yaml + real night data) →
I: calibration (temperature scaling, ECE) → J: per-condition validation.

Imbalance: balanced sampler + focal + oversample rare + targeted collection.
Rare (<20 samples): LIMITED-EVIDENCE flag; retrieval/prototype fallback; never
presented as well-supported. New model: taxonomy entry → licensed images →
validate → dedup → index rebuild → fine-tune → old-vs-new eval → versioned publish.
Robustness: no plate/background reliance (masking ablations in evaluation).
