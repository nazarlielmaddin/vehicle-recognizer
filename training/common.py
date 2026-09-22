"""Shared training utilities: identity-aware splits, balanced sampling, losses."""
from __future__ import annotations
import random
from collections import Counter
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

def identity_aware_split(ids: list[str], ratios=(0.7, 0.15, 0.15), seed=42):
    """Split by vehicle identity / source family so near-duplicates never leak."""
    rng = random.Random(seed)
    uniq = sorted(set(ids))
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
    tr, va, te = set(uniq[:n_tr]), set(uniq[n_tr:n_tr+n_va]), set(uniq[n_tr+n_va:])
    return tr, va, te

def balanced_sampler(labels: list[int]) -> WeightedRandomSampler:
    cnt = Counter(labels)
    w = [1.0 / cnt[l] for l in labels]
    return WeightedRandomSampler(w, len(w), replacement=True)

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()

def arcface_logits(feat: torch.Tensor, weight: torch.Tensor, s=30.0, m=0.5):
    feat = torch.nn.functional.normalize(feat)
    w = torch.nn.functional.normalize(weight)
    cos = feat @ w.T
    # additive angular margin (simplified)
    return s * (cos - m)
