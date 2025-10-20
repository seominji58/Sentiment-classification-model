"""Post-processing utilities for sentiment classification outputs."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def _normalize_keywords(rules: Dict[str, Iterable[str]]) -> Dict[str, Sequence[str]]:
    normalized: Dict[str, Sequence[str]] = {}
    for label, words in rules.items():
        normalized[label] = tuple(str(word).strip() for word in words if str(word).strip())
    return normalized


def boost_with_keywords(
    texts: Sequence[str],
    probs: np.ndarray,
    label_names: Sequence[str],
    keyword_rules: Dict[str, Iterable[str]],
    *,
    boost: float = 0.1,
) -> np.ndarray:
    """Boost class probabilities when the text contains configured keywords.

    Args:
        texts: List of raw text samples.
        probs: Probability matrix shaped (num_samples, num_labels).
        label_names: Label names aligned with probability columns.
        keyword_rules: Mapping of label -> iterable of keywords.
        boost: Additive boost applied to matching label probabilities.

    Returns:
        Adjusted probability matrix (copy) with boosts applied.
    """

    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array")

    normalized_rules = _normalize_keywords(keyword_rules)
    label_index = {label: idx for idx, label in enumerate(label_names)}
    adjusted = probs.copy()

    for row_idx, text in enumerate(texts):
        lowered = str(text).lower()
        for label, keywords in normalized_rules.items():
            col = label_index.get(label)
            if col is None or not keywords:
                continue
            if any(keyword.lower() in lowered for keyword in keywords):
                adjusted[row_idx, col] += boost

    # Re-normalize so that each row sums to 1 (avoid zero division case).
    row_sums = adjusted.sum(axis=1, keepdims=True)
    # To prevent division by zero, fall back to original probabilities.
    zero_mask = row_sums.squeeze(axis=1) == 0
    if np.any(zero_mask):
        adjusted[zero_mask] = probs[zero_mask]
        row_sums = adjusted.sum(axis=1, keepdims=True)
    adjusted /= row_sums
    return adjusted


__all__ = ["boost_with_keywords"]
