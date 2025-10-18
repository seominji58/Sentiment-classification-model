"""Common evaluation helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    *,
    labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_dict = {
        "labels": labels if labels else sorted(set(y_true) | set(y_pred)),
        "matrix": cm.tolist(),
    }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": cm_dict,
    }


def save_metrics(metrics: Dict[str, object], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


__all__ = ["compute_metrics", "save_metrics"]
