"""Utility helpers shared across the emotion classification pipeline."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

_STATE: Dict[str, bool] = {}


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def mark_ok(step: str) -> None:
    _STATE[step] = True


def require(step: str, msg: str | None = None) -> None:
    if not _STATE.get(step):
        raise RuntimeError(msg or f"이전 단계 미실행: {step}")


def fail(msg: str) -> None:
    raise RuntimeError(f"⛔ STOP: {msg}")


def must_have_cols(df, cols: Iterable[str], name: str = "df") -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        fail(f"{name} 결측 컬럼: {sorted(missing)}")


def must_not_empty(df, name: str = "df") -> None:
    if len(df) == 0:
        fail(f"{name} 비어있음")


def save_csv(path: str | Path, df) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if not path.exists() or path.stat().st_size == 0:
        fail(f"파일 저장 실패: {path}")


def log_step(name: str, **stats: Any) -> None:
    info = " | ".join(f"{k}={v}" for k, v in stats.items())
    print(f"✅ [{name}] 완료" + (f" | {info}" if info else ""))


@dataclass
class Config:
    raw: Dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(raw=data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def label_mapping(self) -> Dict[str, str]:
        mapping = self.raw.get("label_mapping")
        if not mapping:
            fail("config.json에 label_mapping 미정의")
        return mapping


def ensure_file(path: str | Path, message: str | None = None) -> None:
    if not Path(path).exists():
        fail(message or f"필수 파일 누락: {path}")


def ensure_columns(df, required: Iterable[str]) -> None:
    must_have_cols(df, required)
    must_not_empty(df)


def to_device(obj, device: str):
    try:
        import torch

        if isinstance(obj, torch.nn.Module):
            return obj.to(device)
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
    except ImportError:
        pass
    return obj


__all__ = [
    "set_seed",
    "mark_ok",
    "require",
    "fail",
    "must_have_cols",
    "must_not_empty",
    "save_csv",
    "log_step",
    "Config",
    "ensure_file",
    "ensure_columns",
    "to_device",
]
