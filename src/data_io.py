"""Data loading helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from . import utils


def load_csv(path: str | Path, *, expected_cols: Optional[set[str]] = None) -> pd.DataFrame:
    utils.ensure_file(path)
    df = pd.read_csv(path)
    if expected_cols:
        utils.must_have_cols(df, expected_cols)
    utils.must_not_empty(df)
    return df


def load_merged(path: str | Path) -> pd.DataFrame:
    df = load_csv(path)
    utils.log_step("load_merged", rows=len(df), path=Path(path).resolve())
    utils.mark_ok("load_merged")
    return df


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    utils.save_csv(path, df)


__all__ = ["load_csv", "load_merged", "save_dataframe"]
