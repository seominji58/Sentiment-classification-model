"""Text cleaning and preprocessing routines."""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from . import utils

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def clean(
    df: pd.DataFrame,
    *,
    text_column: str = "text",
    label_column: str = "label",
    min_text_len: int = 1,
    drop_duplicates: bool = True,
    keep_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a cleaned dataframe with consistent text/label columns."""
    utils.ensure_columns(df, [text_column, label_column])

    work_df = df.copy()
    work_df = work_df.dropna(subset=[text_column, label_column])

    work_df[text_column] = work_df[text_column].astype(str).map(_normalize_text)
    work_df[label_column] = work_df[label_column].astype(str).str.strip()

    mask_valid = work_df[text_column].str.len() >= min_text_len
    mask_valid &= work_df[label_column].ne("")
    cleaned_df = work_df.loc[mask_valid].copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates(subset=[text_column, label_column])

    if keep_columns is not None:
        missing = [col for col in keep_columns if col not in cleaned_df.columns]
        if missing:
            utils.fail(f"전처리 keep_columns 결측: {missing}")
        cleaned_df = cleaned_df[list(keep_columns)]
    else:
        cleaned_df = cleaned_df[[text_column, label_column]]

    utils.must_not_empty(cleaned_df, "cleaned_df")
    utils.log_step("preprocess", rows=len(cleaned_df))
    utils.mark_ok("preprocess")
    return cleaned_df.reset_index(drop=True)


__all__ = ["clean"]
