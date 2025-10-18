"""Label mapping utilities."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from . import utils


def apply_mapping(
    df: pd.DataFrame,
    config_path: str,
    *,
    label_column: str = "label",
    mapped_column: str | None = None,
    drop_unmapped: bool = True,
) -> pd.DataFrame:
    utils.require("preprocess")
    utils.ensure_columns(df, [label_column])

    config = utils.Config.load(config_path)
    label_mapping: Dict[str, str] = config.label_mapping()

    work_df = df.copy()
    target_column = mapped_column or label_column
    work_df[target_column] = work_df[label_column].map(label_mapping)

    if drop_unmapped:
        before = len(work_df)
        work_df = work_df.dropna(subset=[target_column])
        dropped = before - len(work_df)
        if dropped:
            utils.log_step("label_map", dropped=dropped, note="unmapped labels dropped")
    else:
        work_df[target_column] = work_df[target_column].fillna(work_df[label_column])

    utils.must_not_empty(work_df, "label_mapped_df")
    utils.log_step("label_map", rows=len(work_df), unique_labels=work_df[target_column].nunique())
    utils.mark_ok("label_map")
    return work_df.reset_index(drop=True)


__all__ = ["apply_mapping"]
