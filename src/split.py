"""Train/validation split utilities."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from . import utils


def make_splits(
    df: pd.DataFrame,
    test_size: float,
    *,
    label_column: str = "label",
    random_state: int = 42,
    train_path: str | None = None,
    valid_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    utils.require("label_map")
    utils.ensure_columns(df, [label_column])

    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_column],
        random_state=random_state,
    )

    for subset, name, path in [(train_df, "train", train_path), (valid_df, "valid", valid_path)]:
        utils.must_not_empty(subset, f"{name}_df")
        subset = subset.reset_index(drop=True)
        if path:
            utils.save_csv(path, subset)
            utils.log_step(f"split_{name}_save", rows=len(subset), path=path)

    utils.log_step("split", train_rows=len(train_df), valid_rows=len(valid_df))
    utils.mark_ok("split")
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


__all__ = ["make_splits"]
