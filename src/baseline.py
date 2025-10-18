"""Baseline TF-IDF + LinearSVC model training."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from . import evaluation, utils


def _build_pipeline(config: Dict[str, float | int | tuple]) -> Pipeline:
    ngram_range = tuple(config.get("ngram_range", (1, 2)))
    min_df = config.get("min_df", 5)
    max_df = config.get("max_df", 0.95)
    alpha = 1.0 / float(config.get("C", 1.0))

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=max_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="hinge",
                    alpha=alpha,
                    penalty="l2",
                    random_state=42,
                    max_iter=1000,
                    tol=1e-3,
                ),
            ),
        ]
    )


def train_and_evaluate(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    text_column: str = "text",
    label_column: str = "label",
    config_path: str = "configs/config.json",
    model_path: str = "processed/models/baseline/model.joblib",
    metrics_path: str = "processed/models/baseline/metrics.json",
) -> Dict[str, float]:
    utils.ensure_columns(train_df, [text_column, label_column])
    utils.ensure_columns(valid_df, [text_column, label_column])

    labels = sorted(
        set(train_df[label_column].unique()) | set(valid_df[label_column].unique())
    )

    config = utils.Config.load(config_path)
    baseline_cfg = config.get("baseline", {})

    pipeline = _build_pipeline(baseline_cfg)
    pipeline.fit(train_df[text_column], train_df[label_column])

    preds = pipeline.predict(valid_df[text_column])
    gold = valid_df[label_column]

    metrics = evaluation.compute_metrics(gold, preds, labels=labels)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, model_path)

    evaluation.save_metrics(metrics, metrics_path)

    utils.log_step(
        "baseline",
        accuracy=round(metrics["accuracy"], 4),
        macro_f1=round(metrics["macro_f1"], 4),
    )
    utils.mark_ok("baseline")
    return metrics


__all__ = ["train_and_evaluate"]
