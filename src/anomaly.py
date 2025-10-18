"""Anomaly detection for suspicious label assignments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import utils


def check_labels(
    df: pd.DataFrame,
    *,
    text_column: str = "text",
    label_column: str = "label",
    max_features: int = 5000,
    min_df: int = 2,
    top_k: int = 50,
    threshold: float = 0.2,
    output_path: str | None = None,
) -> pd.DataFrame:
    utils.ensure_columns(df, [text_column, label_column])

    if df[label_column].nunique() < 2:
        utils.log_step("anomaly", note="skip - insufficient class variety")
        utils.mark_ok("anomaly")
        return pd.DataFrame(columns=[text_column, label_column, "anomaly_score"])

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=min_df)
    tfidf = vectorizer.fit_transform(df[text_column])

    label_ids = df[label_column].astype(str).values
    unique_labels = np.unique(label_ids)

    centroids = {}
    for label in unique_labels:
        mask = label_ids == label
        centroid = tfidf[mask].mean(axis=0)
        centroids[label] = np.asarray(centroid).ravel()

    centroid_matrix = np.vstack([centroids[label] for label in unique_labels])
    sim_matrix = cosine_similarity(tfidf, centroid_matrix)

    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    diagonal_scores = np.fromiter((sim_matrix[i, label_to_index[label]] for i, label in enumerate(label_ids)), dtype=float)

    results = df.copy()
    results["anomaly_score"] = diagonal_scores
    results = results.sort_values("anomaly_score").reset_index(drop=True)

    if threshold is not None:
        flagged = results.query("anomaly_score < @threshold")
    else:
        flagged = results.head(top_k)

    if output_path:
        utils.save_csv(output_path, flagged)
        utils.log_step("anomaly_save", rows=len(flagged), path=output_path)

    utils.log_step("anomaly", suspicious=len(flagged), min_score=float(results["anomaly_score"].min()))
    utils.mark_ok("anomaly")
    return flagged


__all__ = ["check_labels"]
