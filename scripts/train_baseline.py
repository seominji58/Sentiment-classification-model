"""Train only the baseline TF-IDF + Linear SVM model."""
from __future__ import annotations

import argparse
import pandas as pd

from src import baseline, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TF-IDF baseline model")
    parser.add_argument("--train", default="processed/splits/train.csv", help="Path to train CSV")
    parser.add_argument("--valid", default="processed/splits/valid.csv", help="Path to valid CSV")
    parser.add_argument("--config", default="configs/config.json", help="Config path")
    parser.add_argument(
        "--model-path",
        default="processed/models/baseline/model.joblib",
        help="Where to store the trained model",
    )
    parser.add_argument(
        "--metrics-path",
        default="processed/models/baseline/metrics.json",
        help="Where to store evaluation metrics",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    utils.set_seed(args.seed)
    train_df = pd.read_csv(args.train)
    valid_df = pd.read_csv(args.valid)

    baseline.train_and_evaluate(
        train_df,
        valid_df,
        config_path=args.config,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
    )


if __name__ == "__main__":
    main()
