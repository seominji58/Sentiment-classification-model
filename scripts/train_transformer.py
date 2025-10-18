"""Fine-tune the transformer classifier."""
from __future__ import annotations

import argparse
import pandas as pd

from src import transformer, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the transformer model for emotion classification")
    parser.add_argument("--train", default="processed/splits/train.csv", help="Path to train CSV")
    parser.add_argument("--valid", default="processed/splits/valid.csv", help="Path to valid CSV")
    parser.add_argument("--config", default="configs/config.json", help="Config path")
    parser.add_argument(
        "--checkpoint-dir",
        default="processed/models/transformer/checkpoints",
        help="Directory for intermediate checkpoints",
    )
    parser.add_argument(
        "--final-dir",
        default="processed/models/transformer/final",
        help="Directory for the final model",
    )
    parser.add_argument(
        "--metrics-path",
        default="processed/models/transformer/metrics.json",
        help="Where to store evaluation metrics",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    utils.set_seed(args.seed)
    train_df = pd.read_csv(args.train)
    valid_df = pd.read_csv(args.valid)

    transformer.train(
        train_df,
        valid_df,
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        final_dir=args.final_dir,
        metrics_path=args.metrics_path,
    )


if __name__ == "__main__":
    main()
