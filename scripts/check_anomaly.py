"""Run anomaly detection on a labeled dataset."""
from __future__ import annotations

import argparse
import pandas as pd

from src import anomaly, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect suspicious labels using TF-IDF similarity")
    parser.add_argument("--data", default="processed/splits/train.csv", help="CSV path to inspect")
    parser.add_argument(
        "--output",
        default="processed/anomaly_suspicious.csv",
        help="Where to save the suspicious rows",
    )
    parser.add_argument("--threshold", type=float, default=0.2, help="Similarity threshold")
    parser.add_argument("--top-k", type=int, default=50, help="Fallback number of rows when threshold is None")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data)
    flagged = anomaly.check_labels(
        df,
        threshold=args.threshold,
        top_k=args.top_k,
        output_path=args.output,
    )

    if flagged.empty:
        utils.log_step("anomaly_review", note="no suspicious samples")
    else:
        print(flagged.head())


if __name__ == "__main__":
    main()
