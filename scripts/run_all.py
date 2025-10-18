"""End-to-end pipeline runner for the emotion classification project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import anomaly, baseline, data_io, label_map, preprocess, split, transformer, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Saegim emotion classification pipeline")
    parser.add_argument("--src", default="data/merged.csv", help="Path to the merged raw dataset")
    parser.add_argument("--config", default="configs/config.json", help="Path to pipeline config JSON")
    parser.add_argument(
        "--processed", default="processed", help="Directory to store processed artifacts"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-anomaly", action="store_true", help="Skip anomaly detection stage")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip transformer fine-tuning stage")
    return parser.parse_args()


def save_label_mapping(config: utils.Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.label_mapping(), f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    utils.set_seed(args.seed)

    processed_dir = Path(args.processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    merged_df = data_io.load_merged(args.src)

    # 2. Preprocess text/label columns
    cleaned_df = preprocess.clean(merged_df)
    cleaned_path = processed_dir / "cleaned.csv"
    utils.save_csv(cleaned_path, cleaned_df)

    # 3. Apply label mapping
    mapped_df = label_map.apply_mapping(cleaned_df, args.config)
    final_path = processed_dir / "final.csv"
    utils.save_csv(final_path, mapped_df)

    # Save label mapping for traceability
    config = utils.Config.load(args.config)
    mapping_path = processed_dir / "label_mapping.json"
    save_label_mapping(config, mapping_path)

    # 4. Stratified train/valid split
    splits_dir = processed_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"
    valid_path = splits_dir / "valid.csv"
    train_df, valid_df = split.make_splits(
        mapped_df,
        test_size=config.get("split", {}).get("test_size", 0.1),
        random_state=config.get("split", {}).get("random_state", 42),
        train_path=str(train_path),
        valid_path=str(valid_path),
    )

    # 5. Optional anomaly detection
    if not args.skip_anomaly:
        anomaly_path = processed_dir / "anomaly_suspicious.csv"
        anomaly.check_labels(train_df, output_path=str(anomaly_path))
    else:
        utils.log_step("anomaly", note="skipped by flag")

    # 6. Baseline model training
    baseline_metrics = baseline.train_and_evaluate(
        train_df,
        valid_df,
        config_path=args.config,
        model_path=str(processed_dir / "models" / "baseline" / "model.joblib"),
        metrics_path=str(processed_dir / "models" / "baseline" / "metrics.json"),
    )

    # 7. Transformer fine-tuning
    if not args.skip_transformer:
        transformer.train(
            train_df,
            valid_df,
            config_path=args.config,
            checkpoint_dir=str(processed_dir / "models" / "transformer" / "checkpoints"),
            final_dir=str(processed_dir / "models" / "transformer" / "final"),
            metrics_path=str(processed_dir / "models" / "transformer" / "metrics.json"),
        )
    else:
        utils.log_step("transformer", note="skipped by flag")

    utils.log_step("pipeline_complete", baseline_macro_f1=round(baseline_metrics["macro_f1"], 4))


if __name__ == "__main__":
    main()
