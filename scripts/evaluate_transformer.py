"""Evaluate a trained transformer model on a CSV split and report per-class metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import utils
from src.postprocess import boost_with_keywords


def select_with_threshold(probs: np.ndarray, thresholds: np.ndarray | None) -> np.ndarray:
    if thresholds is None:
        return probs.argmax(axis=-1)

    mask = probs >= thresholds
    masked = np.where(mask, probs, np.zeros_like(probs))
    fallback = probs.argmax(axis=-1)
    none_mask = masked.sum(axis=1) == 0
    masked[none_mask, :] = probs[none_mask, :]
    return masked.argmax(axis=-1)


def load_thresholds(path: Path, id2label: Dict[int, str]) -> torch.Tensor | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    thresholds: List[float] = []
    for idx in range(len(id2label)):
        label = id2label[idx]
        thresholds.append(float(data.get(label, 0.0)))
    return torch.tensor(thresholds, dtype=torch.float32)


def predict(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    *,
    thresholds: torch.Tensor | None = None,
    batch_size: int = 32,
) -> List[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    if isinstance(id2label, dict):
        id2label = {int(k): v for k, v in id2label.items()}

    predictions: List[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            if thresholds is not None:
                probs = torch.softmax(logits, dim=-1)
                mask = probs >= thresholds.to(probs.device)
                masked = torch.where(mask, probs, torch.zeros_like(probs))
                chosen = masked.argmax(dim=-1)
                fallback = probs.argmax(dim=-1)
                none_mask = masked.max(dim=-1).values == 0
                chosen[none_mask] = fallback[none_mask]
            else:
                chosen = logits.argmax(dim=-1)
        predictions.extend(id2label[int(i)] for i in chosen.cpu())
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate transformer classifier")
    parser.add_argument(
        "--model-dir",
        default="processed/models/transformer/final",
        help="Directory containing the trained model and tokenizer",
    )
    parser.add_argument(
        "--split",
        default="processed/splits/valid.csv",
        help="CSV file with text and label columns",
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-report", help="Optional path to save the classification report JSON")
    parser.add_argument("--config", default="configs/config.json", help="Config file with keyword rules")
    parser.add_argument(
        "--keyword-boost",
        type=float,
        default=0.1,
        help="Boost value added when keyword rules match (set 0 to disable)",
    )
    args = parser.parse_args()

    utils.set_seed(42)

    split_path = Path(args.split)
    if not split_path.exists():
        utils.fail(f"Split file not found: {split_path}")

    df = pd.read_csv(split_path)
    utils.ensure_columns(df, [args.text_column, args.label_column])

    texts = df[args.text_column].astype(str).tolist()
    labels = df[args.label_column].astype(str).tolist()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    thresholds = load_thresholds(model_dir / "probability_thresholds.json", model.config.id2label)
    thresholds_np = thresholds.cpu().numpy() if thresholds is not None else None
    keywords = None
    if args.keyword_boost > 0:
        config_path = Path(args.config)
        if config_path.exists():
            config = utils.Config.load(config_path)
            keyword_section = config.get("keyword_rules", {})
            if keyword_section:
                keywords = keyword_section
        else:
            print(f"⚠️ Config file not found: {config_path}. Keyword boost skipped.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    if isinstance(id2label, dict):
        id2label = {int(k): v for k, v in id2label.items()}

    all_probs = []
    predictions: List[str] = []
    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start : start + args.batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        all_probs.append(probs)
        chosen_idx = select_with_threshold(probs, thresholds_np)
        predictions.extend(id2label[int(idx)] for idx in chosen_idx)

    all_probs_np = np.vstack(all_probs)
    if keywords:
        label_order = [id2label[i] for i in range(len(id2label))]
        boosted = boost_with_keywords(texts, all_probs_np, label_order, keywords, boost=args.keyword_boost)
        boosted_idx = select_with_threshold(boosted, thresholds_np)
        predictions = [label_order[int(idx)] for idx in boosted_idx]

    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    labels_order = sorted(set(labels))
    cm = confusion_matrix(labels, predictions, labels=labels_order)

    print("Per-class metrics:")
    for label in labels_order:
        metrics = report[label]
        print(
            f"  {label}: precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}"
        )
    print(f"\nOverall accuracy: {report['accuracy']:.3f}")
    print(
        f"Macro F1: {report['macro avg']['f1-score']:.3f}, "
        f"Weighted F1: {report['weighted avg']['f1-score']:.3f}"
    )

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels_order, columns=labels_order))

    if args.save_report:
        payload = {
            "report": report,
            "labels": labels_order,
            "confusion_matrix": cm.tolist(),
        }
        Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report to {args.save_report}")


if __name__ == "__main__":
    main()
