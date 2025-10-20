"""Clean and consolidate the noisy training dataset."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_config(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def normalise_label(
    code: str,
    label_mapping: Dict[str, str],
    major_base_map: Dict[str, str],
) -> str | None:
    if not code:
        return None
    major = label_mapping.get(code, code)
    major = major_base_map.get(major, major)
    return major


def build_text(row: Dict[str, str]) -> str:
    candidates: List[str] = []
    for key in (
        "text",
        "사람문장1",
        "사람문장2",
        "사람문장3",
        "talk.content.HS01",
        "talk.content.HS02",
        "talk.content.HS03",
    ):
        value = row.get(key, "")
        if value:
            value = value.strip()
            if value:
                candidates.append(value)
    unique: List[str] = []
    seen = set()
    for part in candidates:
        if part not in seen:
            seen.add(part)
            unique.append(part)
    return " ".join(unique)


def keyword_counts(text: str, keyword_rules: Dict[str, Iterable[str]]) -> Dict[str, int]:
    lowered = text.lower()
    counts: Dict[str, int] = {}
    for label, keywords in keyword_rules.items():
        count = 0
        for keyword in keywords:
            kw = str(keyword).strip().lower()
            if kw and kw in lowered:
                count += 1
        counts[label] = count
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare clean training data from noisy CSV")
    parser.add_argument(
        "--input",
        default="processed/splits/train.csv",
        help="Path to the raw training CSV",
    )
    parser.add_argument(
        "--output",
        default="processed/splits/train_clean.csv",
        help="Where to write the cleaned training CSV",
    )
    parser.add_argument(
        "--flagged",
        default="processed/splits/train_flagged.csv",
        help="Where to write flagged samples for manual review",
    )
    parser.add_argument(
        "--config",
        default="configs/config.json",
        help="Config file with label mappings and keyword rules",
    )
    parser.add_argument("--min-length", type=int, default=10, help="Minimum text length to keep")
    parser.add_argument(
        "--keep-flagged",
        action="store_true",
        help="If set, keep flagged rows in the cleaned output as well",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(Path(args.config))
    label_mapping = config.get("label_mapping", {})
    major_base_map = config.get("emotion_hierarchy", {}).get("major_base_map", {})
    keyword_rules = config.get("emotion_hierarchy", {}).get("keyword_rules", {})

    input_path = Path(args.input)
    output_path = Path(args.output)
    flagged_path = Path(args.flagged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    flagged_path.parent.mkdir(parents=True, exist_ok=True)

    seen_texts: set[str] = set()
    kept = 0
    flagged_count = 0
    processed_rows = 0
    reasons_counter: Counter[str] = Counter()

    with input_path.open(encoding="utf-8", newline="") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout, flagged_path.open("w", encoding="utf-8", newline="") as fflag:
        reader = csv.DictReader(fin)
        main_writer = csv.DictWriter(fout, fieldnames=["text", "label"])
        flag_writer = csv.DictWriter(
            fflag,
            fieldnames=[
                "text",
                "label",
                "original_label",
                "reasons",
            ],
        )
        main_writer.writeheader()
        flag_writer.writeheader()

        for row in reader:
            processed_rows += 1
            original_label = row.get("label") or row.get("감정_대분류")
            major_label = normalise_label(original_label or "", label_mapping, major_base_map)
            if major_label is None:
                reasons_counter["missing_label"] += 1
                continue

            text = build_text(row)
            if not text:
                reasons_counter["empty_text"] += 1
                continue

            normalized_text = text.strip()
            if len(normalized_text) < args.min_length:
                reasons_counter["too_short"] += 1
                flag_writer.writerow(
                    {
                        "text": normalized_text,
                        "label": major_label,
                        "original_label": original_label,
                        "reasons": "short_text",
                    }
                )
                flagged_count += 1
                if args.keep_flagged:
                    main_writer.writerow({"text": normalized_text, "label": major_label})
                continue

            if normalized_text in seen_texts:
                reasons_counter["duplicate"] += 1
                flag_writer.writerow(
                    {
                        "text": normalized_text,
                        "label": major_label,
                        "original_label": original_label,
                        "reasons": "duplicate_text",
                    }
                )
                flagged_count += 1
                if args.keep_flagged:
                    main_writer.writerow({"text": normalized_text, "label": major_label})
                continue

            seen_texts.add(normalized_text)

            reasons: List[str] = []
            if keyword_rules:
                counts = keyword_counts(normalized_text, keyword_rules)
                primary = counts.get(major_label, 0)
                other_labels: List[Tuple[str, int]] = [
                    (lbl, cnt) for lbl, cnt in counts.items() if lbl != major_label
                ]
                if other_labels:
                    top_label, top_count = max(other_labels, key=lambda item: item[1])
                    if top_count > primary and top_count > 0:
                        reasons.append(f"keyword_conflict:{top_label}")

            if reasons:
                reasons_counter.update(reasons)
                flag_writer.writerow(
                    {
                        "text": normalized_text,
                        "label": major_label,
                        "original_label": original_label,
                        "reasons": ",".join(reasons),
                    }
                )
                flagged_count += 1
                if args.keep_flagged:
                    main_writer.writerow({"text": normalized_text, "label": major_label})
                continue

            main_writer.writerow({"text": normalized_text, "label": major_label})
            kept += 1

    print(f"Input rows processed: {processed_rows}")
    print(f"Kept rows: {kept}")
    print(f"Flagged rows for review: {flagged_count}")
    if reasons_counter:
        print("Flag reasons summary:")
        for reason, count in reasons_counter.most_common():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
