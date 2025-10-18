#!/usr/bin/env python3
"""Generate label mapping using rule-based aggregation over data splits."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Base mapping from native major emotion to final 5-way target.
MAJOR_BASE_MAP: Dict[str, str] = {
    "기쁨": "행복",
    "불안": "불안",
    "분노": "분노",
    "슬픔": "슬픔",
    "당황": "불안",
    "상처": "슬픔",
}

# Minor-category heuristics.
PEACEFUL_MINOR = {
    "느긋",
    "안도",
    "편안한",
    "신뢰하는",
}
EMBARRASSED_SAD_MINOR = {
    "고립된",
    "남의 시선을 의식하는",
    "부끄러운",
    "열등감",
    "외로운",
    "죄책감의",
    "한심한",
    "혐오스러운",
}
HURT_UNREST_MINOR = {
    "억울한",
    "질투하는",
    "충격 받은",
}

THRESHOLDS = {
    "peaceful_ratio": 0.4,
    "embarrassed_to_sad_ratio": 0.4,
    "hurt_to_unrest_ratio": 0.4,
}

KEYWORD_RULES = {
    "행복": [
        "행복",
        "기쁨",
        "즐거움",
        "신남",
        "감사",
        "만족",
        "기대",
        "좋다",
    ],
    "평온": [
        "차분",
        "평온",
        "안정",
        "편안",
        "느긋",
        "휴식",
        "힐링",
        "쉼",
        "여유",
    ],
    "슬픔": [
        "슬픔",
        "우울",
        "낙담",
        "눈물",
        "실망",
        "외로움",
        "후회",
        "상처",
    ],
    "불안": [
        "불안",
        "초조",
        "긴장",
        "걱정",
        "심장",
        "긴박",
        "당황",
        "혼란",
    ],
    "분노": [
        "화남",
        "분노",
        "짜증",
        "열받",
        "억울",
        "답답",
        "성질",
        "격노",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build label mapping from CSV splits.")
    parser.add_argument(
        "--train",
        default="data/splits/train.csv",
        help="Path to training split CSV (UTF-8).",
    )
    parser.add_argument(
        "--valid",
        default="data/splits/valid.csv",
        help="Path to validation split CSV (UTF-8).",
    )
    parser.add_argument(
        "--config",
        default="configs/config.json",
        help="Config JSON to update with generated mapping.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column containing SV emotion codes.",
    )
    parser.add_argument(
        "--major-column",
        default="감정_대분류",
        help="Column name for native major emotion class.",
    )
    parser.add_argument(
        "--minor-column",
        default="감정_소분류",
        help="Column name for native minor emotion class.",
    )
    return parser.parse_args()


def most_common(counter: Counter[str]) -> Tuple[str, int]:
    if not counter:
        raise ValueError("Counter is empty")
    # Deterministic tie-breaking: value desc, label asc.
    return max(counter.items(), key=lambda item: (item[1], item[0]))


def aggregate_counts(
    paths: Iterable[Path],
    label_column: str,
    major_column: str,
    minor_column: str,
) -> Tuple[Dict[str, Counter[str]], Dict[Tuple[str, str], Counter[str]]]:
    major_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    minor_counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)

    for path in paths:
        with path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row[label_column].strip()
                major = row[major_column].strip()
                minor = row[minor_column].strip()

                major_counts[label][major] += 1
                minor_counts[(label, major)][minor] += 1

    return major_counts, minor_counts


def choose_final_label(major: str, minor_counter: Counter[str]) -> str:
    base = MAJOR_BASE_MAP.get(major)
    if base is None:
        raise KeyError(f"Unsupported major label: {major}")

    total = sum(minor_counter.values())
    if total == 0:
        return base

    if major == "기쁨":
        peaceful_hits = sum(minor_counter[m] for m in PEACEFUL_MINOR if m in minor_counter)
        ratio = peaceful_hits / total
        if ratio >= THRESHOLDS["peaceful_ratio"]:
            return "평온"
        return "행복"

    if major == "당황":
        sad_hits = sum(minor_counter[m] for m in EMBARRASSED_SAD_MINOR if m in minor_counter)
        ratio = sad_hits / total
        if ratio >= THRESHOLDS["embarrassed_to_sad_ratio"]:
            return "슬픔"
        return base

    if major == "상처":
        unrest_hits = sum(minor_counter[m] for m in HURT_UNREST_MINOR if m in minor_counter)
        ratio = unrest_hits / total
        if ratio >= THRESHOLDS["hurt_to_unrest_ratio"]:
            return "불안"
        return base

    return base


def build_mapping(
    major_counts: Dict[str, Counter[str]],
    minor_counts: Dict[Tuple[str, str], Counter[str]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    code_to_major: Dict[str, str] = {}
    code_to_final: Dict[str, str] = {}

    for code, counter in major_counts.items():
        major, _ = most_common(counter)
        code_to_major[code] = major
        minor_counter = minor_counts.get((code, major), Counter())
        code_to_final[code] = choose_final_label(major, minor_counter)

    return code_to_major, code_to_final


def main() -> None:
    args = parse_args()

    train_path = Path(args.train)
    valid_path = Path(args.valid)
    config_path = Path(args.config)

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError("Split CSV 파일을 찾을 수 없습니다.")
    if not config_path.exists():
        raise FileNotFoundError("config.json 파일이 필요합니다.")

    major_counts, minor_counts = aggregate_counts(
        [train_path, valid_path], args.label_column, args.major_column, args.minor_column
    )
    code_to_major, code_to_final = build_mapping(major_counts, minor_counts)

    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    config["label_mapping"] = dict(sorted(code_to_final.items()))
    config["emotion_hierarchy"] = {
        "code_to_major": dict(sorted(code_to_major.items())),
        "major_base_map": MAJOR_BASE_MAP,
        "minor_rules": {
            "기쁨": {
                "peaceful_minor": sorted(PEACEFUL_MINOR),
                "ratio_threshold": THRESHOLDS["peaceful_ratio"],
            },
            "당황": {
                "sad_minor": sorted(EMBARRASSED_SAD_MINOR),
                "ratio_threshold": THRESHOLDS["embarrassed_to_sad_ratio"],
            },
            "상처": {
                "unrest_minor": sorted(HURT_UNREST_MINOR),
                "ratio_threshold": THRESHOLDS["hurt_to_unrest_ratio"],
            },
        },
        "keyword_rules": KEYWORD_RULES,
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    summary = Counter(code_to_final.values())
    print("[build_label_mapping] mapping complete")
    for label in sorted(summary):
        print(f"  {label}: {summary[label]}")


if __name__ == "__main__":
    main()
