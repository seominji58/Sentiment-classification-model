"""Unit tests for preprocess and label mapping helpers."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict

import pandas as pd

from src import label_map, preprocess, utils


class PreprocessCleanTest(unittest.TestCase):
    def setUp(self) -> None:
        utils._STATE.clear()

    def test_clean_drops_na_and_normalises_text(self) -> None:
        df = pd.DataFrame(
            {
                "text": ["  hello   world  ", None, "valid", "  hello   world  ", "trim"],
                "label": [" joy ", "sad", None, " joy ", "  "],
            }
        )

        cleaned = preprocess.clean(df)

        self.assertEqual(len(cleaned), 1)
        row = cleaned.iloc[0]
        self.assertEqual(row["text"], "hello world")
        self.assertEqual(row["label"], "joy")
        self.assertEqual(cleaned.columns.tolist(), ["text", "label"])

    def test_clean_respects_keep_columns(self) -> None:
        df = pd.DataFrame(
            {
                "text": ["text"],
                "label": ["label"],
                "meta": ["info"],
            }
        )

        cleaned = preprocess.clean(df, keep_columns=["text", "label", "meta"])

        self.assertEqual(cleaned.columns.tolist(), ["text", "label", "meta"])
        self.assertEqual(cleaned.loc[0, "meta"], "info")


class LabelMapApplyTest(unittest.TestCase):
    def setUp(self) -> None:
        utils._STATE.clear()
        utils.mark_ok("preprocess")
        self._tmp_dir: tempfile.TemporaryDirectory | None = None

    def tearDown(self) -> None:
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()

    def _write_config(self, mapping: Dict[str, str]) -> Path:
        data = {"label_mapping": mapping}
        self._tmp_dir = tempfile.TemporaryDirectory()
        path = Path(self._tmp_dir.name) / "config.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def test_apply_mapping_drops_unmapped_by_default(self) -> None:
        df = pd.DataFrame(
            {
                "text": ["a", "b", "c"],
                "label": ["A", "B", "C"],
            }
        )
        config_path = self._write_config({"A": "positive", "B": "negative"})

        mapped = label_map.apply_mapping(df, str(config_path))

        self.assertEqual(mapped["label"].tolist(), ["positive", "negative"])
        self.assertEqual(len(mapped), 2)

    def test_apply_mapping_can_preserve_unmapped(self) -> None:
        df = pd.DataFrame(
            {
                "text": ["a", "b"],
                "label": ["A", "Z"],
            }
        )
        config_path = self._write_config({"A": "positive"})

        mapped = label_map.apply_mapping(df, str(config_path), drop_unmapped=False)

        self.assertEqual(mapped["label"].tolist(), ["positive", "Z"])
        self.assertEqual(len(mapped), 2)


if __name__ == "__main__":
    unittest.main()
