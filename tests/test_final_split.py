from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from redaesth.config import RedAesthConfig
from redaesth.final_split import stratified_split_records


def build_config(project_root: Path) -> RedAesthConfig:
    return RedAesthConfig(
        project_root=project_root,
        base_model_id="test/model",
        embedding_model_id="test/embedding",
        final_train_ratio=0.80,
        final_validation_ratio=0.10,
        final_test_ratio=0.10,
        final_min_train_examples=0,
        final_min_validation_examples=0,
        final_min_test_examples=0,
    )


def make_record(record_id: str, source_id: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_id": source_id,
        "text": f"text::{record_id}",
        "conversations": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
    }


class FinalSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_stratified_by_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_config(Path(temp_dir))
            records = [
                make_record(f"a::{index:03d}", "source-a")
                for index in range(60)
            ] + [
                make_record(f"b::{index:03d}", "source-b")
                for index in range(40)
            ]

            train_one, val_one, test_one = stratified_split_records(records, config=config)
            train_two, val_two, test_two = stratified_split_records(records, config=config)

            self.assertEqual([record["record_id"] for record in train_one], [record["record_id"] for record in train_two])
            self.assertEqual([record["record_id"] for record in val_one], [record["record_id"] for record in val_two])
            self.assertEqual([record["record_id"] for record in test_one], [record["record_id"] for record in test_two])

            self.assertEqual(len(train_one), 80)
            self.assertEqual(len(val_one), 10)
            self.assertEqual(len(test_one), 10)
            self.assertEqual(Counter(record["source_id"] for record in val_one), {"source-a": 6, "source-b": 4})
            self.assertEqual(Counter(record["source_id"] for record in test_one), {"source-a": 6, "source-b": 4})


if __name__ == "__main__":
    unittest.main()
