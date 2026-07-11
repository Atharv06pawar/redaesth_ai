from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from redaesth.config import RedAesthConfig
from redaesth.scoring import score_cleaned_dataset


def build_config(project_root: Path) -> RedAesthConfig:
    """Create a test config rooted in a temporary project directory."""

    return RedAesthConfig(
        project_root=project_root,
        base_model_id="test/model",
        embedding_model_id="test/embedding",
    )


def write_cleaned_dataset(project_root: Path, records: list[dict[str, object]]) -> Path:
    """Write a cleaned dataset fixture for scoring tests."""

    path = project_root / "data" / "cleaned" / "cleaned_dataset.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


class ScoringPipelineTests(unittest.TestCase):
    def test_high_quality_coaching_record_passes_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            write_cleaned_dataset(
                project_root,
                [
                    {
                        "conversation_id": "demo::000001",
                        "dataset_id": "demo/coaching",
                        "source_file": "coaching.jsonl",
                        "source_record_index": 1,
                        "source_license": "apache-2.0",
                        "conversations": [
                            {
                                "role": "user",
                                "content": "I'm frustrated because my fat loss has stalled even though I still train four days a week.",
                            },
                            {
                                "role": "assistant",
                                "content": "That sounds genuinely frustrating after staying consistent. Let's keep protein high, track your average steps for 7 days, and reduce calories by 150 if your weekly weight trend still does not move.",
                            },
                        ],
                        "turn_count": 2,
                        "user_turn_count": 1,
                        "assistant_turn_count": 1,
                        "contains_system_message": False,
                        "language_hint": "mostly_ascii",
                        "majority_non_ascii": False,
                        "conversation_characters": 248,
                        "quality_flags": [],
                        "normalized_sha256": "abc123",
                    }
                ],
            )

            scored_path, report_path = score_cleaned_dataset(config=config)
            record = json.loads(scored_path.read_text(encoding="utf-8").splitlines()[0])
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertTrue(record["passes_training_filter"])
            self.assertGreaterEqual(record["overall_quality_score"], 0.55)
            self.assertGreater(record["emotional_acknowledgment_score"], 0.0)
            self.assertEqual(report["totals"]["passes_training_filter"], 1)

    def test_programming_record_is_excluded_as_off_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            write_cleaned_dataset(
                project_root,
                [
                    {
                        "conversation_id": "demo::000002",
                        "dataset_id": "demo/mixed",
                        "source_file": "mixed.jsonl",
                        "source_record_index": 2,
                        "source_license": "apache-2.0",
                        "conversations": [
                            {"role": "user", "content": "Write a Python function to reverse a string."},
                            {
                                "role": "assistant",
                                "content": "```python\ndef reverse_text(value):\n    return value[::-1]\n```",
                            },
                        ],
                        "turn_count": 2,
                        "user_turn_count": 1,
                        "assistant_turn_count": 1,
                        "contains_system_message": False,
                        "language_hint": "mostly_ascii",
                        "majority_non_ascii": False,
                        "conversation_characters": 103,
                        "quality_flags": [],
                        "normalized_sha256": "def456",
                    }
                ],
            )

            scored_path, report_path = score_cleaned_dataset(config=config)
            record = json.loads(scored_path.read_text(encoding="utf-8").splitlines()[0])
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertFalse(record["passes_training_filter"])
            self.assertIn("off_domain_programming", record["exclusion_reasons"])
            self.assertEqual(report["totals"]["exclusion_reasons"]["off_domain_programming"], 1)

    def test_emotional_support_record_is_treated_as_in_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            write_cleaned_dataset(
                project_root,
                [
                    {
                        "conversation_id": "demo::000003",
                        "dataset_id": "demo/emotional",
                        "source_file": "emotional.jsonl",
                        "source_record_index": 3,
                        "source_license": "mit",
                        "conversations": [
                            {
                                "role": "user",
                                "content": "I feel anxious and discouraged because I keep skipping workouts when work stress spikes.",
                            },
                            {
                                "role": "assistant",
                                "content": "That sounds genuinely frustrating. Start with two protected training windows this week, keep one short walk on your busiest day, and notice whether the stress drops when the plan is smaller.",
                            },
                        ],
                        "turn_count": 2,
                        "user_turn_count": 1,
                        "assistant_turn_count": 1,
                        "contains_system_message": False,
                        "language_hint": "mostly_ascii",
                        "majority_non_ascii": False,
                        "conversation_characters": 248,
                        "quality_flags": [],
                        "normalized_sha256": "ghi789",
                    }
                ],
            )

            scored_path, _ = score_cleaned_dataset(config=config)
            record = json.loads(scored_path.read_text(encoding="utf-8").splitlines()[0])

            self.assertIn("emotional_support", record["topic_tags"])
            self.assertGreaterEqual(record["domain_relevance_score"], 0.8)
            self.assertNotIn("weak_domain_signal", record["exclusion_reasons"])

    def test_empty_cleaned_dataset_raises_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            path = project_root / "data" / "cleaned" / "cleaned_dataset.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "No cleaned records were found"):
                score_cleaned_dataset(config=config)


if __name__ == "__main__":
    unittest.main()
