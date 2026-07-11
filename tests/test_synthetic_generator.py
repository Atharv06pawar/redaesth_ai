from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from redaesth.config import RedAesthConfig
from redaesth.synthetic_generator import (
    SMOLLM2_MODEL_ID,
    SmolLM2ChatTemplateTokenizer,
    export_validated_conversations,
    generate_pilot_dataset,
    generate_validated_conversations,
)


def build_config(project_root: Path, *, target_count: int = 12) -> RedAesthConfig:
    return RedAesthConfig(
        project_root=project_root,
        base_model_id=SMOLLM2_MODEL_ID,
        embedding_model_id="test/embedding",
        synthetic_pilot_target_count=target_count,
        synthetic_generation_seed=20260710,
    )


class SyntheticGeneratorTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_config(Path(temp_dir))
            first = generate_validated_conversations(config=config)
            second = generate_validated_conversations(config=config)

            self.assertEqual(first.attempted_count, 12)
            self.assertEqual(
                [conversation.model_dump(mode="json") for conversation in first.conversations],
                [conversation.model_dump(mode="json") for conversation in second.conversations],
            )
            self.assertEqual(
                [result.overall_score for result in first.quality_results],
                [result.overall_score for result in second.quality_results],
            )

    def test_generated_conversations_are_valid_and_memory_adaptive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_validated_conversations(config=build_config(Path(temp_dir)))

            self.assertEqual(result.accepted_count, 12)
            self.assertFalse(result.rejections)
            for conversation, quality in zip(
                result.conversations, result.quality_results, strict=True
            ):
                self.assertTrue(quality.passed)
                self.assertTrue(conversation.expected_coaching_behavior.must_use_memory)
                self.assertTrue(conversation.memory_references)
                self.assertEqual(conversation.all_messages()[-1].role.value, "assistant")
                for reference in conversation.memory_references:
                    self.assertIn(reference.facts[0], conversation.coaching_response.response_text)

    def test_jsonl_export_uses_locked_training_schema_and_chat_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_config(Path(temp_dir), target_count=4)
            result = generate_validated_conversations(config=config)
            output_path = export_validated_conversations(result, config=config)
            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(len(records), 4)
            self.assertEqual(result.dataset_path, output_path)
            tokenizer = SmolLM2ChatTemplateTokenizer()
            for record in records:
                self.assertTrue(
                    {
                        "record_id",
                        "source_id",
                        "language",
                        "domain",
                        "conversations",
                        "text",
                        "synthetic_metadata",
                    }.issubset(record)
                )
                self.assertTrue(record["synthetic_metadata"]["quality_rubric"]["passed"])
                self.assertEqual(
                    record["text"],
                    tokenizer.apply_chat_template(
                        record["conversations"],
                        tokenize=False,
                        add_generation_prompt=False,
                    ),
                )

    def test_pilot_pipeline_writes_exactly_the_configured_count_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_config(Path(temp_dir), target_count=100)
            result = generate_pilot_dataset(config=config)
            records = result.dataset_path.read_text(encoding="utf-8").splitlines()
            report = result.report_path.read_text(encoding="utf-8")

            self.assertEqual(result.attempted_count, 100)
            self.assertEqual(result.accepted_count, 100)
            self.assertFalse(result.rejections)
            self.assertEqual(len(records), 100)
            self.assertIn("- Generated conversations: 100", report)
            self.assertIn("- Accepted: 100", report)
            self.assertIn("- Rejected: 0", report)


if __name__ == "__main__":
    unittest.main()
