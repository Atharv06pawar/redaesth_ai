from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from redaesth.config import RedAesthConfig
from redaesth.final_dataset import build_final_dataset


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        del tokenize
        rendered: list[str] = []
        if messages and messages[0]["role"] != "system":
            rendered.append(
                "<|im_start|>system\n"
                "You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
            )
        for message in messages:
            rendered.append(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            )
        if add_generation_prompt:
            rendered.append("<|im_start|>assistant\n")
        return "".join(rendered)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        del add_special_tokens
        return text.split()


def build_config(project_root: Path) -> RedAesthConfig:
    return RedAesthConfig(
        project_root=project_root,
        base_model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        embedding_model_id="test/embedding",
        final_train_ratio=0.75,
        final_validation_ratio=0.15,
        final_test_ratio=0.10,
        final_min_train_examples=6,
        final_min_validation_examples=2,
        final_min_test_examples=1,
        minimum_mostly_ascii_share=0.50,
        maximum_majority_non_ascii_share=0.50,
        maximum_mental_health_adjacent_share=0.35,
        maximum_single_source_share=0.75,
        final_spot_check_sample_size=10,
    )


def make_record(
    record_id: str,
    dataset_id: str,
    topic_tags: list[str],
) -> dict[str, object]:
    return {
        "conversation_id": record_id,
        "dataset_id": dataset_id,
        "source_license": "test-license",
        "conversations": [
            {"role": "user", "content": "Help me plan this week."},
            {"role": "assistant", "content": "Let's keep three sessions and one recovery walk."},
        ],
        "topic_tags": topic_tags,
        "language_hint": "mostly_ascii",
        "quality_flags": [],
        "normalized_sha256": f"sha::{record_id}",
        "overall_quality_score": 0.82,
        "passes_training_filter": True,
        "exclusion_reasons": [],
    }


def write_scored_dataset(project_root: Path, records: list[dict[str, object]]) -> Path:
    path = project_root / "data" / "scored" / "scored_dataset.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


class FinalDatasetPipelineTests(unittest.TestCase):
    def test_final_dataset_pipeline_builds_locked_artifact_and_split_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            fitness_records = [
                make_record(
                    f"fitness::{index:03d}",
                    "ulysses531/fitness-conversation-dataset",
                    ["strength_training"],
                )
                for index in range(8)
            ]
            emotional_records = [
                make_record(
                    f"emotional::{index:03d}",
                    "hizardev/MentalHealth-Counseling",
                    ["emotional_support"],
                )
                for index in range(10)
            ]
            write_scored_dataset(project_root, fitness_records + emotional_records)

            result = build_final_dataset(config=config, tokenizer=FakeTokenizer())

            final_records = [
                json.loads(line)
                for line in result.final_dataset_path.read_text(encoding="utf-8").splitlines()
            ]
            train_records = [
                json.loads(line)
                for line in result.train_path.read_text(encoding="utf-8").splitlines()
            ]
            val_records = [
                json.loads(line)
                for line in result.validation_path.read_text(encoding="utf-8").splitlines()
            ]
            test_records = [
                json.loads(line)
                for line in result.test_path.read_text(encoding="utf-8").splitlines()
            ]
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            audit = json.loads(result.composition_audit_path.read_text(encoding="utf-8"))
            spot_check = json.loads(result.spot_check_report_path.read_text(encoding="utf-8"))
            readiness_report = result.training_readiness_report_path.read_text(encoding="utf-8")

            self.assertEqual(len(final_records), 12)
            self.assertEqual(len(train_records), 9)
            self.assertEqual(len(val_records), 2)
            self.assertEqual(len(test_records), 1)
            self.assertEqual(manifest["splits"]["train"]["sample_count"], 9)
            self.assertEqual(audit["overall_status"], "PASS")
            self.assertEqual(spot_check["status"], "PASS")
            self.assertIn("## GO / NO GO Decision", readiness_report)

            sample = final_records[0]
            self.assertIn("text", sample)
            self.assertIn("source_id", sample)
            self.assertIn("language", sample)
            self.assertIn("domain", sample)
            self.assertTrue(sample["text"].startswith("<|im_start|>system"))

    def test_final_dataset_raises_when_not_enough_records_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            write_scored_dataset(
                project_root,
                [
                    make_record(
                        f"fitness::{index:03d}",
                        "ulysses531/fitness-conversation-dataset",
                        ["strength_training"],
                    )
                    for index in range(5)
                ],
            )

            with self.assertRaisesRegex(RuntimeError, "minimum size requirements"):
                build_final_dataset(config=config, tokenizer=FakeTokenizer())


if __name__ == "__main__":
    unittest.main()
