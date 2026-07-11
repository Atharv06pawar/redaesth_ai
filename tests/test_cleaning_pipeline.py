from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from redaesth.cleaning import clean_raw_datasets
from redaesth.config import RedAesthConfig


def build_config(project_root: Path) -> RedAesthConfig:
    """Create a test config rooted in a temporary project directory."""

    return RedAesthConfig(
        project_root=project_root,
        base_model_id="test/model",
        embedding_model_id="test/embedding",
    )


def write_raw_manifest(
    project_root: Path,
    *,
    dataset_id: str,
    local_dir: Path,
    files: list[str],
) -> Path:
    """Write a minimal raw-data manifest for the cleaning tests."""

    manifest_path = project_root / "data" / "raw" / "raw_data_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "generated_at": "20260628T181905Z",
                "datasets": [
                    {
                        "id": dataset_id,
                        "license": "apache-2.0",
                        "dataset_url": f"https://huggingface.co/datasets/{dataset_id}",
                        "local_dir": str(local_dir),
                        "files": [{"path": file_name} for file_name in files],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


class CleaningPipelineTests(unittest.TestCase):
    def test_chat_format_dataset_is_cleaned_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            dataset_dir = project_root / "data" / "raw" / "huggingface" / "demo"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "dataset.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "conversations": [
                                    {"role": "user", "content": "  How should I structure my week?  "},
                                    {
                                        "role": "assistant",
                                        "content": " Use three full-body sessions and keep one easy recovery day. ",
                                    },
                                ]
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "conversations": [
                                    {"role": "user", "content": "我最近训练很累。"},
                                    {
                                        "role": "assistant",
                                        "content": "先把强度降一点，再观察睡眠和食欲。",
                                    },
                                ]
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            write_raw_manifest(
                project_root,
                dataset_id="demo/chat-dataset",
                local_dir=dataset_dir,
                files=["dataset.jsonl"],
            )

            cleaned_path, report_path = clean_raw_datasets(config=config)
            cleaned_records = [
                json.loads(line)
                for line in cleaned_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(len(cleaned_records), 2)
            self.assertEqual(cleaned_records[0]["conversations"][0]["content"], "How should I structure my week?")
            self.assertEqual(cleaned_records[1]["language_hint"], "majority_non_ascii")
            self.assertIn("majority_non_ascii", cleaned_records[1]["quality_flags"])
            self.assertEqual(report["totals"]["kept_records"], 2)
            self.assertEqual(report["totals"]["rejected_records"], 0)

    def test_question_answer_pairs_are_normalized_into_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            dataset_dir = project_root / "data" / "raw" / "huggingface" / "qa"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "dataset.json"
            dataset_path.write_text(
                json.dumps(
                    [
                        {
                            "question": "How many rest days should I keep?",
                            "answer": "Start with one or two, then adjust from recovery quality.",
                        }
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            write_raw_manifest(
                project_root,
                dataset_id="demo/qa-dataset",
                local_dir=dataset_dir,
                files=["dataset.json"],
            )

            cleaned_path, report_path = clean_raw_datasets(config=config)
            cleaned_records = [
                json.loads(line)
                for line in cleaned_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(len(cleaned_records), 1)
            self.assertEqual(
                [message["role"] for message in cleaned_records[0]["conversations"]],
                ["user", "assistant"],
            )
            self.assertEqual(cleaned_records[0]["language_hint"], "mostly_ascii")
            self.assertEqual(report["datasets"][0]["processed_files"], ["dataset.json"])

    def test_csv_instruct_rows_are_parsed_into_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            dataset_dir = project_root / "data" / "raw" / "huggingface" / "csv"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "dataset.csv"
            dataset_path.write_text(
                'text\n"<s>[INST]I feel anxious about missing workouts this week.[/INST] That sounds frustrating. Start by protecting two sessions and one short walk, then reassess on Sunday.</s>"\n',
                encoding="utf-8",
            )
            write_raw_manifest(
                project_root,
                dataset_id="demo/csv-dataset",
                local_dir=dataset_dir,
                files=["dataset.csv"],
            )

            cleaned_path, _ = clean_raw_datasets(config=config)
            cleaned_record = json.loads(cleaned_path.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(
                [message["role"] for message in cleaned_record["conversations"]],
                ["user", "assistant"],
            )
            self.assertIn("anxious", cleaned_record["conversations"][0]["content"])

    def test_dataset_with_only_unsupported_files_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            dataset_dir = project_root / "data" / "raw" / "huggingface" / "notes"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "README.md").write_text("metadata only\n", encoding="utf-8")
            write_raw_manifest(
                project_root,
                dataset_id="demo/unsupported-dataset",
                local_dir=dataset_dir,
                files=["README.md"],
            )

            with self.assertRaisesRegex(RuntimeError, "has no supported data files"):
                clean_raw_datasets(config=config)


if __name__ == "__main__":
    unittest.main()
