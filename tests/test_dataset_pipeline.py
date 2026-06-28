from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from redaesth.config import RedAesthConfig
from redaesth.dataset_pipeline import download_approved_datasets


def build_config(project_root: Path) -> RedAesthConfig:
    """Create a test config rooted in a temporary project directory."""

    return RedAesthConfig(
        project_root=project_root,
        base_model_id="test/model",
        embedding_model_id="test/embedding",
    )


def write_approved_report(project_root: Path, approved: list[dict[str, object]]) -> Path:
    """Write a minimal approved dataset report for tests."""

    report_path = project_root / "research" / "dataset_discovery" / "reports" / "approved_datasets.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "generated_at": "20260628T181705Z",
                "source_report": "research/dataset_discovery/reports/discovered.json",
                "approved": approved,
                "rejected": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path


def fake_downloader(
    dataset_id: str,
    target_dir: Path,
    token: str | None,
    force_download: bool,
) -> Path:
    """Materialize a tiny fake dataset snapshot locally."""

    del token, force_download
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "train.jsonl").write_text('{"text": "sample"}\n', encoding="utf-8")
    nested_dir = target_dir / "metadata"
    nested_dir.mkdir(exist_ok=True)
    (nested_dir / "dataset_id.txt").write_text(dataset_id, encoding="utf-8")
    return target_dir


class DatasetPipelineTests(unittest.TestCase):
    def test_download_manifest_is_written_for_approved_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            write_approved_report(
                project_root,
                [
                    {
                        "id": "ulysses531/fitness-conversation-dataset",
                        "license": "apache-2.0",
                        "queries": ["fitness conversation"],
                        "downloads": 44,
                        "likes": 6,
                        "last_modified": "2026-06-01T00:00:00.000Z",
                        "dataset_url": "https://huggingface.co/datasets/ulysses531/fitness-conversation-dataset",
                    }
                ],
            )

            manifest_path = download_approved_datasets(config=config, downloader=fake_downloader)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["approved_count"], 1)
            self.assertFalse(manifest["dry_run"])
            self.assertEqual(manifest["datasets"][0]["status"], "downloaded")
            self.assertEqual(manifest["datasets"][0]["file_count"], 2)
            self.assertEqual(
                manifest["datasets"][0]["id"],
                "ulysses531/fitness-conversation-dataset",
            )
            self.assertTrue(manifest["datasets"][0]["files"][0]["sha256"])

    def test_existing_download_is_reused_without_reinvoking_downloader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            write_approved_report(
                project_root,
                [
                    {
                        "id": "ulysses531/fitness-conversation-dataset",
                        "license": "apache-2.0",
                        "queries": ["fitness conversation"],
                        "downloads": 44,
                        "likes": 6,
                        "last_modified": "2026-06-01T00:00:00.000Z",
                        "dataset_url": "https://huggingface.co/datasets/ulysses531/fitness-conversation-dataset",
                    }
                ],
            )
            download_approved_datasets(config=config, downloader=fake_downloader)

            def failing_downloader(
                dataset_id: str,
                target_dir: Path,
                token: str | None,
                force_download: bool,
            ) -> Path:
                del dataset_id, target_dir, token, force_download
                raise AssertionError("downloader should not be called for existing data")

            manifest_path = download_approved_datasets(config=config, downloader=failing_downloader)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["datasets"][0]["status"], "reused_existing")
            self.assertEqual(manifest["datasets"][0]["file_count"], 2)

    def test_empty_approved_list_raises_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            write_approved_report(project_root, [])

            with self.assertRaisesRegex(ValueError, "No approved datasets were found"):
                download_approved_datasets(config=config, downloader=fake_downloader)


if __name__ == "__main__":
    unittest.main()
