from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from redaesth.config import RedAesthConfig
from redaesth.final_audit import audit_final_dataset


def build_config(project_root: Path) -> RedAesthConfig:
    return RedAesthConfig(
        project_root=project_root,
        base_model_id="test/model",
        embedding_model_id="test/embedding",
        minimum_mostly_ascii_share=0.60,
        maximum_majority_non_ascii_share=0.40,
        maximum_mental_health_adjacent_share=0.35,
        maximum_off_domain_share=0.05,
        maximum_single_source_share=0.70,
        maximum_exact_duplicate_rate=0.0,
    )


def make_record(
    record_id: str,
    *,
    source_id: str,
    language: str,
    domain: str,
    normalized_sha256: str,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_id": source_id,
        "language": language,
        "domain": domain,
        "normalized_sha256": normalized_sha256,
        "text": "sample",
    }


def write_final_dataset(project_root: Path, records: list[dict[str, object]]) -> Path:
    path = project_root / "data" / "final" / "final_dataset.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


class FinalAuditTests(unittest.TestCase):
    def test_audit_passes_for_balanced_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            records = [
                make_record(
                    f"fitness::{index}",
                    source_id="ulysses531/fitness-conversation-dataset",
                    language="mostly_ascii",
                    domain="fitness-coaching-adjacent",
                    normalized_sha256=f"fit::{index}",
                )
                for index in range(4)
            ] + [
                make_record(
                    f"mental::{index}",
                    source_id="hizardev/MentalHealth-Counseling",
                    language="mostly_ascii",
                    domain="mental-health-adjacent",
                    normalized_sha256=f"mental::{index}",
                )
                for index in range(2)
            ]
            write_final_dataset(project_root, records)

            _, report = audit_final_dataset(config=config)

            self.assertEqual(report["overall_status"], "PASS")
            self.assertEqual(report["checks"]["exact_duplicate_rate"]["status"], "PASS")

    def test_audit_fails_language_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            records = [
                make_record(
                    f"sample::{index}",
                    source_id="source-a",
                    language="majority_non_ascii",
                    domain="fitness-coaching-adjacent",
                    normalized_sha256=f"sha::{index}",
                )
                for index in range(4)
            ] + [
                make_record(
                    "sample::ascii",
                    source_id="source-b",
                    language="mostly_ascii",
                    domain="fitness-coaching-adjacent",
                    normalized_sha256="sha::ascii",
                )
            ]
            write_final_dataset(project_root, records)

            _, report = audit_final_dataset(config=config)

            self.assertEqual(report["checks"]["language_mostly_ascii"]["status"], "FAIL")
            self.assertEqual(report["overall_status"], "FAIL")

    def test_audit_fails_source_balance_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            records = [
                make_record(
                    f"dominant::{index}",
                    source_id="dominant-source",
                    language="mostly_ascii",
                    domain="fitness-coaching-adjacent",
                    normalized_sha256=f"dominant::{index}",
                )
                for index in range(8)
            ] + [
                make_record(
                    f"minor::{index}",
                    source_id="minor-source",
                    language="mostly_ascii",
                    domain="mental-health-adjacent",
                    normalized_sha256=f"minor::{index}",
                )
                for index in range(2)
            ]
            write_final_dataset(project_root, records)

            _, report = audit_final_dataset(config=config)

            self.assertEqual(report["checks"]["source_max_single_source"]["status"], "FAIL")

    def test_audit_fails_duplicate_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config = build_config(project_root)
            records = [
                make_record(
                    "sample::0",
                    source_id="source-a",
                    language="mostly_ascii",
                    domain="fitness-coaching-adjacent",
                    normalized_sha256="duplicate",
                ),
                make_record(
                    "sample::1",
                    source_id="source-b",
                    language="mostly_ascii",
                    domain="fitness-coaching-adjacent",
                    normalized_sha256="duplicate",
                ),
            ]
            write_final_dataset(project_root, records)

            _, report = audit_final_dataset(config=config)

            self.assertEqual(report["checks"]["exact_duplicate_rate"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
