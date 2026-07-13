from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from redaesth.config import RedAesthConfig
from redaesth.synthetic_generator import (
    SMOLLM2_MODEL_ID,
    ConversationDeduplicationIndex,
    build_synthetic_conversation,
    evaluate_production_quality_gates,
    generate_production_corpus,
)
from redaesth.synthetic_personas import persona_by_id
from redaesth.synthetic_scenarios import scenario_by_id


def build_config(project_root: Path, *, target_count: int) -> RedAesthConfig:
    return RedAesthConfig(
        project_root=project_root,
        base_model_id=SMOLLM2_MODEL_ID,
        embedding_model_id="test/embedding",
        synthetic_production_target_count=target_count,
        synthetic_factory_batch_size=5,
        synthetic_generation_seed=20260710,
    )


class SyntheticProductionFactoryTests(unittest.TestCase):
    def test_batch_generation_builds_requested_configurable_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_config(Path(temp_dir), target_count=24)
            result = generate_production_corpus(config=config)

            self.assertTrue(result.completed)
            self.assertEqual(result.state.accepted_count, 24)
            records = result.train_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 24)

    def test_resume_continues_from_staging_without_regenerating_accepted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_config(Path(temp_dir), target_count=20)
            interrupted = generate_production_corpus(config=config, batch_size=4, max_batches=1)
            staged_before = [
                json.loads(line)
                for line in interrupted.accepted_staging_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertFalse(interrupted.completed)
            self.assertEqual(len(staged_before), 4)

            resumed = generate_production_corpus(config=config)
            final_records = [
                json.loads(line)
                for line in resumed.train_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertTrue(resumed.completed)
            self.assertEqual(resumed.state.accepted_count, 20)
            self.assertEqual(
                [record["record_id"] for record in staged_before],
                [record["record_id"] for record in final_records[:4]],
            )
            self.assertEqual(
                len({record["factory_metadata"]["candidate_index"] for record in final_records}),
                20,
            )

    def test_deduplication_rejects_exact_and_near_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_config(Path(temp_dir), target_count=20)
            persona = persona_by_id("busy_fat_loss_professional")
            scenario = scenario_by_id("fat_loss_planning")
            original = build_synthetic_conversation(
                persona=persona,
                scenario=scenario,
                sample_index=0,
                variation_index=0,
                config=config,
            )
            index = ConversationDeduplicationIndex(config=config, target_count=20)
            index.register_messages(
                [
                    {"role": turn.role.value, "content": turn.content}
                    for turn in original.all_messages()
                ]
            )

            self.assertEqual(index.rejection_reason(original), "exact_duplicate")

            near_response = original.coaching_response.model_copy(
                update={
                    "response_text": original.coaching_response.response_text
                    + " A very small wording adjustment does not make this a new conversation."
                }
            )
            near_duplicate = original.model_copy(update={"coaching_response": near_response})
            self.assertEqual(index.rejection_reason(near_duplicate), "near_duplicate")

    def test_quality_gates_fail_closed_when_a_required_metric_exceeds_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_config(Path(temp_dir), target_count=10)
            statistics = {
                "validator_pass_rate": 1.0,
                "duplicate_rate": 0.0,
                "distribution_deviations": {
                    "persona": 0.0,
                    "scenario": 0.0,
                },
                "distributions": {
                    "memory_category": {
                        "goal": {"count": 5, "percentage": 50.0},
                    }
                },
            }
            gates = evaluate_production_quality_gates(statistics, config=config)
            self.assertEqual(gates["memory_usage_balance"]["status"], "FAIL")

            statistics["distributions"]["memory_category"]["goal"]["percentage"] = 40.0
            gates = evaluate_production_quality_gates(statistics, config=config)
            self.assertTrue(all(gate["status"] == "PASS" for gate in gates.values()))

    def test_packaging_writes_manifest_card_and_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_config(Path(temp_dir), target_count=24)
            result = generate_production_corpus(config=config)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            statistics = json.loads(result.statistics_path.read_text(encoding="utf-8"))
            card = result.dataset_card_path.read_text(encoding="utf-8")

            self.assertEqual(manifest["sample_count"], 24)
            self.assertEqual(manifest["artifacts"]["synthetic_train"]["sample_count"], 24)
            self.assertIn("sha256", manifest["artifacts"]["synthetic_train"])
            self.assertEqual(statistics["sample_count"], 24)
            self.assertIn("validator_scores", statistics)
            self.assertIn("Dataset SHA256", card)


if __name__ == "__main__":
    unittest.main()
