from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from redaesth.config import resolve_base_model_id


class ConfigResolutionTests(unittest.TestCase):
    def test_selected_model_file_takes_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_model = Path(temp_dir) / "selected_model.txt"
            selected_model.write_text("Qwen/Qwen3-1.7B\n", encoding="utf-8")

            decision_log = Path(temp_dir) / "DECISION_LOG.md"
            decision_log.write_text(
                "## Decision 1: Test\n**Decision:** Select `HuggingFaceTB/SmolLM2-1.7B-Instruct` as the initial BASE_MODEL.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                resolve_base_model_id(selected_model, decision_log),
                "Qwen/Qwen3-1.7B",
            )

    def test_decision_log_is_used_when_selected_model_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_model = Path(temp_dir) / "missing.txt"
            decision_log = Path(temp_dir) / "DECISION_LOG.md"
            decision_log.write_text(
                "## Decision 4: Initial base model selection\n"
                "**Decision:** Select `HuggingFaceTB/SmolLM2-1.7B-Instruct` as the initial BASE_MODEL.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                resolve_base_model_id(selected_model, decision_log),
                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            )


if __name__ == "__main__":
    unittest.main()
