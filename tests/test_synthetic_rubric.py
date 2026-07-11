from __future__ import annotations

import unittest

from redaesth.synthetic_rubric import evaluate_synthetic_conversation, synthetic_quality_contract
from tests.synthetic_test_fixtures import (
    build_invalid_synthetic_conversation,
    build_valid_synthetic_conversation,
)


class SyntheticRubricTests(unittest.TestCase):
    def test_quality_contract_exposes_named_thresholds(self) -> None:
        contract = synthetic_quality_contract()
        self.assertIn("overall_quality", contract)
        self.assertIn("empathy", contract)
        self.assertIn("scientific_consistency", contract)

    def test_valid_fixture_passes_quality_rubric(self) -> None:
        result = evaluate_synthetic_conversation(build_valid_synthetic_conversation())
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.overall_score, synthetic_quality_contract()["overall_quality"])

    def test_invalid_fixture_fails_quality_rubric(self) -> None:
        result = evaluate_synthetic_conversation(build_invalid_synthetic_conversation())
        self.assertFalse(result.passed)
        self.assertTrue(result.blockers)


if __name__ == "__main__":
    unittest.main()
