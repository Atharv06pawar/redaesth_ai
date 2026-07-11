from __future__ import annotations

import unittest

from redaesth.synthetic_validation import render_quality_summary, run_all_validators
from tests.synthetic_test_fixtures import (
    build_invalid_synthetic_conversation,
    build_valid_synthetic_conversation,
)


class SyntheticValidationTests(unittest.TestCase):
    def test_valid_fixture_passes_all_validators(self) -> None:
        results = run_all_validators(build_valid_synthetic_conversation())
        summary = render_quality_summary(
            type(
                "Report",
                (),
                {
                    "passed": all(result.passed for result in results),
                    "overall_score": sum(result.score for result in results) / len(results),
                    "blockers": [],
                    "validator_results": results,
                },
            )()
        )

        self.assertTrue(all(result.passed for result in results))
        self.assertGreaterEqual(summary["empathy"], 0.7)
        self.assertGreaterEqual(summary["long_term_memory_usage"], 0.7)

    def test_invalid_fixture_fails_multiple_validators(self) -> None:
        results = run_all_validators(build_invalid_synthetic_conversation())
        failed = {result.validator_name for result in results if not result.passed}

        self.assertIn("scientific_consistency", failed)
        self.assertIn("hallucination_detection", failed)
        self.assertIn("follow_up_questioning", failed)


if __name__ == "__main__":
    unittest.main()
