from __future__ import annotations

import unittest

from redaesth.synthetic_scenarios import get_scenario_library, validate_scenario_library


class SyntheticScenarioTests(unittest.TestCase):
    def test_scenario_library_is_valid(self) -> None:
        scenarios = get_scenario_library()
        validate_scenario_library(scenarios)
        self.assertEqual(len(scenarios), 12)

    def test_missing_required_scenario_is_rejected(self) -> None:
        scenarios = tuple(
            scenario for scenario in get_scenario_library() if scenario.scenario_id != "travel_continuity"
        )

        with self.assertRaisesRegex(ValueError, "missing required scenarios"):
            validate_scenario_library(scenarios)


if __name__ == "__main__":
    unittest.main()
