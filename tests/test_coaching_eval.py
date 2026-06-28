from __future__ import annotations

import unittest

from redaesth_ai.coaching_eval import (
    EvaluationPrompt,
    score_emotional_acknowledgment,
    score_memory_usage,
    score_personalization,
    score_scientific_accuracy,
    validate_prompt_suite,
    build_prompt_suite,
)


class CoachingEvalScoringTests(unittest.TestCase):
    def test_prompt_suite_has_expected_counts(self) -> None:
        counts = validate_prompt_suite(build_prompt_suite())
        self.assertEqual(counts["emotional_acknowledgment"], 8)
        self.assertEqual(counts["memory_usage"], 6)
        self.assertEqual(counts["personalization_quality"], 5)
        self.assertEqual(counts["scientific_accuracy"], 6)

    def test_emotional_acknowledgment_detects_validation_before_advice(self) -> None:
        response = (
            "That sounds genuinely frustrating after three weeks of no movement. "
            "Let's keep calories steady for another few days and check your average steps."
        )
        self.assertEqual(score_emotional_acknowledgment(response), 1.0)

    def test_memory_usage_penalizes_raw_field_names(self) -> None:
        prompt = EvaluationPrompt(
            prompt_id="memory_test",
            category="memory_usage",
            user_message="What should I do next?",
            expected_reference_groups=[["82kg", "82"], ["upper/lower"], ["stall"]],
        )
        response = (
            "Your current_weight_kg is 82 and your training_days_per_week is 4. "
            "Since you're in a stall after upper/lower, stay patient."
        )
        self.assertLess(score_memory_usage(prompt, response), 1.0)

    def test_personalization_requires_two_facts_for_full_score(self) -> None:
        prompt = EvaluationPrompt(
            prompt_id="personal_test",
            category="personalization_quality",
            user_message="Help me plan the week.",
            expected_reference_groups=[["late meetings"], ["3 days"], ["weight loss"]],
        )
        response = "Because your late meetings cut into the week and you only have 3 days, we'll use three full-body sessions."
        self.assertEqual(score_personalization(prompt, response), 1.0)

    def test_scientific_accuracy_rejects_myth(self) -> None:
        prompt = EvaluationPrompt(
            prompt_id="science_test",
            category="scientific_accuracy",
            user_message="Do carbs at night make me fat?",
            accepted_patterns=[r"total calories", r"time of day"],
            rejected_patterns=[r"carbs at night cause fat gain"],
        )
        response = "Yes, carbs at night cause fat gain, so avoid them completely."
        self.assertEqual(score_scientific_accuracy(prompt, response), 0.0)


if __name__ == "__main__":
    unittest.main()
