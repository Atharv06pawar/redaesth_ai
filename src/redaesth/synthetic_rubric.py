"""Convenience surface for the synthetic conversation quality rubric."""

from __future__ import annotations

from .config import RedAesthConfig, config
from .synthetic_schema import SyntheticCoachingConversation
from .synthetic_validation import (
    QualityRubricResult,
    render_quality_summary,
    score_synthetic_conversation,
    validator_thresholds,
)


def evaluate_synthetic_conversation(
    conversation: SyntheticCoachingConversation,
    *,
    config: RedAesthConfig = config,
) -> QualityRubricResult:
    """Run the deterministic synthetic quality rubric."""

    return score_synthetic_conversation(conversation, config=config)


def synthetic_quality_contract(config: RedAesthConfig = config) -> dict[str, float]:
    """Expose the current validator thresholds plus the overall pass threshold."""

    thresholds = validator_thresholds(config)
    thresholds["overall_quality"] = config.synthetic_quality_threshold
    return thresholds


__all__ = [
    "evaluate_synthetic_conversation",
    "render_quality_summary",
    "synthetic_quality_contract",
]
