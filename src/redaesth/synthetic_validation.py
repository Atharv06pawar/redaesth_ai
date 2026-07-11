"""Validation and quality rubric for structured synthetic coaching conversations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from redaesth_ai.coaching_eval import RAW_MEMORY_FIELD_PATTERNS, score_emotional_acknowledgment

from .config import RedAesthConfig, config
from .scoring import (
    cliche_penalty,
    coaching_signal_score,
    detect_topic_tags,
    domain_relevance_score,
    normalized_text,
    repetition_penalty,
    safety_penalty,
    specificity_score,
)
from .synthetic_memory import memory_spec_by_event_type
from .synthetic_schema import SyntheticCoachingConversation


VALIDATOR_WEIGHTS = {
    "empathy": 0.12,
    "coaching_quality": 0.12,
    "personalization": 0.10,
    "behavioral_adaptation": 0.10,
    "scientific_consistency": 0.12,
    "long_term_memory_usage": 0.12,
    "follow_up_questioning": 0.08,
    "hallucination_detection": 0.10,
    "repetitive_responses": 0.06,
    "scenario_consistency": 0.08,
}

HALLUCINATION_RISK_PATTERNS = (
    "guarantee",
    "cure",
    "always",
    "never",
    "prescribe",
    "diagnose",
    "everyone needs",
    "all you need",
)


@dataclass(slots=True)
class ValidationResult:
    """One deterministic validator result."""

    validator_name: str
    score: float
    passed: bool
    evidence: list[str]


@dataclass(slots=True)
class QualityRubricResult:
    """Aggregate quality result for a synthetic conversation specification."""

    validator_results: list[ValidationResult]
    overall_score: float
    passed: bool
    blockers: list[str]


def validator_thresholds(config: RedAesthConfig) -> dict[str, float]:
    """Return threshold mapping for the synthetic quality rubric."""

    return {
        "empathy": config.synthetic_min_empathy_score,
        "coaching_quality": config.synthetic_min_coaching_quality_score,
        "personalization": config.synthetic_min_personalization_score,
        "behavioral_adaptation": config.synthetic_min_behavioral_adaptation_score,
        "scientific_consistency": config.synthetic_min_scientific_consistency_score,
        "long_term_memory_usage": config.synthetic_min_memory_usage_score,
        "follow_up_questioning": config.synthetic_min_follow_up_questioning_score,
        "hallucination_detection": config.synthetic_min_hallucination_safety_score,
        "repetitive_responses": config.synthetic_min_repetition_score,
        "scenario_consistency": config.synthetic_min_scenario_consistency_score,
    }


def clamp(score: float) -> float:
    """Clamp a score into the 0-1 range."""

    return max(0.0, min(score, 1.0))


def response_text(conversation: SyntheticCoachingConversation) -> str:
    """Return normalized assistant response text."""

    return conversation.coaching_response.response_text


def important_profile_facts(conversation: SyntheticCoachingConversation) -> list[str]:
    """Collect profile facts that should plausibly personalize the response."""

    profile = conversation.user_profile
    persona = conversation.persona
    facts = [
        profile.equipment_access,
        conversation.coaching_goal.summary,
        persona.motivation_style.value.replace("_", " "),
        persona.lifestyle.value.replace("_", " "),
        *profile.active_injuries,
        *profile.schedule_constraints[:2],
        *profile.priorities[:2],
        *profile.nutrition_constraints[:2],
    ]
    normalized = []
    for fact in facts:
        fact = fact.strip().lower()
        if fact and fact not in normalized:
            normalized.append(fact)
    return normalized


def text_contains_any(text: str, phrases: list[str]) -> int:
    """Count distinct phrase matches in normalized text."""

    normalized = normalized_text(text)
    count = 0
    for phrase in phrases:
        if phrase and normalized_text(phrase) in normalized:
            count += 1
    return count


def evaluate_empathy(conversation: SyntheticCoachingConversation, thresholds: dict[str, float]) -> ValidationResult:
    """Score emotional acknowledgment using the existing coaching-eval heuristic."""

    user_text = conversation.final_user_message()
    score = score_emotional_acknowledgment(response_text(conversation))
    if not conversation.expected_coaching_behavior.emotional_objectives and not conversation.scenario.typical_emotions:
        score = 1.0 if response_text(conversation).strip() else 0.0
    evidence = [f"score={score:.2f} from acknowledgement heuristic"]
    return ValidationResult("empathy", score, score >= thresholds["empathy"], evidence)


def evaluate_coaching_quality(
    conversation: SyntheticCoachingConversation,
    thresholds: dict[str, float],
) -> ValidationResult:
    """Score whether the response provides actionable, specific coaching."""

    assistant_text = response_text(conversation)
    score = (
        0.40 * coaching_signal_score(assistant_text)
        + 0.35 * specificity_score(assistant_text)
        + 0.15 * (1.0 - min(safety_penalty(assistant_text) / 0.20, 1.0))
        + 0.10 * (1.0 - min(cliche_penalty(assistant_text) / 0.10, 1.0))
    )
    score = clamp(score)
    evidence = [
        f"coaching_signal={coaching_signal_score(assistant_text):.2f}",
        f"specificity={specificity_score(assistant_text):.2f}",
    ]
    return ValidationResult(
        "coaching_quality",
        score,
        score >= thresholds["coaching_quality"],
        evidence,
    )


def evaluate_personalization(
    conversation: SyntheticCoachingConversation,
    thresholds: dict[str, float],
) -> ValidationResult:
    """Score use of persona and user-profile facts."""

    facts = important_profile_facts(conversation)
    matched = text_contains_any(response_text(conversation), facts)
    denominator = max(1, min(len(facts), 4))
    score = clamp(matched / denominator)
    evidence = [f"matched_profile_facts={matched}", f"profile_fact_pool={len(facts)}"]
    return ValidationResult(
        "personalization",
        score,
        score >= thresholds["personalization"],
        evidence,
    )


def evaluate_behavioral_adaptation(
    conversation: SyntheticCoachingConversation,
    thresholds: dict[str, float],
) -> ValidationResult:
    """Score whether the response adapts to constraints instead of offering generic advice."""

    adaptive_constraints = (
        conversation.user_profile.active_injuries
        + conversation.user_profile.schedule_constraints
        + conversation.user_profile.nutrition_constraints
    )
    if conversation.user_profile.stress_level.value == "high":
        adaptive_constraints.append("stress")
    if conversation.user_profile.sleep_hours_average < 6.0:
        adaptive_constraints.append("sleep")
    matched = text_contains_any(response_text(conversation), adaptive_constraints)
    score = 0.0
    if conversation.coaching_response.adaptation_summary:
        score += 0.5
    if adaptive_constraints:
        score += 0.5 * min(matched / max(1, min(len(adaptive_constraints), 2)), 1.0)
    else:
        score = 1.0 if conversation.coaching_response.adaptation_summary else 0.5
    score = clamp(score)
    evidence = [f"matched_constraints={matched}", f"constraint_pool={len(adaptive_constraints)}"]
    return ValidationResult(
        "behavioral_adaptation",
        score,
        score >= thresholds["behavioral_adaptation"],
        evidence,
    )


def evaluate_scientific_consistency(
    conversation: SyntheticCoachingConversation,
    thresholds: dict[str, float],
) -> ValidationResult:
    """Score whether the response remains grounded and safety-aware."""

    assistant_text = response_text(conversation)
    tags = detect_topic_tags(f"{conversation.final_user_message()}\n{assistant_text}")
    domain_score, _ = domain_relevance_score(tags, assistant_text)
    principle_score = min(len(conversation.coaching_response.cited_principles) / 2, 1.0)
    safety_score = 1.0 - min(safety_penalty(assistant_text) / 0.20, 1.0)
    score = clamp(0.35 * principle_score + 0.35 * safety_score + 0.30 * domain_score)
    evidence = [
        f"principle_count={len(conversation.coaching_response.cited_principles)}",
        f"domain_score={domain_score:.2f}",
    ]
    return ValidationResult(
        "scientific_consistency",
        score,
        score >= thresholds["scientific_consistency"],
        evidence,
    )


def evaluate_long_term_memory_usage(
    conversation: SyntheticCoachingConversation,
    thresholds: dict[str, float],
) -> ValidationResult:
    """Score whether memory is used naturally and only when appropriate."""

    assistant_text = normalized_text(response_text(conversation))
    if not conversation.memory_references:
        score = 1.0 if not conversation.expected_coaching_behavior.must_use_memory else 0.0
        evidence = ["no_memory_references_present"]
        return ValidationResult(
            "long_term_memory_usage",
            score,
            score >= thresholds["long_term_memory_usage"],
            evidence,
        )

    matched_references = 0
    avoid_hits = 0
    for reference in conversation.memory_references:
        if any(normalized_text(fact) in assistant_text for fact in reference.facts if fact.strip()):
            matched_references += 1
        if reference.usage_mode.value == "avoid" and any(
            normalized_text(fact) in assistant_text for fact in reference.facts if fact.strip()
        ):
            avoid_hits += 1
        _ = memory_spec_by_event_type(reference.event_type)

    score = matched_references / len(conversation.memory_references)
    if any(field in assistant_text for field in RAW_MEMORY_FIELD_PATTERNS):
        score -= 0.5
    if avoid_hits:
        score -= 0.25 * avoid_hits
    score = clamp(score)
    evidence = [
        f"matched_memory_references={matched_references}/{len(conversation.memory_references)}",
        f"avoid_hits={avoid_hits}",
    ]
    return ValidationResult(
        "long_term_memory_usage",
        score,
        score >= thresholds["long_term_memory_usage"],
        evidence,
    )


def evaluate_follow_up_questioning(
    conversation: SyntheticCoachingConversation,
    thresholds: dict[str, float],
) -> ValidationResult:
    """Score follow-up question coverage against the scenario contract."""

    required = max(
        conversation.expected_coaching_behavior.required_follow_up_questions,
        conversation.scenario.required_follow_up_questions,
    )
    actual = len(conversation.coaching_response.follow_up_questions)
    score = 1.0 if required == 0 else clamp(actual / required)
    evidence = [f"required={required}", f"actual={actual}"]
    return ValidationResult(
        "follow_up_questioning",
        score,
        score >= thresholds["follow_up_questioning"],
        evidence,
    )


def evaluate_hallucination_detection(
    conversation: SyntheticCoachingConversation,
    thresholds: dict[str, float],
) -> ValidationResult:
    """Score whether the response avoids risky or fabricated claims."""

    assistant_text = normalized_text(response_text(conversation))
    penalty = 0.0
    matches = []
    for pattern in HALLUCINATION_RISK_PATTERNS:
        if pattern in assistant_text:
            matches.append(pattern)
            penalty += 0.20
    injury_relevant = (
        any(domain.value == "injury_recovery" for domain in conversation.scenario.domains)
        or "pain" in normalized_text(conversation.final_user_message())
        or "injury" in normalized_text(conversation.final_user_message())
    )
    if (
        injury_relevant
        and conversation.user_profile.active_injuries
        and "pain" not in assistant_text
        and "modify" not in assistant_text
    ):
        penalty += 0.20
        matches.append("injury_without_modification_language")
    score = clamp(1.0 - penalty)
    evidence = matches or ["no_high_risk_patterns"]
    return ValidationResult(
        "hallucination_detection",
        score,
        score >= thresholds["hallucination_detection"],
        evidence,
    )


def evaluate_repetitive_responses(
    conversation: SyntheticCoachingConversation,
    thresholds: dict[str, float],
) -> ValidationResult:
    """Score whether the response avoids repeated sentence fragments."""

    penalty = repetition_penalty(response_text(conversation))
    score = clamp(1.0 - min(penalty / 0.15, 1.0))
    evidence = [f"repetition_penalty={penalty:.2f}"]
    return ValidationResult(
        "repetitive_responses",
        score,
        score >= thresholds["repetitive_responses"],
        evidence,
    )


def evaluate_scenario_consistency(
    conversation: SyntheticCoachingConversation,
    thresholds: dict[str, float],
) -> ValidationResult:
    """Score whether the response fits the declared scenario and objectives."""

    assistant_text = response_text(conversation)
    keyword_pool = conversation.scenario.response_keywords or conversation.scenario.coaching_objectives
    keyword_matches = text_contains_any(assistant_text, keyword_pool)
    tags = detect_topic_tags(f"{conversation.final_user_message()}\n{assistant_text}")
    domain_score, _ = domain_relevance_score(tags, assistant_text)
    keyword_score = min(keyword_matches / max(1, min(len(keyword_pool), 3)), 1.0)
    score = clamp(0.60 * keyword_score + 0.40 * domain_score)
    evidence = [f"keyword_matches={keyword_matches}", f"domain_score={domain_score:.2f}"]
    return ValidationResult(
        "scenario_consistency",
        score,
        score >= thresholds["scenario_consistency"],
        evidence,
    )


def run_all_validators(
    conversation: SyntheticCoachingConversation,
    *,
    config: RedAesthConfig = config,
) -> list[ValidationResult]:
    """Run the full deterministic validator suite for one conversation."""

    thresholds = validator_thresholds(config)
    return [
        evaluate_empathy(conversation, thresholds),
        evaluate_coaching_quality(conversation, thresholds),
        evaluate_personalization(conversation, thresholds),
        evaluate_behavioral_adaptation(conversation, thresholds),
        evaluate_scientific_consistency(conversation, thresholds),
        evaluate_long_term_memory_usage(conversation, thresholds),
        evaluate_follow_up_questioning(conversation, thresholds),
        evaluate_hallucination_detection(conversation, thresholds),
        evaluate_repetitive_responses(conversation, thresholds),
        evaluate_scenario_consistency(conversation, thresholds),
    ]


def score_synthetic_conversation(
    conversation: SyntheticCoachingConversation,
    *,
    config: RedAesthConfig = config,
) -> QualityRubricResult:
    """Apply the full synthetic quality rubric and return PASS / FAIL."""

    thresholds = validator_thresholds(config)
    results = run_all_validators(conversation, config=config)
    weighted_sum = 0.0
    for result in results:
        weighted_sum += result.score * VALIDATOR_WEIGHTS[result.validator_name]
    overall_score = clamp(weighted_sum / sum(VALIDATOR_WEIGHTS.values()))

    blockers: list[str] = []
    for result in results:
        threshold = thresholds[result.validator_name]
        if result.score < threshold:
            blockers.append(
                f"{result.validator_name} scored {result.score:.2f} below threshold {threshold:.2f}"
            )

    for required in conversation.quality_metadata.must_pass_validators:
        matching = next((result for result in results if result.validator_name == required), None)
        if matching is None:
            blockers.append(f"required validator `{required}` is not implemented")
        elif not matching.passed:
            blockers.append(f"required validator `{required}` did not pass")

    if overall_score < config.synthetic_quality_threshold:
        blockers.append(
            f"overall synthetic quality scored {overall_score:.2f} below threshold "
            f"{config.synthetic_quality_threshold:.2f}"
        )

    return QualityRubricResult(
        validator_results=results,
        overall_score=overall_score,
        passed=not blockers,
        blockers=blockers,
    )


def render_quality_summary(result: QualityRubricResult) -> dict[str, float | bool | list[str]]:
    """Return a stable summary payload for documentation and tests."""

    return {
        "passed": result.passed,
        "overall_score": round(result.overall_score, 4),
        "blockers": list(result.blockers),
        **{
            validator_result.validator_name: round(validator_result.score, 4)
            for validator_result in result.validator_results
        },
    }
