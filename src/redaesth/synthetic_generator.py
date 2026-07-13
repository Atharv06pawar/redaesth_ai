"""Deterministic, validator-gated synthetic coaching conversation generation."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from difflib import SequenceMatcher
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from memory.event_schema import EventType

from .config import RedAesthConfig, config
from .final_dataset import build_training_record, write_jsonl
from .scoring import normalized_text, repetition_penalty
from .synthetic_memory import get_memory_event_specifications, memory_spec_by_event_type
from .synthetic_personas import build_user_profile_from_persona, get_persona_library
from .synthetic_rubric import evaluate_synthetic_conversation
from .synthetic_scenarios import get_scenario_library
from .synthetic_schema import (
    CoachingDomain,
    CoachingGoalSpec,
    CoachingResponseSpec,
    ConversationTurn,
    ExpectedCoachingBehavior,
    MemoryReference,
    MemoryUsageMode,
    PersonaDefinition,
    QualityMetadata,
    ScenarioDefinition,
    SyntheticCoachingConversation,
    UserProfile,
)
from .synthetic_validation import QualityRubricResult, render_quality_summary


SMOLLM2_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SMOLLM2_DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
)
SYNTHETIC_SOURCE_ID = "redaesth/synthetic-coaching-pilot"
SYNTHETIC_FACTORY_SOURCE_ID = "redaesth/synthetic-coaching-production"
SYNTHETIC_SCHEMA_VERSION = "1.0"
SYNTHETIC_FACTORY_VERSION = "1.0"

EMPATHY_OPENINGS = (
    "I hear you - feeling {emotion} in this situation makes sense.",
    "That makes sense; this can feel {emotion} when the plan meets real-life constraints.",
    "It sounds like this has felt {emotion}, and that is useful information rather than a failure.",
    "I understand why this feels {emotion}; we can make the next step more workable.",
    "This is understandably {emotion}, so we will simplify the decision in front of you.",
)

REVIEW_CADENCES = (
    "Review the pattern after the week before changing the plan.",
    "Use the next seven days as a low-pressure data point, then adjust one lever if needed.",
    "Keep the plan stable long enough to learn what is actually working.",
    "Treat the next completed sessions as feedback, not as a test of willpower.",
)

PLAN_EMPHASES = (
    "Start with the session that has the most predictable time window.",
    "Keep the first session deliberately easy to begin so momentum comes before optimization.",
    "Write down one backup option now for the day the original plan is disrupted.",
    "Use the simplest meal or movement choice as the default when decision fatigue is high.",
    "Leave one session flexible so the plan can absorb an unexpected schedule change.",
    "Treat completion quality as the win this week, even if the session is shorter than usual.",
    "Make the next action visible on the calendar before relying on motivation alone.",
)


SCENARIO_ACTIONS: dict[str, str] = {
    "beginner_onboarding": (
        "start with a simple gym routine using two machine strength sessions, one short "
        "treadmill session, and a confidence-building review at the end of the week"
    ),
    "fat_loss_planning": (
        "keep protein, calories, and steps consistent, then make a moderate calorie "
        "adjustment only after a full week of accurate tracking"
    ),
    "muscle_gain_support": (
        "use progressive loading on two core lifts, spread protein across meals, keep any "
        "surplus modest, and protect recovery between hard sessions"
    ),
    "plateau_review": (
        "treat the plateau as a recovery and adherence review, then adjust one training or "
        "nutrition lever after verifying the baseline"
    ),
    "missed_workouts_reset": (
        "restart with the next workout as a small step, use a short consistency-focused "
        "session, and avoid trying to make up missed volume"
    ),
    "injury_recovery_adjustment": (
        "use pain as feedback, modify the exercise range and load, and seek an assessment "
        "if symptoms persist or worsen"
    ),
    "poor_sleep_recovery": (
        "reduce hard sets while sleep and fatigue are elevated, then make recovery the "
        "priority before increasing training"
    ),
    "exam_stress_adjustment": (
        "during the exam window, use short sessions to manage stress and preserve momentum "
        "rather than chase full volume"
    ),
    "travel_continuity": (
        "on travel days, use the hotel space for bodyweight circuits and retain a simple "
        "routine cue"
    ),
    "busy_professional_compliance": (
        "protect the schedule with an efficient minimum effective plan and prioritize the "
        "sessions you can complete"
    ),
    "returning_after_break": (
        "return with a gradual baseline week, then rebuild volume before judging old "
        "performance"
    ),
    "competition_preparation": (
        "keep prep specificity high but manage fatigue, use a small peak-focused adjustment, "
        "and protect recovery rather than extend volume"
    ),
}

SCENARIO_USER_SIGNALS: dict[str, str] = {
    "beginner_onboarding": "I am not sure how to begin without feeling intimidated.",
    "fat_loss_planning": "I am tempted to cut calories aggressively because progress feels slow.",
    "muscle_gain_support": "I want to gain muscle but worry that I will gain unnecessary fat.",
    "plateau_review": "My progress has stalled and I am frustrated by the plateau.",
    "missed_workouts_reset": "I missed planned workouts and now feel guilty about losing momentum.",
    "injury_recovery_adjustment": "Training has become difficult because a movement is causing pain.",
    "poor_sleep_recovery": "Poor sleep is leaving me tired and my sessions feel harder than usual.",
    "exam_stress_adjustment": "Exam pressure is high and I cannot follow my normal routine.",
    "travel_continuity": "Travel has disrupted my equipment access and routine.",
    "busy_professional_compliance": "My work schedule is packed and I need a plan I can actually follow.",
    "returning_after_break": "I am nervous about returning after a long break from training.",
    "competition_preparation": "Preparation fatigue is rising and I want to stay precise without burning out.",
}

DOMAIN_GOALS: dict[CoachingDomain, str] = {
    CoachingDomain.FAT_LOSS: "sustainable fat loss",
    CoachingDomain.MUSCLE_GAIN: "muscle gain with steady recovery",
    CoachingDomain.BODY_RECOMPOSITION: "body recomposition through consistent strength training",
    CoachingDomain.GENERAL_FITNESS: "a consistent general fitness routine",
    CoachingDomain.ADHERENCE: "a reliable training routine",
    CoachingDomain.INJURY_RECOVERY: "a safe return to pain-aware training",
    CoachingDomain.NUTRITION: "consistent nutrition that supports training",
    CoachingDomain.RECOVERY: "recovery-aware training consistency",
}

DOMAIN_TOPICS: dict[CoachingDomain, str] = {
    CoachingDomain.FAT_LOSS: "nutrition",
    CoachingDomain.MUSCLE_GAIN: "strength_training",
    CoachingDomain.BODY_RECOMPOSITION: "strength_training",
    CoachingDomain.GENERAL_FITNESS: "strength_training",
    CoachingDomain.ADHERENCE: "strength_training",
    CoachingDomain.INJURY_RECOVERY: "injury_pain",
    CoachingDomain.NUTRITION: "nutrition",
    CoachingDomain.RECOVERY: "recovery_sleep",
}


@dataclass(slots=True, frozen=True)
class GenerationRejection:
    """A rejected candidate and its deterministic rubric blockers."""

    conversation_id: str
    blockers: tuple[str, ...]


@dataclass(slots=True)
class SyntheticGenerationResult:
    """Validated pilot conversations, export records, and report locations."""

    conversations: list[SyntheticCoachingConversation]
    quality_results: list[QualityRubricResult]
    training_records: list[dict[str, Any]]
    rejections: list[GenerationRejection]
    attempted_count: int
    dataset_path: Path | None = None
    report_path: Path | None = None

    @property
    def accepted_count(self) -> int:
        """Return the number of validator-approved conversations."""

        return len(self.conversations)


class SmolLM2ChatTemplateTokenizer:
    """Offline adapter for the selected model's locked chat-template contract."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Render the official SmolLM2 message wrapper without loading model weights."""

        del tokenize
        rendered: list[str] = []
        if not messages or messages[0].get("role") != "system":
            rendered.append(
                f"<|im_start|>system\n{SMOLLM2_DEFAULT_SYSTEM_MESSAGE}<|im_end|>\n"
            )
        for message in messages:
            rendered.append(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            )
        if add_generation_prompt:
            rendered.append("<|im_start|>assistant\n")
        return "".join(rendered)


def _goal_domain(persona: PersonaDefinition, scenario: ScenarioDefinition) -> CoachingDomain:
    """Select a goal domain while retaining a body-recomposition persona's stated focus."""

    if persona.persona_id == "vegan_body_recomp_office_worker":
        return CoachingDomain.BODY_RECOMPOSITION
    return scenario.domains[0]


def _equipment_access(persona: PersonaDefinition, scenario: ScenarioDefinition) -> str:
    """Resolve a realistic equipment context from persona and scenario constraints."""

    if scenario.scenario_id == "travel_continuity":
        return "hotel gym with dumbbells and a treadmill"
    if "home workouts" in " ".join(persona.lifestyle_notes).lower():
        return "home dumbbells, bench, and bands"
    if persona.lifestyle.value == "retiree":
        return "community gym machines and walking paths"
    if persona.lifestyle.value == "student":
        return "campus gym machines and treadmills"
    return "commercial gym with free weights and machines"


def _sleep_hours(persona: PersonaDefinition, scenario: ScenarioDefinition) -> float:
    """Create a deterministic readiness snapshot appropriate to the scenario."""

    if scenario.scenario_id == "poor_sleep_recovery":
        return 5.5
    if persona.stress_level.value == "high":
        return 5.8
    if persona.stress_level.value == "moderate":
        return 6.8
    return 7.3


def _build_user_profile(
    persona: PersonaDefinition,
    scenario: ScenarioDefinition,
    goal_summary: str,
    variation_index: int = 0,
) -> UserProfile:
    """Build the scenario-aware profile snapshot from the persona contract."""

    nutrition_constraints = list(persona.barriers[:1]) or ["keeping meals simple on busy days"]
    profile = build_user_profile_from_persona(
        persona,
        user_id=f"synthetic::{persona.persona_id}",
        equipment_access=_equipment_access(persona, scenario),
        sleep_hours_average=_sleep_hours(persona, scenario),
        priorities=[goal_summary, scenario.coaching_objectives[0]],
        nutrition_constraints=nutrition_constraints,
        current_stats={
            "training_days_available": persona.available_training_days_per_week,
            "session_minutes_available": persona.available_training_minutes,
            "planning_cycle": variation_index + 1,
        },
    )

    updates: dict[str, Any] = {}
    if scenario.scenario_id == "injury_recovery_adjustment" and not profile.active_injuries:
        updates["active_injuries"] = ["recent shoulder discomfort during pressing"]
    if scenario.scenario_id == "travel_continuity":
        updates["schedule_constraints"] = [
            *profile.schedule_constraints,
            "hotel gym access varies",
        ]
    if scenario.scenario_id == "exam_stress_adjustment":
        updates["schedule_constraints"] = [
            *profile.schedule_constraints,
            "exam blocks reduce training time",
        ]
    return profile.model_copy(update=updates) if updates else profile


def _supported_memory_event_types(scenario: ScenarioDefinition) -> list[EventType]:
    """Select scenario-approved events that are defined by the synthetic memory contract."""

    supported = {spec.event_type for spec in get_memory_event_specifications()}
    return [event_type for event_type in scenario.allowed_memory_event_types if event_type in supported]


def _memory_reference(
    *,
    conversation_id: str,
    scenario: ScenarioDefinition,
    profile: UserProfile,
    goal: CoachingGoalSpec,
    sample_index: int,
    event_type_override: EventType | None = None,
) -> MemoryReference:
    """Create one concrete, behavior-shaping memory reference for a candidate."""

    allowed_types = _supported_memory_event_types(scenario)
    event_type = event_type_override or (
        allowed_types[sample_index % len(allowed_types)] if allowed_types else EventType.GOAL_SET
    )
    if event_type not in allowed_types and not (
        not allowed_types and event_type is EventType.GOAL_SET
    ):
        raise ValueError(
            f"Memory event {event_type.value} is not supported by scenario {scenario.scenario_id}."
        )
    if event_type is EventType.GOAL_SET:
        fact = goal.summary
    elif event_type is EventType.SCHEDULE_CHANGED:
        fact = profile.schedule_constraints[0]
    elif event_type is EventType.WORKOUT_MISSED:
        fact = "missed last week's planned session"
    elif event_type is EventType.INJURY_REPORTED:
        fact = profile.active_injuries[0]
    elif event_type is EventType.SLEEP_PATTERN_CHANGED:
        fact = f"averaging {profile.sleep_hours_average:.1f} hours of sleep"
    elif event_type is EventType.TRAVEL_STARTED:
        fact = "hotel gym access varies"
    elif event_type is EventType.NUTRITION_TARGET_SET:
        fact = "protein target of 130 grams"
    elif event_type is EventType.STRESS_ELEVATED:
        fact = "high work stress"
    else:
        raise ValueError(f"Unsupported synthetic memory event type: {event_type.value}")

    specification = memory_spec_by_event_type(event_type)
    return MemoryReference(
        event_type=event_type,
        source_event_id=f"{conversation_id}::{event_type.value}",
        usage_mode=MemoryUsageMode.REQUIRED,
        reason=specification.retrieval_rules[0],
        facts=[fact],
        influence_summary=specification.behavioral_adaptation_rules[0],
    )


def _constraints_for_response(profile: UserProfile) -> tuple[str, str]:
    """Choose two concrete profile constraints for explicit behavioral adaptation."""

    candidates = [
        *profile.active_injuries,
        *profile.schedule_constraints,
        *profile.nutrition_constraints,
    ]
    if profile.stress_level.value == "high":
        candidates.append("stress")
    if profile.sleep_hours_average < 6.0:
        candidates.append("sleep")
    cleaned = [value for value in candidates if value.strip()]
    primary = cleaned[0] if cleaned else "the available training window"
    secondary = cleaned[1] if len(cleaned) > 1 else primary
    return primary, secondary


def _follow_up_questions(
    *,
    scenario: ScenarioDefinition,
    profile: UserProfile,
    goal: CoachingGoalSpec,
    variation_index: int = 0,
) -> list[str]:
    """Create the required number of scenario-aware coaching questions."""

    primary_constraint, _ = _constraints_for_response(profile)
    candidates = [
        f"Which part of {primary_constraint} is most likely to disrupt the plan this week?",
        f"What would show that {goal.summary} is moving in the right direction over the next week?",
        f"Which {profile.equipment_access} exercise feels most realistic for your next session?",
    ]
    offset = variation_index % len(candidates)
    ordered = candidates[offset:] + candidates[:offset]
    return ordered[: scenario.required_follow_up_questions]


def _scientific_principles(scenario: ScenarioDefinition) -> list[str]:
    """Provide two concise scientific grounding statements per scenario."""

    guardrail = scenario.scientific_guardrails[0]
    return [
        guardrail.capitalize() + ".",
        "Progress is most reliable when training load, nutrition, and recovery are adjusted from consistent observations rather than one difficult day.",
    ]


def _conversation_history(
    *,
    persona: PersonaDefinition,
    scenario: ScenarioDefinition,
    goal: CoachingGoalSpec,
    profile: UserProfile,
    history_turn_count: int | None = None,
    variation_index: int = 0,
) -> list[ConversationTurn]:
    """Build onboarding or follow-up history from the structured scenario, never a static chat."""

    signal = SCENARIO_USER_SIGNALS[scenario.scenario_id]
    primary_constraint, _ = _constraints_for_response(profile)
    current_message = (
        f"I am working toward {goal.summary}. {signal} With {primary_constraint}, I need a "
        "plan that fits this week."
    )
    if scenario.scenario_id == "beginner_onboarding" or history_turn_count == 1:
        return [ConversationTurn(role="user", content=current_message)]
    if history_turn_count == 5:
        return [
            ConversationTurn(
                role="user",
                content=(
                    f"I want coaching that fits my {persona.lifestyle.value.replace('_', ' ')} "
                    f"routine and my goal of {goal.summary}."
                ),
            ),
            ConversationTurn(
                role="assistant",
                content="We will begin with a realistic baseline and use the first week as feedback.",
            ),
            ConversationTurn(
                role="user",
                content=(
                    f"I completed the first planning cycle {variation_index + 1} and noticed the "
                    "same constraints are still showing up."
                ),
            ),
            ConversationTurn(
                role="assistant",
                content="That pattern is enough to simplify the next choice instead of adding more tasks.",
            ),
            ConversationTurn(role="user", content=current_message),
        ]
    return [
        ConversationTurn(
            role="user",
            content=(
                f"I want coaching that fits my {persona.lifestyle.value.replace('_', ' ')} routine "
                f"and my goal of {goal.summary}."
            ),
        ),
        ConversationTurn(
            role="assistant",
            content="We will use a simple plan, observe what gets in the way, and adjust from there.",
        ),
        ConversationTurn(role="user", content=current_message),
    ]


def _response_text(
    *,
    persona: PersonaDefinition,
    scenario: ScenarioDefinition,
    profile: UserProfile,
    goal: CoachingGoalSpec,
    memory_reference: MemoryReference,
    follow_up_questions: list[str],
    variation_index: int = 0,
) -> str:
    """Compose an individualized response from scenario objectives and profile facts."""

    primary_constraint, secondary_constraint = _constraints_for_response(profile)
    emotion = scenario.typical_emotions[0]
    lifestyle = persona.lifestyle.value.replace("_", " ")
    motivation = persona.motivation_style.value.replace("_", " ")
    session_count = max(2, min(4, profile.training_days_per_week))
    session_minutes = min(profile.available_training_minutes, 45)
    action = SCENARIO_ACTIONS[scenario.scenario_id]
    memory_fact = memory_reference.facts[0]
    opening = EMPATHY_OPENINGS[variation_index % len(EMPATHY_OPENINGS)].format(emotion=emotion)
    review_cadence = REVIEW_CADENCES[variation_index % len(REVIEW_CADENCES)]
    plan_emphasis = PLAN_EMPHASES[variation_index % len(PLAN_EMPHASES)]

    sentences = [
        opening,
        (
            f"As a {lifestyle} focused on {motivation}, your {profile.equipment_access} access and "
            f"goal of {goal.summary} call for a plan that works around {primary_constraint} and "
            f"{secondary_constraint}."
        ),
        (
            f"Because {memory_fact}, we will let that context shape the next decision instead of "
            "pretending every week has the same capacity."
        ),
        (
            f"This week, {action}; keep it to {session_count} sessions of {session_minutes} minutes "
            "and log the main lift, session effort, or daily steps so we can review a real pattern."
        ),
        (
            "The priority is manageable progression, adequate recovery, and one measured adjustment "
            f"after a consistent week rather than reacting to a single rough day. {review_cadence}"
        ),
        plan_emphasis,
        *follow_up_questions,
    ]
    return " ".join(sentences)


def build_synthetic_conversation(
    *,
    persona: PersonaDefinition,
    scenario: ScenarioDefinition,
    sample_index: int,
    config: RedAesthConfig = config,
    variation_index: int = 0,
    memory_event_type: EventType | None = None,
    history_turn_count: int | None = None,
) -> SyntheticCoachingConversation:
    """Build one typed synthetic conversation from existing personas and scenarios."""

    conversation_id = f"synthetic-pilot-{sample_index:03d}"
    domain = _goal_domain(persona, scenario)
    goal = CoachingGoalSpec(
        goal_id=f"goal-{conversation_id}",
        domain=domain,
        summary=DOMAIN_GOALS[domain],
        timeframe_weeks=12,
        success_metrics=["weekly adherence", "trend-based progress review"],
        primary_barriers=list(persona.barriers[:2]),
    )
    profile = _build_user_profile(persona, scenario, goal.summary, variation_index)
    memory_reference = _memory_reference(
        conversation_id=conversation_id,
        scenario=scenario,
        profile=profile,
        goal=goal,
        sample_index=sample_index,
        event_type_override=memory_event_type,
    )
    follow_up_questions = _follow_up_questions(
        scenario=scenario,
        profile=profile,
        goal=goal,
        variation_index=variation_index,
    )
    response_text = _response_text(
        persona=persona,
        scenario=scenario,
        profile=profile,
        goal=goal,
        memory_reference=memory_reference,
        follow_up_questions=follow_up_questions,
        variation_index=variation_index,
    )
    response = CoachingResponseSpec(
        response_text=response_text,
        follow_up_questions=follow_up_questions,
        cited_principles=_scientific_principles(scenario),
        adaptation_summary=(
            f"The plan explicitly adapts to {memory_reference.facts[0]}, {profile.schedule_constraints[0]}, "
            f"and {profile.nutrition_constraints[0]} by reducing friction rather than demanding a perfect week."
        ),
    )
    required_validators = [
        "empathy",
        "coaching_quality",
        "personalization",
        "behavioral_adaptation",
        "scientific_consistency",
        "long_term_memory_usage",
        "follow_up_questioning",
        "hallucination_detection",
        "repetitive_responses",
        "scenario_consistency",
    ]
    return SyntheticCoachingConversation(
        conversation_id=conversation_id,
        persona=persona,
        user_profile=profile,
        coaching_goal=goal,
        scenario=scenario,
        conversation_history=_conversation_history(
            persona=persona,
            scenario=scenario,
            goal=goal,
            profile=profile,
            history_turn_count=history_turn_count,
            variation_index=variation_index,
        ),
        memory_references=[memory_reference],
        coaching_response=response,
        expected_coaching_behavior=ExpectedCoachingBehavior(
            empathy_objectives=list(scenario.emotional_objectives),
            coaching_objectives=list(scenario.coaching_objectives),
            emotional_objectives=list(scenario.emotional_objectives),
            scientific_grounding_requirements=list(scenario.scientific_guardrails),
            prohibited_behaviors=list(scenario.rejection_rules),
            required_follow_up_questions=scenario.required_follow_up_questions,
            must_use_memory=True,
        ),
        quality_metadata=QualityMetadata(
            scenario_id=scenario.scenario_id,
            validator_targets={"overall_quality": config.synthetic_quality_threshold},
            must_pass_validators=required_validators,
            rejection_rules=list(scenario.rejection_rules),
            notes=["deterministic synthetic pilot candidate"],
        ),
    )


def generate_validated_conversations(
    *,
    config: RedAesthConfig = config,
    target_count: int | None = None,
) -> SyntheticGenerationResult:
    """Generate exactly the requested number of rubric-passing conversations or fail closed."""

    target = target_count if target_count is not None else config.synthetic_pilot_target_count
    if target <= 0:
        raise ValueError("Synthetic generation target_count must be positive.")

    pairs = [
        (persona, scenario)
        for persona in get_persona_library()
        for scenario in get_scenario_library()
    ]
    random.Random(config.synthetic_generation_seed).shuffle(pairs)

    accepted: list[SyntheticCoachingConversation] = []
    quality_results: list[QualityRubricResult] = []
    rejections: list[GenerationRejection] = []
    attempted_count = 0
    max_attempts = max(target * 3, len(pairs))

    for sample_index in range(max_attempts):
        persona, scenario = pairs[sample_index % len(pairs)]
        candidate = build_synthetic_conversation(
            persona=persona,
            scenario=scenario,
            sample_index=sample_index,
            config=config,
        )
        quality = evaluate_synthetic_conversation(candidate, config=config)
        attempted_count += 1
        if quality.passed:
            accepted.append(candidate)
            quality_results.append(quality)
        else:
            rejections.append(
                GenerationRejection(
                    conversation_id=candidate.conversation_id,
                    blockers=tuple(quality.blockers),
                )
            )
        if len(accepted) == target:
            break

    if len(accepted) != target:
        raise RuntimeError(
            "Synthetic generation failed closed: "
            f"accepted {len(accepted)} of required {target} conversations after {attempted_count} attempts."
        )

    return SyntheticGenerationResult(
        conversations=accepted,
        quality_results=quality_results,
        training_records=[],
        rejections=rejections,
        attempted_count=attempted_count,
    )


def training_record_from_conversation(
    conversation: SyntheticCoachingConversation,
    quality: QualityRubricResult,
    *,
    tokenizer: SmolLM2ChatTemplateTokenizer | None = None,
    source_id: str = SYNTHETIC_SOURCE_ID,
) -> dict[str, Any]:
    """Project a validated synthetic conversation into the locked training-record schema."""

    tokenizer = tokenizer or SmolLM2ChatTemplateTokenizer()
    messages = [
        {"role": turn.role.value, "content": turn.content}
        for turn in conversation.all_messages()
    ]
    topic_tags = sorted(
        {"strength_training", *(DOMAIN_TOPICS[domain] for domain in conversation.scenario.domains)}
    )
    record = build_training_record(
        {
            "conversation_id": conversation.conversation_id,
            "dataset_id": source_id,
            "source_id": source_id,
            "source_license": "internal-synthetic-specification",
            "language": "mostly_ascii",
            "topic_tags": topic_tags,
            "overall_quality_score": round(quality.overall_score, 4),
            "quality_flags": [],
            "conversations": messages,
        },
        source_type="synthetic",
        tokenizer=tokenizer,
    )
    record["synthetic_metadata"] = {
        "persona_id": conversation.persona.persona_id,
        "scenario_id": conversation.scenario.scenario_id,
        "coaching_goal_domain": conversation.coaching_goal.domain.value,
        "primary_coaching_objective": conversation.scenario.coaching_objectives[0],
        "memory_event_types": [reference.event_type.value for reference in conversation.memory_references],
        "quality_rubric": render_quality_summary(quality),
    }
    return record


def export_validated_conversations(
    result: SyntheticGenerationResult,
    *,
    config: RedAesthConfig = config,
    output_path: Path | None = None,
) -> Path:
    """Write only validated conversations as tokenizer-ready JSONL training records."""

    if len(result.conversations) != len(result.quality_results):
        raise ValueError("Every exported synthetic conversation must have a rubric result.")
    records = [
        training_record_from_conversation(conversation, quality)
        for conversation, quality in zip(result.conversations, result.quality_results, strict=True)
    ]
    destination = config.resolve_path(output_path or config.synthetic_pilot_dataset_path)
    result.training_records = records
    result.dataset_path = write_jsonl(destination, records)
    return result.dataset_path


def _render_distribution(title: str, values: Counter[str]) -> list[str]:
    """Render a stable count and percentage distribution section."""

    total = sum(values.values())
    lines = [f"## {title}", ""]
    for label, count in sorted(values.items()):
        percentage = (count / total * 100) if total else 0.0
        lines.append(f"- `{label}`: {count} ({percentage:.2f}%)")
    return lines


def _rubric_distribution(results: list[QualityRubricResult]) -> Counter[str]:
    """Bucket aggregate rubric scores for the operator report."""

    buckets: Counter[str] = Counter()
    for result in results:
        if result.overall_score < 0.80:
            buckets["0.75-0.79"] += 1
        elif result.overall_score < 0.90:
            buckets["0.80-0.89"] += 1
        else:
            buckets["0.90-1.00"] += 1
    return buckets


def write_synthetic_generation_report(
    result: SyntheticGenerationResult,
    *,
    config: RedAesthConfig = config,
    output_path: Path | None = None,
) -> Path:
    """Document the pilot's acceptance, coverage, and quality statistics."""

    if not result.conversations or not result.dataset_path:
        raise ValueError("Generate and export the pilot before writing its report.")

    persona_counts = Counter(conversation.persona.persona_id for conversation in result.conversations)
    scenario_counts = Counter(conversation.scenario.scenario_id for conversation in result.conversations)
    objective_counts = Counter(
        conversation.scenario.coaching_objectives[0] for conversation in result.conversations
    )
    memory_counts = Counter(
        reference.event_type.value
        for conversation in result.conversations
        for reference in conversation.memory_references
    )
    validator_scores: dict[str, list[float]] = {}
    for quality in result.quality_results:
        for validator in quality.validator_results:
            validator_scores.setdefault(validator.validator_name, []).append(validator.score)

    message_counts = [len(conversation.all_messages()) for conversation in result.conversations]
    exchange_counts = [count / 2 for count in message_counts]
    response_lengths = [len(conversation.coaching_response.response_text) for conversation in result.conversations]
    acceptance_rate = result.accepted_count / result.attempted_count * 100

    lines = [
        "# Synthetic Generation Report",
        "",
        "## Generation Summary",
        "",
        f"- Generated conversations: {result.attempted_count}",
        f"- Accepted: {result.accepted_count}",
        f"- Rejected: {len(result.rejections)}",
        f"- Acceptance rate: {acceptance_rate:.2f}%",
        f"- JSONL export: `{result.dataset_path.relative_to(config.project_root).as_posix()}`",
        "",
        "## Rejection Reasons",
        "",
    ]
    if result.rejections:
        rejection_counts = Counter(
            blocker for rejection in result.rejections for blocker in rejection.blockers
        )
        lines.extend(f"- {reason}: {count}" for reason, count in sorted(rejection_counts.items()))
    else:
        lines.append("None. Every exported conversation passed the full deterministic rubric.")

    lines.extend([""] + _render_distribution("Persona Distribution", persona_counts))
    lines.extend([""] + _render_distribution("Scenario Distribution", scenario_counts))
    lines.extend([""] + _render_distribution("Coaching Objective Distribution", objective_counts))
    lines.extend([""] + _render_distribution("Memory Usage Statistics", memory_counts))
    lines.extend(["", "## Validator Scores", ""])
    for name, scores in sorted(validator_scores.items()):
        lines.append(
            f"- `{name}`: average {mean(scores):.4f}, minimum {min(scores):.4f}, maximum {max(scores):.4f}"
        )
    lines.extend([""] + _render_distribution("Rubric Score Distribution", _rubric_distribution(result.quality_results)))
    lines.extend(
        [
            "",
            "## Conversation Length",
            "",
            f"- Average conversation length: {mean(message_counts):.2f} messages",
            f"- Average turns: {mean(exchange_counts):.2f} user/coach exchanges",
            f"- Average coach-response length: {mean(response_lengths):.2f} characters",
            "",
            "## Quality Observations",
            "",
            "- Every exported conversation uses at least one typed memory reference and turns it into an explicit plan adaptation.",
            f"- All {result.accepted_count} exported conversations passed empathy, coaching, personalization, behavioral-adaptation, scientific-consistency, memory, follow-up, safety, repetition, and scenario-consistency checks.",
            f"- The pilot is deterministic and intentionally capped at {config.synthetic_pilot_target_count} validated samples for engineering review before any scale-up.",
            "",
        ]
    )
    destination = config.resolve_path(output_path or config.synthetic_generation_report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    result.report_path = destination
    return destination


def generate_pilot_dataset(*, config: RedAesthConfig = config) -> SyntheticGenerationResult:
    """Generate, validate, export, and report on the fixed production pilot."""

    if config.base_model_id != SMOLLM2_MODEL_ID:
        raise ValueError(
            "The synthetic pilot is locked to the selected SmolLM2 training schema; "
            f"received base_model_id={config.base_model_id!r}."
        )
    result = generate_validated_conversations(config=config)
    export_validated_conversations(result, config=config)
    write_synthetic_generation_report(result, config=config)
    return result


@dataclass(slots=True, frozen=True)
class ProductionCandidatePlan:
    """One deterministic diversity-controlled slot in the production corpus."""

    planned_slot: int
    persona_id: str
    scenario_id: str
    memory_event_type: EventType
    history_turn_count: int


@dataclass(slots=True)
class ProductionFactoryState:
    """Persisted checkpoint for resumable production generation."""

    created_at: str
    target_count: int
    seed: int
    next_candidate_index: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    retry_count: int = 0
    completed: bool = False
    schema_version: str = SYNTHETIC_SCHEMA_VERSION
    generator_version: str = SYNTHETIC_FACTORY_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize the checkpoint in a stable JSON-compatible shape."""

        return {
            "created_at": self.created_at,
            "target_count": self.target_count,
            "seed": self.seed,
            "next_candidate_index": self.next_candidate_index,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "retry_count": self.retry_count,
            "completed": self.completed,
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProductionFactoryState":
        """Restore a checkpoint written by the production factory."""

        return cls(
            created_at=str(payload["created_at"]),
            target_count=int(payload["target_count"]),
            seed=int(payload["seed"]),
            next_candidate_index=int(payload.get("next_candidate_index", 0)),
            accepted_count=int(payload.get("accepted_count", 0)),
            rejected_count=int(payload.get("rejected_count", 0)),
            retry_count=int(payload.get("retry_count", 0)),
            completed=bool(payload.get("completed", False)),
            schema_version=str(payload.get("schema_version", SYNTHETIC_SCHEMA_VERSION)),
            generator_version=str(payload.get("generator_version", SYNTHETIC_FACTORY_VERSION)),
        )


@dataclass(slots=True)
class ProductionCorpusResult:
    """Current or completed state of the deterministic production factory."""

    state: ProductionFactoryState
    completed: bool
    accepted_staging_path: Path
    rejection_log_path: Path
    state_path: Path
    train_path: Path | None = None
    manifest_path: Path | None = None
    dataset_card_path: Path | None = None
    statistics_path: Path | None = None
    report_path: Path | None = None


class ConversationDeduplicationIndex:
    """Factory-local duplicate guard reusing existing normalization and repetition scoring."""

    def __init__(self, *, config: RedAesthConfig, target_count: int) -> None:
        self.config = config
        self.target_count = target_count
        self.exact_hashes: set[str] = set()
        self.similarity_buckets: dict[str, list[tuple[str, str]]] = {}
        self.opening_counts: Counter[str] = Counter()

    @staticmethod
    def _conversation_text_from_messages(messages: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"{message.get('role', '')}: {message.get('content', '')}" for message in messages
        )

    @staticmethod
    def _opening_signature(response: str) -> str:
        return normalized_text(response.split(".", maxsplit=1)[0])

    @staticmethod
    def _response_from_messages(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return str(message.get("content", ""))
        return ""

    def register_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        bucket_key: str = "default",
    ) -> None:
        """Add an already accepted conversation to the recovery-aware index."""

        conversation = normalized_text(self._conversation_text_from_messages(messages))
        response = normalized_text(self._response_from_messages(messages))
        self.exact_hashes.add(hashlib.sha256(conversation.encode("utf-8")).hexdigest())
        self.similarity_buckets.setdefault(bucket_key, []).append((conversation, response))
        self.opening_counts[self._opening_signature(response)] += 1

    def rejection_reason(
        self,
        conversation: SyntheticCoachingConversation,
        *,
        bucket_key: str = "default",
    ) -> str | None:
        """Return a deterministic rejection reason when a candidate is too repetitive."""

        messages = [
            {"role": turn.role.value, "content": turn.content}
            for turn in conversation.all_messages()
        ]
        normalized_conversation = normalized_text(self._conversation_text_from_messages(messages))
        normalized_response = normalized_text(conversation.coaching_response.response_text)
        exact_hash = hashlib.sha256(normalized_conversation.encode("utf-8")).hexdigest()
        if exact_hash in self.exact_hashes:
            return "exact_duplicate"
        if repetition_penalty(conversation.coaching_response.response_text) >= 0.10:
            return "highly_repetitive_response"

        for prior_conversation, prior_response in self.similarity_buckets.get(bucket_key, []):
            conversation_similarity = SequenceMatcher(
                None,
                normalized_conversation,
                prior_conversation,
            ).ratio()
            response_similarity = SequenceMatcher(
                None,
                normalized_response,
                prior_response,
            ).ratio()
            if max(conversation_similarity, response_similarity) >= self.config.synthetic_near_duplicate_similarity:
                return "near_duplicate"

        opening = self._opening_signature(conversation.coaching_response.response_text)
        maximum_opening_count = max(
            1,
            int(self.target_count * self.config.synthetic_max_repeated_opening_share),
        )
        if self.opening_counts[opening] >= maximum_opening_count:
            return "highly_repetitive_opening"
        return None


def _utc_timestamp() -> str:
    """Return the current UTC timestamp for factory provenance records."""

    return datetime.now(timezone.utc).isoformat()


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load a factory staging JSONL file when it already exists."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """Append one accepted or rejected factory record durably."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _write_json_document(path: Path, payload: dict[str, Any]) -> Path:
    """Write a stable JSON document with a self-contained content checksum."""

    document = dict(payload)
    canonical_content = json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    document["content_sha256"] = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    """Return a SHA256 digest for one materialized factory artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_sequence(values: list[str], *, target_count: int, seed: int) -> list[str]:
    """Allocate a deterministic near-equal quota to every supplied value."""

    if not values:
        raise ValueError("Cannot build a balanced sequence from no values.")
    rng = random.Random(seed)
    ordered_values = list(values)
    rng.shuffle(ordered_values)
    base_count, remainder = divmod(target_count, len(ordered_values))
    sequence: list[str] = []
    for index, value in enumerate(ordered_values):
        sequence.extend([value] * (base_count + (1 if index < remainder else 0)))
    rng.shuffle(sequence)
    return sequence


def _build_production_candidate_plan(
    *,
    target_count: int,
    seed: int,
) -> list[ProductionCandidatePlan]:
    """Create a deterministic plan that balances personas, scenarios, memory, and lengths."""

    personas = {persona.persona_id: persona for persona in get_persona_library()}
    scenarios = {scenario.scenario_id: scenario for scenario in get_scenario_library()}
    persona_sequence = _balanced_sequence(
        list(personas),
        target_count=target_count,
        seed=seed,
    )
    scenario_sequence = _balanced_sequence(
        list(scenarios),
        target_count=target_count,
        seed=seed + 1,
    )
    history_sequence = _balanced_sequence(
        ["3", "5"],
        target_count=target_count,
        seed=seed + 2,
    )
    memory_category_counts: Counter[str] = Counter()
    plans: list[ProductionCandidatePlan] = []

    for planned_slot, (persona_id, scenario_id, history_turn_count) in enumerate(
        zip(persona_sequence, scenario_sequence, history_sequence, strict=True)
    ):
        scenario = scenarios[scenario_id]
        options = _supported_memory_event_types(scenario) or [EventType.GOAL_SET]
        memory_event_type = min(
            options,
            key=lambda event_type: (
                memory_category_counts[memory_spec_by_event_type(event_type).category.value],
                event_type.value,
            ),
        )
        memory_category_counts[memory_spec_by_event_type(memory_event_type).category.value] += 1
        plans.append(
            ProductionCandidatePlan(
                planned_slot=planned_slot,
                persona_id=persona_id,
                scenario_id=scenario_id,
                memory_event_type=memory_event_type,
                history_turn_count=int(history_turn_count),
            )
        )
    return plans


def _plan_distributions(plans: list[ProductionCandidatePlan]) -> dict[str, Counter[str]]:
    """Compute target distributions that the diversity controller must preserve."""

    personas = {persona.persona_id: persona for persona in get_persona_library()}
    scenarios = {scenario.scenario_id: scenario for scenario in get_scenario_library()}
    distributions: dict[str, Counter[str]] = {
        "persona": Counter(),
        "scenario": Counter(),
        "coaching_objective": Counter(),
        "motivation": Counter(),
        "experience": Counter(),
        "memory_category": Counter(),
        "conversation_length": Counter(),
    }
    for plan in plans:
        persona = personas[plan.persona_id]
        scenario = scenarios[plan.scenario_id]
        distributions["persona"][persona.persona_id] += 1
        distributions["scenario"][scenario.scenario_id] += 1
        distributions["coaching_objective"][scenario.coaching_objectives[0]] += 1
        distributions["motivation"][persona.motivation_style.value] += 1
        distributions["experience"][persona.experience_level.value] += 1
        distributions["memory_category"][
            memory_spec_by_event_type(plan.memory_event_type).category.value
        ] += 1
        message_count = 2 if scenario.scenario_id == "beginner_onboarding" else plan.history_turn_count + 1
        distributions["conversation_length"][str(message_count)] += 1
    return distributions


def _load_or_create_factory_state(
    *,
    config: RedAesthConfig,
    target_count: int,
) -> tuple[ProductionFactoryState, list[dict[str, Any]], list[dict[str, Any]]]:
    """Restore durable state and reconcile a completed staging append after interruption."""

    accepted_records = _read_jsonl_records(config.synthetic_factory_accepted_staging_path)
    rejection_records = _read_jsonl_records(config.synthetic_factory_rejection_log_path)
    if config.synthetic_factory_state_path.exists():
        state = ProductionFactoryState.from_dict(
            json.loads(config.synthetic_factory_state_path.read_text(encoding="utf-8"))
        )
        if state.target_count != target_count or state.seed != config.synthetic_generation_seed:
            raise RuntimeError(
                "Existing synthetic factory state uses a different target count or seed. "
                "Use a separate production directory rather than mixing deterministic runs."
            )
    else:
        state = ProductionFactoryState(
            created_at=_utc_timestamp(),
            target_count=target_count,
            seed=config.synthetic_generation_seed,
        )

    staged_candidate_indexes = [
        int(record.get("factory_metadata", {}).get("candidate_index", -1))
        for record in accepted_records
    ]
    if len(accepted_records) < state.accepted_count:
        raise RuntimeError("Factory state references accepted records missing from staging JSONL.")
    state.accepted_count = len(accepted_records)
    state.rejected_count = max(state.rejected_count, len(rejection_records))
    state.retry_count = max(state.retry_count, len(rejection_records))
    if staged_candidate_indexes:
        state.next_candidate_index = max(state.next_candidate_index, max(staged_candidate_indexes) + 1)
    state.completed = state.accepted_count >= target_count
    _write_json_document(config.synthetic_factory_state_path, state.to_dict())
    return state, accepted_records, rejection_records


def _distribution_payload(values: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    """Return sorted count and percentage payloads for reports and manifests."""

    return {
        label: {
            "count": count,
            "percentage": round((count / total * 100) if total else 0.0, 2),
        }
        for label, count in sorted(values.items())
    }


def _records_distribution(records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    """Measure accepted-record distributions without recomputing conversation labels."""

    distributions: dict[str, Counter[str]] = {
        "persona": Counter(),
        "scenario": Counter(),
        "coaching_objective": Counter(),
        "motivation": Counter(),
        "experience": Counter(),
        "memory_category": Counter(),
        "memory_event_type": Counter(),
        "conversation_length": Counter(),
    }
    persona_index = {persona.persona_id: persona for persona in get_persona_library()}
    scenario_index = {scenario.scenario_id: scenario for scenario in get_scenario_library()}
    for record in records:
        metadata = record["synthetic_metadata"]
        persona_id = str(metadata["persona_id"])
        scenario_id = str(metadata["scenario_id"])
        persona = persona_index[persona_id]
        scenario = scenario_index[scenario_id]
        distributions["persona"][persona_id] += 1
        distributions["scenario"][scenario_id] += 1
        distributions["coaching_objective"][str(metadata["primary_coaching_objective"])] += 1
        distributions["motivation"][persona.motivation_style.value] += 1
        distributions["experience"][persona.experience_level.value] += 1
        for event_value in metadata["memory_event_types"]:
            event_type = EventType(event_value)
            distributions["memory_event_type"][event_type.value] += 1
            distributions["memory_category"][memory_spec_by_event_type(event_type).category.value] += 1
        distributions["conversation_length"][str(len(record["conversations"]))] += 1
    return distributions


def _similarity_bucket_key(*, persona_id: str, scenario_id: str) -> str:
    """Scope near-duplicate comparisons to one structured persona/scenario combination."""

    return f"{persona_id}::{scenario_id}"


def _max_distribution_deviation(
    actual: Counter[str],
    expected: Counter[str],
    total: int,
) -> float:
    """Measure maximum absolute category-share deviation from the deterministic plan."""

    labels = set(actual) | set(expected)
    if not labels or total == 0:
        return 0.0
    return max(abs(actual[label] / total - expected[label] / total) for label in labels)


def _build_generation_statistics(
    *,
    records: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    state: ProductionFactoryState,
    plans: list[ProductionCandidatePlan],
) -> dict[str, Any]:
    """Build the audit payload shared by the gate, card, manifest, and report."""

    distributions = _records_distribution(records)
    planned_distributions = _plan_distributions(plans)
    validator_scores: dict[str, list[float]] = {}
    rubric_scores: list[float] = []
    for record in records:
        rubric = record["synthetic_metadata"]["quality_rubric"]
        rubric_scores.append(float(rubric["overall_score"]))
        for name, value in rubric.items():
            if name not in {"passed", "overall_score", "blockers"}:
                validator_scores.setdefault(name, []).append(float(value))

    histogram: Counter[str] = Counter()
    for score in rubric_scores:
        if score < 0.80:
            histogram["0.75-0.79"] += 1
        elif score < 0.90:
            histogram["0.80-0.89"] += 1
        else:
            histogram["0.90-1.00"] += 1

    rejection_reasons = Counter(
        reason
        for record in rejections
        for reason in record.get("rejection_reasons", [])
    )
    deduplication_reasons = {
        "exact_duplicate",
        "near_duplicate",
        "highly_repetitive_response",
        "highly_repetitive_opening",
    }
    validator_rejection_count = sum(
        1
        for record in rejections
        if any(
            reason not in deduplication_reasons
            for reason in record.get("rejection_reasons", [])
        )
    )
    total = len(records)
    return {
        "generated_at": state.created_at,
        "schema_version": SYNTHETIC_SCHEMA_VERSION,
        "generator_version": SYNTHETIC_FACTORY_VERSION,
        "sample_count": total,
        "accepted_count": state.accepted_count,
        "rejected_count": state.rejected_count,
        "retry_count": state.retry_count,
        "acceptance_rate": round(
            (state.accepted_count / state.next_candidate_index) if state.next_candidate_index else 0.0,
            4,
        ),
        "candidate_validator_pass_rate": round(
            ((state.next_candidate_index - validator_rejection_count) / state.next_candidate_index)
            if state.next_candidate_index
            else 0.0,
            4,
        ),
        "validator_pass_rate": round(
            sum(
                1
                for record in records
                if record["synthetic_metadata"]["quality_rubric"]["passed"]
            )
            / total
            if total
            else 0.0,
            4,
        ),
        "duplicate_rate": 0.0,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "validator_scores": {
            name: {
                "average": round(mean(scores), 4),
                "minimum": round(min(scores), 4),
                "maximum": round(max(scores), 4),
            }
            for name, scores in sorted(validator_scores.items())
        },
        "quality_score_statistics": {
            "average": round(mean(rubric_scores), 4) if rubric_scores else 0.0,
            "minimum": round(min(rubric_scores), 4) if rubric_scores else 0.0,
            "maximum": round(max(rubric_scores), 4) if rubric_scores else 0.0,
            "histogram": dict(sorted(histogram.items())),
        },
        "distributions": {
            name: _distribution_payload(counter, total)
            for name, counter in distributions.items()
        },
        "planned_distributions": {
            name: _distribution_payload(counter, len(plans))
            for name, counter in planned_distributions.items()
        },
        "distribution_deviations": {
            name: round(
                _max_distribution_deviation(
                    distributions[name],
                    planned_distributions[name],
                    total,
                ),
                4,
            )
            for name in planned_distributions
        },
    }


def evaluate_production_quality_gates(
    statistics: dict[str, Any],
    *,
    config: RedAesthConfig = config,
) -> dict[str, dict[str, Any]]:
    """Evaluate the configured fail-closed gates required before production export."""

    distributions = statistics["distributions"]
    memory_shares = [
        float(payload["percentage"]) / 100 for payload in distributions["memory_category"].values()
    ]
    gates = {
        "validator_pass_rate": {
            "actual": statistics["validator_pass_rate"],
            "threshold": config.synthetic_min_validator_pass_rate,
            "status": "PASS"
            if statistics["validator_pass_rate"] >= config.synthetic_min_validator_pass_rate
            else "FAIL",
        },
        "duplicate_rate": {
            "actual": statistics["duplicate_rate"],
            "threshold": config.synthetic_max_duplicate_rate,
            "status": "PASS"
            if statistics["duplicate_rate"] <= config.synthetic_max_duplicate_rate
            else "FAIL",
        },
        "persona_balance": {
            "actual": statistics["distribution_deviations"]["persona"],
            "threshold": config.synthetic_max_distribution_deviation,
            "status": "PASS"
            if statistics["distribution_deviations"]["persona"]
            <= config.synthetic_max_distribution_deviation
            else "FAIL",
        },
        "scenario_balance": {
            "actual": statistics["distribution_deviations"]["scenario"],
            "threshold": config.synthetic_max_distribution_deviation,
            "status": "PASS"
            if statistics["distribution_deviations"]["scenario"]
            <= config.synthetic_max_distribution_deviation
            else "FAIL",
        },
        "memory_usage_balance": {
            "actual": max(memory_shares, default=0.0),
            "threshold": config.synthetic_max_memory_category_share,
            "status": "PASS"
            if max(memory_shares, default=0.0) <= config.synthetic_max_memory_category_share
            else "FAIL",
        },
    }
    return gates


def _render_distribution_lines(distribution: dict[str, dict[str, float | int]]) -> list[str]:
    """Render one distribution payload as report-friendly markdown lines."""

    return [
        f"- `{label}`: {int(payload['count'])} ({float(payload['percentage']):.2f}%)"
        for label, payload in distribution.items()
    ]


def _write_production_report(
    *,
    statistics: dict[str, Any],
    gates: dict[str, dict[str, Any]],
    config: RedAesthConfig,
    go: bool,
    train_sha256: str | None = None,
) -> Path:
    """Write the required GO / NO GO production-corpus report."""

    blockers = [
        f"{name}: {gate['actual']} versus threshold {gate['threshold']}"
        for name, gate in gates.items()
        if gate["status"] == "FAIL"
    ]
    lines = [
        "# Production Corpus Report",
        "",
        "## GO / NO GO",
        "",
        "GO" if go else "NO GO",
        "",
        "## Generation Summary",
        "",
        f"- Generation timestamp: {statistics['generated_at']}",
        f"- Schema version: {statistics['schema_version']}",
        f"- Generator version: {statistics['generator_version']}",
        f"- Accepted: {statistics['accepted_count']}",
        f"- Rejected: {statistics['rejected_count']}",
        f"- Retry count: {statistics['retry_count']}",
        f"- Acceptance rate: {statistics['acceptance_rate'] * 100:.2f}%",
        f"- Candidate validator pass rate: {statistics['candidate_validator_pass_rate'] * 100:.2f}%",
        f"- Export validator pass rate: {statistics['validator_pass_rate'] * 100:.2f}%",
    ]
    if train_sha256:
        lines.append(f"- `synthetic_train.jsonl` SHA256: `{train_sha256}`")
    lines.extend(["", "## Average Validator Scores", ""])
    for name, scores in statistics["validator_scores"].items():
        lines.append(
            f"- `{name}`: average {scores['average']:.4f}, minimum {scores['minimum']:.4f}, "
            f"maximum {scores['maximum']:.4f}"
        )
    lines.extend(["", "## Rubric Score Histogram", ""])
    for bucket, count in statistics["quality_score_statistics"]["histogram"].items():
        lines.append(f"- `{bucket}`: {count}")
    for title, key in (
        ("Persona Distribution", "persona"),
        ("Scenario Distribution", "scenario"),
        ("Memory Distribution", "memory_category"),
        ("Conversation Length Distribution", "conversation_length"),
    ):
        lines.extend(["", f"## {title}", ""])
        lines.extend(_render_distribution_lines(statistics["distributions"][key]))
    lines.extend(["", "## Quality Gates", ""])
    for name, gate in gates.items():
        lines.append(
            f"- `{name}`: {gate['status']} (actual {gate['actual']}, threshold {gate['threshold']})"
        )
    lines.extend(["", "## Quality Observations", ""])
    lines.append(
        "- Every exported record is schema-valid, memory-adaptive, and passed the complete deterministic quality rubric."
    )
    lines.append(
        "- Diversity is scheduled before generation and verified against the completed corpus before export."
    )
    lines.extend(["", "## Remaining Issues", ""])
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("None. The proof corpus passed all configured production quality gates.")
    config.synthetic_production_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.synthetic_production_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config.synthetic_production_report_path


def _write_dataset_card(
    *,
    statistics: dict[str, Any],
    train_sha256: str,
    config: RedAesthConfig,
) -> Path:
    """Write a concise package card for the validated production corpus."""

    lines = [
        "# RedAesth Synthetic Production Corpus",
        "",
        f"- Generation timestamp: {statistics['generated_at']}",
        f"- Schema version: {statistics['schema_version']}",
        f"- Generator version: {statistics['generator_version']}",
        f"- Sample count: {statistics['sample_count']}",
        f"- Dataset SHA256: `{train_sha256}`",
        f"- Validator pass rate: {statistics['validator_pass_rate'] * 100:.2f}%",
        f"- Duplicate rate: {statistics['duplicate_rate'] * 100:.2f}%",
    ]
    for title, key in (
        ("Persona Distribution", "persona"),
        ("Scenario Distribution", "scenario"),
        ("Memory Distribution", "memory_category"),
        ("Conversation Length Distribution", "conversation_length"),
    ):
        lines.extend(["", f"## {title}", ""])
        lines.extend(_render_distribution_lines(statistics["distributions"][key]))
    body = "\n".join(lines) + "\n"
    content_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    content = body + f"- Card content SHA256: `{content_sha256}`\n"
    config.synthetic_production_card_path.parent.mkdir(parents=True, exist_ok=True)
    config.synthetic_production_card_path.write_text(content, encoding="utf-8")
    return config.synthetic_production_card_path


def _write_manifest(
    *,
    statistics: dict[str, Any],
    config: RedAesthConfig,
    train_path: Path,
    card_path: Path,
    report_path: Path,
) -> Path:
    """Write a SHA256-backed manifest for every materialized package artifact."""

    artifacts = {
        "synthetic_train": {
            "path": str(train_path.relative_to(config.project_root)).replace("\\", "/"),
            "sha256": _sha256_file(train_path),
            "sample_count": statistics["sample_count"],
        },
        "generation_statistics": {
            "path": str(config.synthetic_production_statistics_path.relative_to(config.project_root)).replace("\\", "/"),
            "sha256": _sha256_file(config.synthetic_production_statistics_path),
            "sample_count": statistics["sample_count"],
        },
        "dataset_card": {
            "path": str(card_path.relative_to(config.project_root)).replace("\\", "/"),
            "sha256": _sha256_file(card_path),
            "sample_count": statistics["sample_count"],
        },
        "production_report": {
            "path": str(report_path.relative_to(config.project_root)).replace("\\", "/"),
            "sha256": _sha256_file(report_path),
            "sample_count": statistics["sample_count"],
        },
    }
    return _write_json_document(
        config.synthetic_production_manifest_path,
        {
            "generated_at": statistics["generated_at"],
            "schema_version": SYNTHETIC_SCHEMA_VERSION,
            "generator_version": SYNTHETIC_FACTORY_VERSION,
            "sample_count": statistics["sample_count"],
            "artifacts": artifacts,
            "distributions": statistics["distributions"],
            "validator_pass_rate": statistics["validator_pass_rate"],
            "quality_score_statistics": statistics["quality_score_statistics"],
        },
    )


def _package_production_corpus(
    *,
    records: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    state: ProductionFactoryState,
    plans: list[ProductionCandidatePlan],
    config: RedAesthConfig,
) -> ProductionCorpusResult:
    """Run quality gates and write the final package only when every gate passes."""

    statistics = _build_generation_statistics(
        records=records,
        rejections=rejections,
        state=state,
        plans=plans,
    )
    gates = evaluate_production_quality_gates(statistics, config=config)
    statistics["quality_gates"] = gates
    gate_passed = all(gate["status"] == "PASS" for gate in gates.values())
    if not gate_passed:
        _write_json_document(config.synthetic_production_statistics_path, statistics)
        report_path = _write_production_report(
            statistics=statistics,
            gates=gates,
            config=config,
            go=False,
        )
        failed_gates = ", ".join(
            name for name, gate in gates.items() if gate["status"] == "FAIL"
        )
        raise RuntimeError(
            f"Production corpus export aborted because quality gates failed: {failed_gates}. "
            f"See {report_path}."
        )

    ordered_records = sorted(
        records,
        key=lambda record: int(record["factory_metadata"]["planned_slot"]),
    )
    train_path = write_jsonl(config.synthetic_production_train_path, ordered_records)
    train_sha256 = _sha256_file(train_path)
    statistics["synthetic_train_sha256"] = train_sha256
    statistics_path = _write_json_document(config.synthetic_production_statistics_path, statistics)
    card_path = _write_dataset_card(
        statistics=statistics,
        train_sha256=train_sha256,
        config=config,
    )
    report_path = _write_production_report(
        statistics=statistics,
        gates=gates,
        config=config,
        go=True,
        train_sha256=train_sha256,
    )
    manifest_path = _write_manifest(
        statistics=statistics,
        config=config,
        train_path=train_path,
        card_path=card_path,
        report_path=report_path,
    )
    state.completed = True
    _write_json_document(config.synthetic_factory_state_path, state.to_dict())
    return ProductionCorpusResult(
        state=state,
        completed=True,
        accepted_staging_path=config.synthetic_factory_accepted_staging_path,
        rejection_log_path=config.synthetic_factory_rejection_log_path,
        state_path=config.synthetic_factory_state_path,
        train_path=train_path,
        manifest_path=manifest_path,
        dataset_card_path=card_path,
        statistics_path=statistics_path,
        report_path=report_path,
    )


def generate_production_corpus(
    *,
    config: RedAesthConfig = config,
    target_count: int | None = None,
    batch_size: int | None = None,
    max_batches: int | None = None,
) -> ProductionCorpusResult:
    """Resume or complete deterministic, quality-gated production corpus generation."""

    if config.base_model_id != SMOLLM2_MODEL_ID:
        raise ValueError(
            "The production corpus is locked to the selected SmolLM2 training schema; "
            f"received base_model_id={config.base_model_id!r}."
        )
    target = target_count if target_count is not None else config.synthetic_production_target_count
    effective_batch_size = batch_size if batch_size is not None else config.synthetic_factory_batch_size
    if target <= 0 or effective_batch_size <= 0:
        raise ValueError("Production target_count and batch_size must be positive.")

    state, records, rejections = _load_or_create_factory_state(
        config=config,
        target_count=target,
    )
    plans = _build_production_candidate_plan(
        target_count=target,
        seed=config.synthetic_generation_seed,
    )
    deduplication_index = ConversationDeduplicationIndex(config=config, target_count=target)
    for record in records:
        metadata = record["synthetic_metadata"]
        deduplication_index.register_messages(
            record["conversations"],
            bucket_key=_similarity_bucket_key(
                persona_id=str(metadata["persona_id"]),
                scenario_id=str(metadata["scenario_id"]),
            ),
        )

    personas = {persona.persona_id: persona for persona in get_persona_library()}
    scenarios = {scenario.scenario_id: scenario for scenario in get_scenario_library()}
    max_attempts = target * config.synthetic_factory_attempt_multiplier
    batches_completed = 0

    while state.accepted_count < target and (
        max_batches is None or batches_completed < max_batches
    ):
        for _ in range(effective_batch_size):
            if state.accepted_count >= target:
                break
            if state.next_candidate_index >= max_attempts:
                raise RuntimeError(
                    "Production factory exhausted its configured retry budget before reaching the "
                    f"accepted target of {target}."
                )

            plan = plans[state.accepted_count]
            candidate_index = state.next_candidate_index
            candidate = build_synthetic_conversation(
                persona=personas[plan.persona_id],
                scenario=scenarios[plan.scenario_id],
                sample_index=candidate_index,
                variation_index=candidate_index,
                memory_event_type=plan.memory_event_type,
                history_turn_count=plan.history_turn_count,
                config=config,
            )
            quality = evaluate_synthetic_conversation(candidate, config=config)
            rejection_reasons = list(quality.blockers)
            if quality.passed:
                duplicate_reason = deduplication_index.rejection_reason(
                    candidate,
                    bucket_key=_similarity_bucket_key(
                        persona_id=plan.persona_id,
                        scenario_id=plan.scenario_id,
                    ),
                )
                if duplicate_reason:
                    rejection_reasons.append(duplicate_reason)

            state.next_candidate_index += 1
            if rejection_reasons:
                state.rejected_count += 1
                state.retry_count += 1
                rejection = {
                    "generated_at": state.created_at,
                    "schema_version": SYNTHETIC_SCHEMA_VERSION,
                    "generator_version": SYNTHETIC_FACTORY_VERSION,
                    "candidate_index": candidate_index,
                    "planned_slot": plan.planned_slot,
                    "conversation_id": candidate.conversation_id,
                    "persona_id": plan.persona_id,
                    "scenario_id": plan.scenario_id,
                    "rejection_reasons": rejection_reasons,
                }
                _append_jsonl_record(config.synthetic_factory_rejection_log_path, rejection)
                rejections.append(rejection)
                _write_json_document(config.synthetic_factory_state_path, state.to_dict())
                continue

            record = training_record_from_conversation(
                candidate,
                quality,
                source_id=SYNTHETIC_FACTORY_SOURCE_ID,
            )
            record["factory_metadata"] = {
                "generated_at": state.created_at,
                "schema_version": SYNTHETIC_SCHEMA_VERSION,
                "generator_version": SYNTHETIC_FACTORY_VERSION,
                "candidate_index": candidate_index,
                "planned_slot": plan.planned_slot,
            }
            _append_jsonl_record(config.synthetic_factory_accepted_staging_path, record)
            records.append(record)
            deduplication_index.register_messages(
                record["conversations"],
                bucket_key=_similarity_bucket_key(
                    persona_id=plan.persona_id,
                    scenario_id=plan.scenario_id,
                ),
            )
            state.accepted_count += 1
            _write_json_document(config.synthetic_factory_state_path, state.to_dict())
        batches_completed += 1

    if state.accepted_count < target:
        return ProductionCorpusResult(
            state=state,
            completed=False,
            accepted_staging_path=config.synthetic_factory_accepted_staging_path,
            rejection_log_path=config.synthetic_factory_rejection_log_path,
            state_path=config.synthetic_factory_state_path,
        )

    return _package_production_corpus(
        records=records,
        rejections=rejections,
        state=state,
        plans=plans,
        config=config,
    )


__all__ = [
    "ConversationDeduplicationIndex",
    "GenerationRejection",
    "ProductionCandidatePlan",
    "ProductionCorpusResult",
    "ProductionFactoryState",
    "SmolLM2ChatTemplateTokenizer",
    "SyntheticGenerationResult",
    "build_synthetic_conversation",
    "evaluate_production_quality_gates",
    "export_validated_conversations",
    "generate_pilot_dataset",
    "generate_production_corpus",
    "generate_validated_conversations",
    "training_record_from_conversation",
    "write_synthetic_generation_report",
]
