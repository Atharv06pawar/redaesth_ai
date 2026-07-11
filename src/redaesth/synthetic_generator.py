"""Deterministic, validator-gated synthetic coaching conversation generation."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from memory.event_schema import EventType

from .config import RedAesthConfig, config
from .final_dataset import build_training_record, write_jsonl
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
) -> MemoryReference:
    """Create one concrete, behavior-shaping memory reference for a candidate."""

    allowed_types = _supported_memory_event_types(scenario)
    event_type = (
        allowed_types[sample_index % len(allowed_types)] if allowed_types else EventType.GOAL_SET
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
) -> list[str]:
    """Create the required number of scenario-aware coaching questions."""

    primary_constraint, _ = _constraints_for_response(profile)
    candidates = [
        f"Which part of {primary_constraint} is most likely to disrupt the plan this week?",
        f"What would show that {goal.summary} is moving in the right direction over the next week?",
        f"Which {profile.equipment_access} exercise feels most realistic for your next session?",
    ]
    return candidates[: scenario.required_follow_up_questions]


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
) -> list[ConversationTurn]:
    """Build onboarding or follow-up history from the structured scenario, never a static chat."""

    signal = SCENARIO_USER_SIGNALS[scenario.scenario_id]
    primary_constraint, _ = _constraints_for_response(profile)
    current_message = (
        f"I am working toward {goal.summary}. {signal} With {primary_constraint}, I need a "
        "plan that fits this week."
    )
    if scenario.scenario_id == "beginner_onboarding":
        return [ConversationTurn(role="user", content=current_message)]
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

    sentences = [
        f"I hear you - feeling {emotion} in this situation makes sense.",
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
            "after a consistent week rather than reacting to a single rough day."
        ),
        *follow_up_questions,
    ]
    return " ".join(sentences)


def build_synthetic_conversation(
    *,
    persona: PersonaDefinition,
    scenario: ScenarioDefinition,
    sample_index: int,
    config: RedAesthConfig = config,
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
    profile = _build_user_profile(persona, scenario, goal.summary)
    memory_reference = _memory_reference(
        conversation_id=conversation_id,
        scenario=scenario,
        profile=profile,
        goal=goal,
        sample_index=sample_index,
    )
    follow_up_questions = _follow_up_questions(
        scenario=scenario,
        profile=profile,
        goal=goal,
    )
    response_text = _response_text(
        persona=persona,
        scenario=scenario,
        profile=profile,
        goal=goal,
        memory_reference=memory_reference,
        follow_up_questions=follow_up_questions,
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
            "dataset_id": SYNTHETIC_SOURCE_ID,
            "source_id": SYNTHETIC_SOURCE_ID,
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


__all__ = [
    "GenerationRejection",
    "SmolLM2ChatTemplateTokenizer",
    "SyntheticGenerationResult",
    "build_synthetic_conversation",
    "export_validated_conversations",
    "generate_pilot_dataset",
    "generate_validated_conversations",
    "training_record_from_conversation",
    "write_synthetic_generation_report",
]
