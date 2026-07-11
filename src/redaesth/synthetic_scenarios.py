"""Structured scenario library for synthetic coaching dataset generation."""

from __future__ import annotations

from memory.event_schema import EventType

from .synthetic_schema import CoachingDomain, ScenarioDefinition


SCENARIO_LIBRARY: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(
        scenario_id="beginner_onboarding",
        title="Beginner Onboarding",
        summary="The user is starting training and needs a low-friction first plan.",
        domains=[CoachingDomain.GENERAL_FITNESS, CoachingDomain.ADHERENCE],
        typical_emotions=["uncertain", "hopeful", "intimidated"],
        coaching_objectives=[
            "reduce overwhelm",
            "define the first training week",
            "set an adherence-friendly starting point",
        ],
        emotional_objectives=["build confidence", "normalize being a beginner"],
        allowed_memory_event_types=[EventType.GOAL_SET, EventType.SCHEDULE_CHANGED],
        required_follow_up_questions=2,
        response_keywords=["start", "simple", "week", "gym", "confidence"],
        scientific_guardrails=["begin with manageable volume", "avoid advanced specialization"],
        rejection_rules=["do not prescribe maximal training volume", "do not shame the user"],
    ),
    ScenarioDefinition(
        scenario_id="fat_loss_planning",
        title="Fat Loss Planning",
        summary="The user wants sustainable fat loss without aggressive restriction.",
        domains=[CoachingDomain.FAT_LOSS, CoachingDomain.NUTRITION],
        typical_emotions=["motivated", "impatient"],
        coaching_objectives=["set a moderate deficit", "protect adherence", "preserve protein intake"],
        emotional_objectives=["keep expectations realistic"],
        allowed_memory_event_types=[EventType.GOAL_SET, EventType.NUTRITION_TARGET_SET],
        required_follow_up_questions=1,
        response_keywords=["protein", "calories", "steps", "consistent"],
        scientific_guardrails=["recommend moderate deficits", "avoid crash-diet advice"],
        rejection_rules=["do not recommend starvation diets"],
    ),
    ScenarioDefinition(
        scenario_id="muscle_gain_support",
        title="Muscle Gain Support",
        summary="The user wants to gain muscle while balancing recovery and food intake.",
        domains=[CoachingDomain.MUSCLE_GAIN, CoachingDomain.NUTRITION],
        typical_emotions=["eager", "uncertain"],
        coaching_objectives=["set surplus expectations", "prioritize progressive training", "support protein intake"],
        emotional_objectives=["reduce fear of gaining unnecessary fat"],
        allowed_memory_event_types=[EventType.GOAL_SET, EventType.PERSONAL_RECORD_SET],
        required_follow_up_questions=1,
        response_keywords=["progressive", "protein", "surplus", "recovery"],
        scientific_guardrails=["protein guidance should be evidence-aligned", "surplus should be modest"],
        rejection_rules=["do not recommend dirty bulking"],
    ),
    ScenarioDefinition(
        scenario_id="plateau_review",
        title="Plateau Review",
        summary="The user feels stuck and needs a structured diagnosis rather than generic motivation.",
        domains=[CoachingDomain.ADHERENCE, CoachingDomain.GENERAL_FITNESS],
        typical_emotions=["frustrated", "discouraged"],
        coaching_objectives=["diagnose plateau causes", "review recovery and adherence", "adjust one lever at a time"],
        emotional_objectives=["acknowledge frustration", "restore agency"],
        allowed_memory_event_types=[EventType.PLATEAU_IDENTIFIED, EventType.SLEEP_PATTERN_CHANGED],
        required_follow_up_questions=2,
        response_keywords=["plateau", "recovery", "adherence", "adjust"],
        scientific_guardrails=["avoid oversimplified stall explanations"],
        rejection_rules=["do not promise immediate breakthroughs"],
    ),
    ScenarioDefinition(
        scenario_id="missed_workouts_reset",
        title="Missed Workouts Reset",
        summary="The user has missed sessions and needs a shame-free restart plan.",
        domains=[CoachingDomain.ADHERENCE],
        typical_emotions=["guilty", "ashamed", "discouraged"],
        coaching_objectives=["decatastrophize the lapse", "restart with a specific next action"],
        emotional_objectives=["reduce guilt", "protect self-efficacy"],
        allowed_memory_event_types=[EventType.WORKOUT_MISSED, EventType.STRESS_ELEVATED],
        required_follow_up_questions=1,
        response_keywords=["restart", "next workout", "small step", "consistency"],
        scientific_guardrails=["focus on behavior restart, not punishment"],
        rejection_rules=["do not use punitive language"],
    ),
    ScenarioDefinition(
        scenario_id="injury_recovery_adjustment",
        title="Injury Recovery Adjustment",
        summary="The user is dealing with pain or a recent injury and needs safe modification.",
        domains=[CoachingDomain.INJURY_RECOVERY, CoachingDomain.RECOVERY],
        typical_emotions=["worried", "frustrated"],
        coaching_objectives=["protect healing tissue", "modify training intelligently", "escalate when needed"],
        emotional_objectives=["reduce fear without dismissing pain"],
        allowed_memory_event_types=[EventType.INJURY_REPORTED, EventType.MEDICAL_NOTE_ADDED],
        required_follow_up_questions=2,
        response_keywords=["pain", "modify", "range", "assessment"],
        scientific_guardrails=["do not diagnose from chat", "avoid push-through-pain advice"],
        rejection_rules=["do not promise a cure", "do not ignore red flags"],
    ),
    ScenarioDefinition(
        scenario_id="poor_sleep_recovery",
        title="Poor Sleep Recovery",
        summary="The user is sleeping poorly and training performance is suffering.",
        domains=[CoachingDomain.RECOVERY, CoachingDomain.ADHERENCE],
        typical_emotions=["tired", "overwhelmed"],
        coaching_objectives=["adjust workload", "protect minimum viable training", "improve recovery habits"],
        emotional_objectives=["validate fatigue"],
        allowed_memory_event_types=[EventType.SLEEP_PATTERN_CHANGED, EventType.STRESS_ELEVATED],
        required_follow_up_questions=1,
        response_keywords=["sleep", "fatigue", "reduce", "recovery"],
        scientific_guardrails=["recommend workload adjustments before adding volume"],
        rejection_rules=["do not ignore chronic fatigue signs"],
    ),
    ScenarioDefinition(
        scenario_id="exam_stress_adjustment",
        title="Exam Stress Adjustment",
        summary="A student is under intense exam pressure and needs a temporary plan adaptation.",
        domains=[CoachingDomain.ADHERENCE, CoachingDomain.RECOVERY],
        typical_emotions=["stressed", "guilty"],
        coaching_objectives=["adapt volume temporarily", "retain momentum", "protect stress management"],
        emotional_objectives=["reduce all-or-nothing thinking"],
        allowed_memory_event_types=[EventType.STRESS_ELEVATED, EventType.SCHEDULE_CHANGED],
        required_follow_up_questions=1,
        response_keywords=["exam", "stress", "short sessions", "momentum"],
        scientific_guardrails=["shorter sessions are acceptable during peak stress"],
        rejection_rules=["do not frame reduced volume as failure"],
    ),
    ScenarioDefinition(
        scenario_id="travel_continuity",
        title="Travel Continuity",
        summary="The user is traveling and needs realistic training continuity.",
        domains=[CoachingDomain.ADHERENCE, CoachingDomain.GENERAL_FITNESS],
        typical_emotions=["disrupted", "determined"],
        coaching_objectives=["adjust to available equipment", "preserve routine cues"],
        emotional_objectives=["normalize imperfection while traveling"],
        allowed_memory_event_types=[EventType.TRAVEL_STARTED, EventType.SCHEDULE_CHANGED],
        required_follow_up_questions=1,
        response_keywords=["travel", "hotel", "bodyweight", "routine"],
        scientific_guardrails=["prioritize feasibility over optimization"],
        rejection_rules=["do not recommend all-or-nothing cancellation"],
    ),
    ScenarioDefinition(
        scenario_id="busy_professional_compliance",
        title="Busy Professional Compliance",
        summary="The user has limited bandwidth and needs efficient, high-compliance coaching.",
        domains=[CoachingDomain.ADHERENCE, CoachingDomain.GENERAL_FITNESS],
        typical_emotions=["busy", "stretched", "frustrated"],
        coaching_objectives=["simplify decisions", "prioritize efficient sessions", "use schedule-aware coaching"],
        emotional_objectives=["acknowledge time pressure"],
        allowed_memory_event_types=[EventType.SCHEDULE_CHANGED, EventType.STRESS_ELEVATED],
        required_follow_up_questions=1,
        response_keywords=["schedule", "efficient", "minimum effective", "prioritize"],
        scientific_guardrails=["recommend the minimum effective dose when needed"],
        rejection_rules=["do not prescribe complex meal or training plans"],
    ),
    ScenarioDefinition(
        scenario_id="returning_after_break",
        title="Returning After a Break",
        summary="The user is re-entering training after a layoff and needs re-ramping guidance.",
        domains=[CoachingDomain.GENERAL_FITNESS, CoachingDomain.ADHERENCE],
        typical_emotions=["nervous", "hopeful"],
        coaching_objectives=["rebuild gradually", "avoid comparing to old performance"],
        emotional_objectives=["reduce fear of lost progress"],
        allowed_memory_event_types=[EventType.PROGRAM_CHANGED, EventType.WORKOUT_MISSED],
        required_follow_up_questions=1,
        response_keywords=["return", "gradual", "baseline", "rebuild"],
        scientific_guardrails=["reintroduce volume progressively"],
        rejection_rules=["do not push for previous peak numbers immediately"],
    ),
    ScenarioDefinition(
        scenario_id="competition_preparation",
        title="Competition Preparation",
        summary="The user is preparing for an event and needs precise, fatigue-aware coaching.",
        domains=[CoachingDomain.MUSCLE_GAIN, CoachingDomain.RECOVERY],
        typical_emotions=["focused", "fatigued", "anxious"],
        coaching_objectives=["balance specificity with recovery", "manage prep fatigue", "protect execution quality"],
        emotional_objectives=["support discipline without glorifying burnout"],
        allowed_memory_event_types=[EventType.PROGRAM_CHANGED, EventType.DELOAD_STARTED, EventType.PERSONAL_RECORD_SET],
        required_follow_up_questions=2,
        response_keywords=["prep", "specificity", "fatigue", "peak"],
        scientific_guardrails=["do not romanticize overtraining", "support recovery decisions"],
        rejection_rules=["do not encourage reckless dehydration or extreme volume"],
    ),
)


def get_scenario_library() -> tuple[ScenarioDefinition, ...]:
    """Return the complete scenario library."""

    return SCENARIO_LIBRARY


def scenario_by_id(scenario_id: str) -> ScenarioDefinition:
    """Look up one scenario by identifier."""

    for scenario in SCENARIO_LIBRARY:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise KeyError(f"Unknown scenario_id: {scenario_id}")


def validate_scenario_library(scenarios: tuple[ScenarioDefinition, ...] | None = None) -> None:
    """Validate scenario coverage and identifier uniqueness."""

    scenarios = scenarios or SCENARIO_LIBRARY
    if not scenarios:
        raise ValueError("Scenario library must not be empty.")

    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Scenario IDs must be unique.")

    required_ids = {
        "beginner_onboarding",
        "fat_loss_planning",
        "muscle_gain_support",
        "plateau_review",
        "missed_workouts_reset",
        "injury_recovery_adjustment",
        "poor_sleep_recovery",
        "exam_stress_adjustment",
        "travel_continuity",
        "busy_professional_compliance",
        "returning_after_break",
        "competition_preparation",
    }
    if required_ids.difference(scenario_ids):
        missing = ", ".join(sorted(required_ids.difference(scenario_ids)))
        raise ValueError(f"Scenario library is missing required scenarios: {missing}")
