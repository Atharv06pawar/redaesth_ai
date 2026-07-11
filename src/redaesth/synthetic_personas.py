"""Structured persona library for future synthetic coaching conversations."""

from __future__ import annotations

from .synthetic_schema import (
    AdherenceLevel,
    ConfidenceLevel,
    DietStyle,
    ExperienceLevel,
    LifestyleType,
    MotivationStyle,
    PersonaDefinition,
    StressLevel,
    UserProfile,
)


PERSONA_LIBRARY: tuple[PersonaDefinition, ...] = (
    PersonaDefinition(
        persona_id="student_beginner_exam_stress",
        display_name="Aisha",
        age=20,
        lifestyle=LifestyleType.STUDENT,
        experience_level=ExperienceLevel.BEGINNER,
        motivation_style=MotivationStyle.CONFIDENCE,
        schedule_constraints=["exam blocks reduce training time", "campus dining is inconsistent"],
        available_training_days_per_week=3,
        available_training_minutes=45,
        diet_style=DietStyle.VEGETARIAN,
        adherence_level=AdherenceLevel.FRAGILE,
        stress_level=StressLevel.HIGH,
        confidence_level=ConfidenceLevel.LOW,
        lifestyle_notes=["lives in a dorm", "walks between classes daily"],
        preferred_training_modalities=["machines", "short treadmill sessions"],
        barriers=["gym intimidation", "irregular sleep", "studying late"],
    ),
    PersonaDefinition(
        persona_id="busy_fat_loss_professional",
        display_name="Marcus",
        age=37,
        lifestyle=LifestyleType.OFFICE_WORKER,
        experience_level=ExperienceLevel.NOVICE,
        motivation_style=MotivationStyle.HEALTH,
        active_injuries=["intermittent low-back tightness"],
        schedule_constraints=["client meetings spill into evenings", "two school pickups each week"],
        available_training_days_per_week=4,
        available_training_minutes=40,
        diet_style=DietStyle.FLEXIBLE,
        adherence_level=AdherenceLevel.INCONSISTENT,
        stress_level=StressLevel.MODERATE,
        confidence_level=ConfidenceLevel.MODERATE,
        lifestyle_notes=["desk job", "commutes 50 minutes daily"],
        preferred_training_modalities=["dumbbell circuits", "incline walking"],
        barriers=["meal skipping", "weekend overeating"],
    ),
    PersonaDefinition(
        persona_id="muscle_gain_new_parent",
        display_name="Elena",
        age=31,
        lifestyle=LifestyleType.PARENT,
        experience_level=ExperienceLevel.INTERMEDIATE,
        motivation_style=MotivationStyle.AESTHETICS,
        historical_injuries=["past wrist irritation from barbell front rack"],
        schedule_constraints=["newborn sleep interruptions", "training only before work"],
        available_training_days_per_week=3,
        available_training_minutes=50,
        diet_style=DietStyle.HIGH_PROTEIN,
        adherence_level=AdherenceLevel.REBUILDING,
        stress_level=StressLevel.HIGH,
        confidence_level=ConfidenceLevel.MODERATE,
        lifestyle_notes=["prefers home workouts during weekdays"],
        preferred_training_modalities=["adjustable dumbbells", "bench work", "bands"],
        barriers=["fragmented sleep", "reduced appetite after stressful days"],
    ),
    PersonaDefinition(
        persona_id="traveling_sales_runner",
        display_name="Dev",
        age=29,
        lifestyle=LifestyleType.FREQUENT_TRAVELER,
        experience_level=ExperienceLevel.INTERMEDIATE,
        motivation_style=MotivationStyle.PERFORMANCE,
        active_injuries=["mild right Achilles irritation"],
        schedule_constraints=["flies twice a month", "hotel gym access varies"],
        available_training_days_per_week=5,
        available_training_minutes=60,
        diet_style=DietStyle.OMNIVORE,
        adherence_level=AdherenceLevel.MOSTLY_CONSISTENT,
        stress_level=StressLevel.MODERATE,
        confidence_level=ConfidenceLevel.HIGH,
        lifestyle_notes=["trains for half-marathons", "tracks steps daily"],
        preferred_training_modalities=["running", "hotel strength circuits"],
        barriers=["jet lag", "missed long runs"],
    ),
    PersonaDefinition(
        persona_id="returning_after_break_creative",
        display_name="Noah",
        age=27,
        lifestyle=LifestyleType.CREATIVE_FREELANCER,
        experience_level=ExperienceLevel.NOVICE,
        motivation_style=MotivationStyle.ROUTINE,
        historical_injuries=["previous shoulder discomfort during pressing"],
        schedule_constraints=["project deadlines cluster unpredictably"],
        available_training_days_per_week=4,
        available_training_minutes=55,
        diet_style=DietStyle.FLEXIBLE,
        adherence_level=AdherenceLevel.REBUILDING,
        stress_level=StressLevel.MODERATE,
        confidence_level=ConfidenceLevel.LOW,
        lifestyle_notes=["previously trained consistently before a six-month break"],
        preferred_training_modalities=["upper-lower split", "bodyweight finishers"],
        barriers=["fear of losing prior progress", "inconsistent mornings"],
    ),
    PersonaDefinition(
        persona_id="shift_worker_recovery_focus",
        display_name="Priya",
        age=34,
        lifestyle=LifestyleType.SHIFT_WORKER,
        experience_level=ExperienceLevel.NOVICE,
        motivation_style=MotivationStyle.STRESS_RELIEF,
        active_injuries=["recurring neck and upper-back tension"],
        schedule_constraints=["rotating night shifts", "sleep window changes weekly"],
        available_training_days_per_week=3,
        available_training_minutes=35,
        diet_style=DietStyle.OMNIVORE,
        adherence_level=AdherenceLevel.FRAGILE,
        stress_level=StressLevel.HIGH,
        confidence_level=ConfidenceLevel.LOW,
        lifestyle_notes=["hospital nurse", "often relies on cafeteria meals"],
        preferred_training_modalities=["mobility work", "rowing machine", "light resistance training"],
        barriers=["poor sleep", "fatigue after shifts"],
    ),
    PersonaDefinition(
        persona_id="retiree_general_fitness",
        display_name="Helen",
        age=64,
        lifestyle=LifestyleType.RETIREE,
        experience_level=ExperienceLevel.BEGINNER,
        motivation_style=MotivationStyle.HEALTH,
        active_injuries=["knee stiffness when descending stairs"],
        schedule_constraints=["cares for grandchildren twice weekly"],
        available_training_days_per_week=4,
        available_training_minutes=45,
        diet_style=DietStyle.OMNIVORE,
        adherence_level=AdherenceLevel.MOSTLY_CONSISTENT,
        stress_level=StressLevel.LOW,
        confidence_level=ConfidenceLevel.MODERATE,
        lifestyle_notes=["walks in the morning most days"],
        preferred_training_modalities=["machines", "balance drills", "walking"],
        barriers=["fear of aggravating knee discomfort"],
    ),
    PersonaDefinition(
        persona_id="vegan_body_recomp_office_worker",
        display_name="Jordan",
        age=26,
        lifestyle=LifestyleType.OFFICE_WORKER,
        experience_level=ExperienceLevel.INTERMEDIATE,
        motivation_style=MotivationStyle.AESTHETICS,
        schedule_constraints=["works late on Tuesdays and Thursdays"],
        available_training_days_per_week=5,
        available_training_minutes=70,
        diet_style=DietStyle.VEGAN,
        adherence_level=AdherenceLevel.MOSTLY_CONSISTENT,
        stress_level=StressLevel.MODERATE,
        confidence_level=ConfidenceLevel.MODERATE,
        lifestyle_notes=["meal preps on Sundays", "prefers data-driven plans"],
        preferred_training_modalities=["hypertrophy training", "step count targets"],
        barriers=["protein variety fatigue", "social meals on weekends"],
    ),
    PersonaDefinition(
        persona_id="competition_prep_advanced",
        display_name="Sofia",
        age=28,
        lifestyle=LifestyleType.CREATIVE_FREELANCER,
        experience_level=ExperienceLevel.ADVANCED,
        motivation_style=MotivationStyle.PERFORMANCE,
        historical_injuries=["resolved left hip flexor strain"],
        schedule_constraints=["photo shoots alter meal timing"],
        available_training_days_per_week=6,
        available_training_minutes=90,
        diet_style=DietStyle.HIGH_PROTEIN,
        adherence_level=AdherenceLevel.HIGHLY_CONSISTENT,
        stress_level=StressLevel.MODERATE,
        confidence_level=ConfidenceLevel.HIGH,
        lifestyle_notes=["preparing for a physique competition"],
        preferred_training_modalities=["bodybuilding split", "posing practice", "steady-state cardio"],
        barriers=["prep fatigue", "social isolation during cuts"],
    ),
    PersonaDefinition(
        persona_id="muscle_gain_student_confidence_building",
        display_name="Leo",
        age=22,
        lifestyle=LifestyleType.STUDENT,
        experience_level=ExperienceLevel.NOVICE,
        motivation_style=MotivationStyle.CONFIDENCE,
        schedule_constraints=["shared campus gym gets crowded in evenings"],
        available_training_days_per_week=4,
        available_training_minutes=60,
        diet_style=DietStyle.OMNIVORE,
        adherence_level=AdherenceLevel.INCONSISTENT,
        stress_level=StressLevel.MODERATE,
        confidence_level=ConfidenceLevel.LOW,
        lifestyle_notes=["wants to feel less intimidated in the weight room"],
        preferred_training_modalities=["machines", "simple upper-lower split"],
        barriers=["skips meals", "compares self to stronger lifters"],
    ),
)


def get_persona_library() -> tuple[PersonaDefinition, ...]:
    """Return the full synthetic persona library."""

    return PERSONA_LIBRARY


def persona_by_id(persona_id: str) -> PersonaDefinition:
    """Look up one persona by identifier."""

    for persona in PERSONA_LIBRARY:
        if persona.persona_id == persona_id:
            return persona
    raise KeyError(f"Unknown persona_id: {persona_id}")


def build_user_profile_from_persona(
    persona: PersonaDefinition,
    *,
    user_id: str,
    equipment_access: str,
    sleep_hours_average: float,
    priorities: list[str] | None = None,
    current_stats: dict[str, str | float | int] | None = None,
    nutrition_constraints: list[str] | None = None,
) -> UserProfile:
    """Create a typed user profile snapshot from a persona definition."""

    return UserProfile(
        user_id=user_id,
        age=persona.age,
        experience_level=persona.experience_level,
        training_days_per_week=persona.available_training_days_per_week,
        available_training_minutes=persona.available_training_minutes,
        equipment_access=equipment_access,
        diet_style=persona.diet_style,
        adherence_level=persona.adherence_level,
        stress_level=persona.stress_level,
        confidence_level=persona.confidence_level,
        sleep_hours_average=sleep_hours_average,
        active_injuries=list(persona.active_injuries),
        historical_injuries=list(persona.historical_injuries),
        nutrition_constraints=nutrition_constraints or [],
        schedule_constraints=list(persona.schedule_constraints),
        current_stats=current_stats or {},
        priorities=priorities or [],
    )


def validate_persona_library(personas: tuple[PersonaDefinition, ...] | None = None) -> None:
    """Validate library-wide persona coverage and identifier uniqueness."""

    personas = personas or PERSONA_LIBRARY
    if not personas:
        raise ValueError("Persona library must not be empty.")

    persona_ids = [persona.persona_id for persona in personas]
    if len(persona_ids) != len(set(persona_ids)):
        raise ValueError("Persona IDs must be unique.")

    ages = {persona.age for persona in personas}
    lifestyles = {persona.lifestyle for persona in personas}
    experience_levels = {persona.experience_level for persona in personas}
    diet_styles = {persona.diet_style for persona in personas}
    if min(ages) >= 25 or max(ages) <= 55:
        raise ValueError("Persona library should cover both younger and older adults.")
    if len(lifestyles) < 5:
        raise ValueError("Persona library should span at least five lifestyle types.")
    if len(experience_levels) < 4:
        raise ValueError("Persona library should span all experience levels.")
    if len(diet_styles) < 4:
        raise ValueError("Persona library should span multiple diet styles.")
