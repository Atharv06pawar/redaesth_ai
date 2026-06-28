from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(Enum):
    """Canonical event categories for user memory."""

    WEIGHT_RECORDED = "weight_recorded"
    BODY_FAT_RECORDED = "body_fat_recorded"
    MEASUREMENTS_RECORDED = "measurements_recorded"
    GOAL_SET = "goal_set"
    GOAL_CHANGED = "goal_changed"
    GOAL_ACHIEVED = "goal_achieved"
    WORKOUT_COMPLETED = "workout_completed"
    WORKOUT_MISSED = "workout_missed"
    PERSONAL_RECORD_SET = "personal_record_set"
    PROGRAM_CHANGED = "program_changed"
    DELOAD_STARTED = "deload_started"
    DELOAD_ENDED = "deload_ended"
    NUTRITION_TARGET_SET = "nutrition_target_set"
    NUTRITION_TARGET_UPDATED = "nutrition_target_updated"
    DIET_CHANGED = "diet_changed"
    SUPPLEMENT_STARTED = "supplement_started"
    SUPPLEMENT_STOPPED = "supplement_stopped"
    INJURY_REPORTED = "injury_reported"
    INJURY_RECOVERED = "injury_recovered"
    SLEEP_PATTERN_CHANGED = "sleep_pattern_changed"
    MEDICAL_NOTE_ADDED = "medical_note_added"
    TRAVEL_STARTED = "travel_started"
    TRAVEL_ENDED = "travel_ended"
    SCHEDULE_CHANGED = "schedule_changed"
    STRESS_ELEVATED = "stress_elevated"
    STRESS_RESOLVED = "stress_resolved"
    MILESTONE_CELEBRATED = "milestone_celebrated"
    PLATEAU_IDENTIFIED = "plateau_identified"
    PLATEAU_RESOLVED = "plateau_resolved"
    COACH_NOTE_ADDED = "coach_note_added"


@dataclass(slots=True)
class WeightRecordedPayload:
    """A bodyweight measurement captured from the user or a device."""

    weight_kg: float
    measured_at: datetime | None = None
    note: str | None = None


@dataclass(slots=True)
class BodyFatRecordedPayload:
    """A body-fat estimate with optional method metadata."""

    body_fat_pct: float
    method: str | None = None
    measured_at: datetime | None = None


@dataclass(slots=True)
class MeasurementsRecordedPayload:
    """Tape or circumference measurements relevant to physique progress."""

    waist_cm: float | None = None
    chest_cm: float | None = None
    hips_cm: float | None = None
    arms_cm: float | None = None
    thighs_cm: float | None = None
    note: str | None = None


@dataclass(slots=True)
class GoalSetPayload:
    """A newly established user goal."""

    goal: str
    target_date: datetime | None = None
    target_value: str | None = None
    rationale: str | None = None


@dataclass(slots=True)
class GoalChangedPayload:
    """A change from a prior goal to a new one."""

    previous_goal: str | None
    new_goal: str
    reason: str | None = None


@dataclass(slots=True)
class GoalAchievedPayload:
    """A user goal that has been reached."""

    goal: str
    achieved_value: str | None = None
    celebration_note: str | None = None


@dataclass(slots=True)
class WorkoutCompletedPayload:
    """A completed training session."""

    session_name: str | None = None
    program_name: str | None = None
    duration_minutes: int | None = None
    exercises: list[str] = field(default_factory=list)
    perceived_exertion: float | None = None
    note: str | None = None


@dataclass(slots=True)
class WorkoutMissedPayload:
    """A scheduled workout that did not happen."""

    session_name: str | None = None
    reason: str | None = None
    rescheduled_for: datetime | None = None


@dataclass(slots=True)
class PersonalRecordSetPayload:
    """A new best performance on a lift or performance metric."""

    metric: str
    value: float
    unit: str
    previous_best: float | None = None
    note: str | None = None


@dataclass(slots=True)
class ProgramChangedPayload:
    """A switch from one training program structure to another."""

    previous_program: str | None
    new_program: str
    reason: str | None = None


@dataclass(slots=True)
class DeloadStartedPayload:
    """The beginning of a deload block."""

    reason: str | None = None
    expected_end_date: datetime | None = None


@dataclass(slots=True)
class DeloadEndedPayload:
    """The completion of a deload block."""

    outcome: str | None = None
    note: str | None = None


@dataclass(slots=True)
class NutritionTargetSetPayload:
    """Initial calorie or macro targets."""

    calories: int | None = None
    protein_g: int | None = None
    carbs_g: int | None = None
    fat_g: int | None = None
    rationale: str | None = None


@dataclass(slots=True)
class NutritionTargetUpdatedPayload:
    """An update to calorie or macro targets."""

    previous_calories: int | None = None
    new_calories: int | None = None
    previous_protein_g: int | None = None
    new_protein_g: int | None = None
    reason: str | None = None


@dataclass(slots=True)
class DietChangedPayload:
    """A change in dietary approach."""

    previous_diet: str | None
    new_diet: str
    reason: str | None = None


@dataclass(slots=True)
class SupplementStartedPayload:
    """A supplement the user began taking."""

    supplement_name: str
    dose: str | None = None
    reason: str | None = None


@dataclass(slots=True)
class SupplementStoppedPayload:
    """A supplement the user stopped taking."""

    supplement_name: str
    reason: str | None = None


@dataclass(slots=True)
class InjuryReportedPayload:
    """An active injury or pain point the user disclosed."""

    body_part: str
    severity: str | None = None
    onset: datetime | None = None
    note: str | None = None


@dataclass(slots=True)
class InjuryRecoveredPayload:
    """A previously active injury that is now recovered."""

    body_part: str
    recovered_at: datetime | None = None
    note: str | None = None


@dataclass(slots=True)
class SleepPatternChangedPayload:
    """A notable shift in sleep behavior or sleep quality."""

    previous_average_hours: float | None = None
    new_average_hours: float | None = None
    reason: str | None = None


@dataclass(slots=True)
class MedicalNoteAddedPayload:
    """A medical clearance, restriction, or clinically relevant note."""

    note: str
    clinician: str | None = None
    restriction: str | None = None


@dataclass(slots=True)
class TravelStartedPayload:
    """The beginning of travel that may affect routine."""

    destination: str | None = None
    expected_end_date: datetime | None = None
    training_constraints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TravelEndedPayload:
    """The end of a travel period."""

    destination: str | None = None
    return_note: str | None = None


@dataclass(slots=True)
class ScheduleChangedPayload:
    """A change in the user's available training schedule."""

    previous_schedule: str | None
    new_schedule: str
    reason: str | None = None


@dataclass(slots=True)
class StressElevatedPayload:
    """An increase in perceived life or work stress."""

    source: str | None = None
    severity: str | None = None
    note: str | None = None


@dataclass(slots=True)
class StressResolvedPayload:
    """A previously elevated stressor that has improved."""

    source: str | None = None
    resolution_note: str | None = None


@dataclass(slots=True)
class MilestoneCelebratedPayload:
    """A meaningful success worth surfacing later."""

    title: str
    details: str | None = None


@dataclass(slots=True)
class PlateauIdentifiedPayload:
    """A sustained stall in progress."""

    domain: str
    duration_days: int | None = None
    note: str | None = None


@dataclass(slots=True)
class PlateauResolvedPayload:
    """A previously identified plateau that has been broken."""

    domain: str
    resolution: str | None = None


@dataclass(slots=True)
class CoachNoteAddedPayload:
    """A durable coach-authored note that should shape future guidance."""

    note: str
    priority: str | None = None


EVENT_PAYLOAD_SCHEMAS: dict[EventType, type[Any]] = {
    EventType.WEIGHT_RECORDED: WeightRecordedPayload,
    EventType.BODY_FAT_RECORDED: BodyFatRecordedPayload,
    EventType.MEASUREMENTS_RECORDED: MeasurementsRecordedPayload,
    EventType.GOAL_SET: GoalSetPayload,
    EventType.GOAL_CHANGED: GoalChangedPayload,
    EventType.GOAL_ACHIEVED: GoalAchievedPayload,
    EventType.WORKOUT_COMPLETED: WorkoutCompletedPayload,
    EventType.WORKOUT_MISSED: WorkoutMissedPayload,
    EventType.PERSONAL_RECORD_SET: PersonalRecordSetPayload,
    EventType.PROGRAM_CHANGED: ProgramChangedPayload,
    EventType.DELOAD_STARTED: DeloadStartedPayload,
    EventType.DELOAD_ENDED: DeloadEndedPayload,
    EventType.NUTRITION_TARGET_SET: NutritionTargetSetPayload,
    EventType.NUTRITION_TARGET_UPDATED: NutritionTargetUpdatedPayload,
    EventType.DIET_CHANGED: DietChangedPayload,
    EventType.SUPPLEMENT_STARTED: SupplementStartedPayload,
    EventType.SUPPLEMENT_STOPPED: SupplementStoppedPayload,
    EventType.INJURY_REPORTED: InjuryReportedPayload,
    EventType.INJURY_RECOVERED: InjuryRecoveredPayload,
    EventType.SLEEP_PATTERN_CHANGED: SleepPatternChangedPayload,
    EventType.MEDICAL_NOTE_ADDED: MedicalNoteAddedPayload,
    EventType.TRAVEL_STARTED: TravelStartedPayload,
    EventType.TRAVEL_ENDED: TravelEndedPayload,
    EventType.SCHEDULE_CHANGED: ScheduleChangedPayload,
    EventType.STRESS_ELEVATED: StressElevatedPayload,
    EventType.STRESS_RESOLVED: StressResolvedPayload,
    EventType.MILESTONE_CELEBRATED: MilestoneCelebratedPayload,
    EventType.PLATEAU_IDENTIFIED: PlateauIdentifiedPayload,
    EventType.PLATEAU_RESOLVED: PlateauResolvedPayload,
    EventType.COACH_NOTE_ADDED: CoachNoteAddedPayload,
}


@dataclass(slots=True)
class Event:
    """An append-only memory event extracted from user interactions."""

    event_id: str
    user_id: str
    event_type: EventType
    timestamp: datetime
    payload: dict[str, Any]
    source: str
    confidence: float
    conversation_id: str | None = None
    tags: list[str] = field(default_factory=list)
    superseded_by: str | None = None
