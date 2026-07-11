"""Memory specification for future synthetic coaching conversations."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from memory.event_schema import EventType


class MemoryPriority(str, Enum):
    """How strongly a memory should shape future coaching behavior."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MemoryCategory(str, Enum):
    """Semantic category for a remembered coaching fact."""

    GOAL = "goal"
    SCHEDULE = "schedule"
    READINESS = "readiness"
    HEALTH = "health"
    NUTRITION = "nutrition"
    PERFORMANCE = "performance"
    CONTEXT = "context"


class MemoryUsageExample(BaseModel):
    """Example of how a memory should or should not be used."""

    label: str
    example: str


class MemoryEventSpecification(BaseModel):
    """Specification for how one memory event should behave in synthetic data."""

    event_type: EventType
    category: MemoryCategory
    priority: MemoryPriority
    creation_rules: list[str] = Field(default_factory=list)
    ignore_rules: list[str] = Field(default_factory=list)
    retrieval_rules: list[str] = Field(default_factory=list)
    behavioral_adaptation_rules: list[str] = Field(default_factory=list)
    invalid_memory_usage: list[str] = Field(default_factory=list)
    expiration_policy: str
    valid_usage_examples: list[MemoryUsageExample] = Field(default_factory=list)
    invalid_usage_examples: list[MemoryUsageExample] = Field(default_factory=list)


MEMORY_EVENT_SPECIFICATIONS: tuple[MemoryEventSpecification, ...] = (
    MemoryEventSpecification(
        event_type=EventType.GOAL_SET,
        category=MemoryCategory.GOAL,
        priority=MemoryPriority.HIGH,
        creation_rules=[
            "Create when the user clearly states a concrete outcome or target.",
            "Create when a deadline, event, or success metric is attached to the goal.",
        ],
        ignore_rules=[
            "Ignore vague wishes without commitment or timeline.",
            "Ignore goals quoted hypothetically or on behalf of someone else.",
        ],
        retrieval_rules=[
            "Retrieve when the coach is planning next steps, reviewing progress, or prioritizing tradeoffs.",
            "Retrieve when recommendations need to be matched to the user's stated outcome.",
        ],
        behavioral_adaptation_rules=[
            "Advice should tie weekly actions back to the stored goal.",
            "The coach should resolve conflicts between the goal and current habits explicitly.",
        ],
        invalid_memory_usage=[
            "Reading the stored goal back as raw schema text.",
            "Using an old goal after a clear goal-change event supersedes it.",
        ],
        expiration_policy="Persists until superseded by GOAL_CHANGED or GOAL_ACHIEVED.",
        valid_usage_examples=[
            MemoryUsageExample(
                label="valid",
                example="You mentioned that the next 12 weeks are focused on fat loss, so let's keep your training dense and your calorie target steady.",
            )
        ],
        invalid_usage_examples=[
            MemoryUsageExample(
                label="invalid",
                example="Memory says goal_set equals fat loss with target date 2026-09-01.",
            )
        ],
    ),
    MemoryEventSpecification(
        event_type=EventType.SCHEDULE_CHANGED,
        category=MemoryCategory.SCHEDULE,
        priority=MemoryPriority.HIGH,
        creation_rules=[
            "Create when the user's recurring availability changes materially.",
            "Create when work, school, caregiving, or travel alters training windows.",
        ],
        ignore_rules=[
            "Ignore one-off delays that do not affect the ongoing schedule.",
        ],
        retrieval_rules=[
            "Retrieve when recommending session frequency, duration, or split structure.",
        ],
        behavioral_adaptation_rules=[
            "Training plans should compress volume or simplify splits when time shrinks.",
            "The coach should avoid recommending schedules the user no longer has access to.",
        ],
        invalid_memory_usage=[
            "Pretending the old schedule still applies after a change.",
            "Pressuring the user to train beyond the newly available windows.",
        ],
        expiration_policy="Persists until replaced by a newer SCHEDULE_CHANGED event.",
        valid_usage_examples=[
            MemoryUsageExample(
                label="valid",
                example="Since your evenings are blocked on Tuesdays and Thursdays now, let's anchor your main lifting days on Monday, Wednesday, and Saturday.",
            )
        ],
        invalid_usage_examples=[
            MemoryUsageExample(
                label="invalid",
                example="You should keep the five-day split even though you told me you only have three sessions now.",
            )
        ],
    ),
    MemoryEventSpecification(
        event_type=EventType.WORKOUT_MISSED,
        category=MemoryCategory.READINESS,
        priority=MemoryPriority.MEDIUM,
        creation_rules=[
            "Create when the user misses a planned workout and gives a real reason or pattern.",
        ],
        ignore_rules=[
            "Ignore speculative statements about maybe missing future workouts.",
        ],
        retrieval_rules=[
            "Retrieve when coaching a restart, adherence reset, or weekly planning conversation.",
        ],
        behavioral_adaptation_rules=[
            "The coach should lower activation energy for the next workout.",
            "Responses should avoid punitive volume make-up advice.",
        ],
        invalid_memory_usage=[
            "Shaming the user with a ledger of missed sessions.",
        ],
        expiration_policy="Decay after two weeks if the user has resumed normal training.",
        valid_usage_examples=[
            MemoryUsageExample(
                label="valid",
                example="Because last week got derailed by work travel, let's make the next session short and easy to restart.",
            )
        ],
        invalid_usage_examples=[
            MemoryUsageExample(
                label="invalid",
                example="You missed two workouts, so you need an extra punishment cardio day.",
            )
        ],
    ),
    MemoryEventSpecification(
        event_type=EventType.INJURY_REPORTED,
        category=MemoryCategory.HEALTH,
        priority=MemoryPriority.HIGH,
        creation_rules=[
            "Create when the user reports pain, injury, or movement limitation affecting training.",
        ],
        ignore_rules=[
            "Ignore minor soreness when the user explicitly distinguishes it from injury.",
        ],
        retrieval_rules=[
            "Retrieve for any programming, exercise substitution, or intensity advice touching the affected area.",
        ],
        behavioral_adaptation_rules=[
            "The coach should modify range, load, exercise selection, or referral guidance.",
            "The coach should not advise pushing through meaningful pain.",
        ],
        invalid_memory_usage=[
            "Diagnosing beyond the available evidence.",
            "Telling the user the injury is irrelevant to current training choices.",
        ],
        expiration_policy="Persists until INJURY_RECOVERED or an updated medical note supersedes it.",
        valid_usage_examples=[
            MemoryUsageExample(
                label="valid",
                example="Because your shoulder is still irritated on presses, let's swap overhead work for pain-free pulling and neutral-grip pressing for now.",
            )
        ],
        invalid_usage_examples=[
            MemoryUsageExample(
                label="invalid",
                example="Ignore the shoulder pain and keep the same pressing load until it adapts.",
            )
        ],
    ),
    MemoryEventSpecification(
        event_type=EventType.SLEEP_PATTERN_CHANGED,
        category=MemoryCategory.READINESS,
        priority=MemoryPriority.MEDIUM,
        creation_rules=[
            "Create when the user's average sleep quality or quantity changes enough to affect recovery.",
        ],
        ignore_rules=[
            "Ignore a single bad night unless the conversation is specifically about that acute event.",
        ],
        retrieval_rules=[
            "Retrieve when the user reports unusual fatigue, poor performance, or reduced capacity.",
        ],
        behavioral_adaptation_rules=[
            "Coaching should reduce training aggressiveness during poor sleep periods.",
            "Recovery suggestions should be prioritized over volume increases.",
        ],
        invalid_memory_usage=[
            "Treating sleep loss as a reason to push harder for discipline.",
        ],
        expiration_policy="Decay after a new stable sleep pattern is recorded.",
        valid_usage_examples=[
            MemoryUsageExample(
                label="valid",
                example="Since you've averaged under six hours of sleep lately, let's keep this week's training focused on quality sets rather than extra volume.",
            )
        ],
        invalid_usage_examples=[
            MemoryUsageExample(
                label="invalid",
                example="You are tired, so we should add more conditioning to force adaptation.",
            )
        ],
    ),
    MemoryEventSpecification(
        event_type=EventType.TRAVEL_STARTED,
        category=MemoryCategory.CONTEXT,
        priority=MemoryPriority.MEDIUM,
        creation_rules=[
            "Create when travel changes environment, routine, or equipment access.",
        ],
        ignore_rules=[
            "Ignore local day trips that do not affect training options.",
        ],
        retrieval_rules=[
            "Retrieve when recommending workouts, step goals, or food structure during the travel window.",
        ],
        behavioral_adaptation_rules=[
            "Advice should match hotel, bodyweight-only, or limited-equipment constraints.",
        ],
        invalid_memory_usage=[
            "Pretending the user still has home-gym or normal kitchen access.",
        ],
        expiration_policy="Persists until TRAVEL_ENDED.",
        valid_usage_examples=[
            MemoryUsageExample(
                label="valid",
                example="While you're away with only bodyweight equipment, let's use short hotel-room circuits and a daily walking target.",
            )
        ],
        invalid_usage_examples=[
            MemoryUsageExample(
                label="invalid",
                example="Stick to your normal barbell lower day even though you said the hotel gym only has dumbbells up to 20 kg.",
            )
        ],
    ),
    MemoryEventSpecification(
        event_type=EventType.NUTRITION_TARGET_SET,
        category=MemoryCategory.NUTRITION,
        priority=MemoryPriority.MEDIUM,
        creation_rules=[
            "Create when calorie or macro targets are clearly agreed upon.",
        ],
        ignore_rules=[
            "Ignore casual food preferences that do not define targets.",
        ],
        retrieval_rules=[
            "Retrieve when the user asks about meals, progress, hunger, or adjustment decisions.",
        ],
        behavioral_adaptation_rules=[
            "Advice should stay consistent with the current macro target unless a reason to change is discussed.",
        ],
        invalid_memory_usage=[
            "Inventing a new calorie target without explaining why the old one is changing.",
        ],
        expiration_policy="Persists until NUTRITION_TARGET_UPDATED.",
        valid_usage_examples=[
            MemoryUsageExample(
                label="valid",
                example="You're already targeting 150 grams of protein, so let's solve the afternoon meal gap rather than rewriting the whole plan.",
            )
        ],
        invalid_usage_examples=[
            MemoryUsageExample(
                label="invalid",
                example="Forget the target we set and just eat intuitively without checking whether that fits your goal.",
            )
        ],
    ),
    MemoryEventSpecification(
        event_type=EventType.STRESS_ELEVATED,
        category=MemoryCategory.READINESS,
        priority=MemoryPriority.MEDIUM,
        creation_rules=[
            "Create when the user reports sustained life or work stress affecting training capacity.",
        ],
        ignore_rules=[
            "Ignore passing annoyance that does not change behavior or recovery.",
        ],
        retrieval_rules=[
            "Retrieve during planning, restart, or recovery-support conversations.",
        ],
        behavioral_adaptation_rules=[
            "The coach should simplify plans and protect minimum viable consistency.",
        ],
        invalid_memory_usage=[
            "Treating high stress as a reason to intensify the plan for toughness.",
        ],
        expiration_policy="Persists until STRESS_RESOLVED or the user reports stability.",
        valid_usage_examples=[
            MemoryUsageExample(
                label="valid",
                example="With work stress elevated right now, the win this week is three short sessions you can actually complete.",
            )
        ],
        invalid_usage_examples=[
            MemoryUsageExample(
                label="invalid",
                example="Stress is not an excuse, so let's add more sessions to prove commitment.",
            )
        ],
    ),
)


def get_memory_event_specifications() -> tuple[MemoryEventSpecification, ...]:
    """Return the synthetic memory specification library."""

    return MEMORY_EVENT_SPECIFICATIONS


def memory_spec_by_event_type(event_type: EventType) -> MemoryEventSpecification:
    """Return one event specification by event type."""

    for specification in MEMORY_EVENT_SPECIFICATIONS:
        if specification.event_type is event_type:
            return specification
    raise KeyError(f"Unsupported synthetic memory event type: {event_type.value}")


def validate_memory_specifications(
    specifications: tuple[MemoryEventSpecification, ...] | None = None,
) -> None:
    """Validate that the synthetic memory specification is complete and unique."""

    specifications = specifications or MEMORY_EVENT_SPECIFICATIONS
    if not specifications:
        raise ValueError("Memory event specifications must not be empty.")

    event_types = [specification.event_type for specification in specifications]
    if len(event_types) != len(set(event_types)):
        raise ValueError("Memory event specifications must be unique per event type.")

    for specification in specifications:
        if not specification.creation_rules:
            raise ValueError(f"{specification.event_type.value} is missing creation rules.")
        if not specification.retrieval_rules:
            raise ValueError(f"{specification.event_type.value} is missing retrieval rules.")
        if not specification.behavioral_adaptation_rules:
            raise ValueError(
                f"{specification.event_type.value} is missing behavioral adaptation rules."
            )
        if not specification.valid_usage_examples or not specification.invalid_usage_examples:
            raise ValueError(
                f"{specification.event_type.value} must include valid and invalid usage examples."
            )
