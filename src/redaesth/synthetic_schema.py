"""Typed schema for synthetic coaching conversation specifications."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from memory.event_schema import EventType


class MessageRole(str, Enum):
    """Supported chat roles for synthetic coaching turns."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class CoachingDomain(str, Enum):
    """High-level domains covered by the synthetic coaching corpus."""

    FAT_LOSS = "fat_loss"
    MUSCLE_GAIN = "muscle_gain"
    BODY_RECOMPOSITION = "body_recomposition"
    GENERAL_FITNESS = "general_fitness"
    ADHERENCE = "adherence"
    INJURY_RECOVERY = "injury_recovery"
    NUTRITION = "nutrition"
    RECOVERY = "recovery"


class ExperienceLevel(str, Enum):
    """Training experience band for a persona or scenario."""

    BEGINNER = "beginner"
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class MotivationStyle(str, Enum):
    """Primary motivational frame for the user."""

    HEALTH = "health"
    AESTHETICS = "aesthetics"
    PERFORMANCE = "performance"
    ROUTINE = "routine"
    CONFIDENCE = "confidence"
    STRESS_RELIEF = "stress_relief"


class AdherenceLevel(str, Enum):
    """Reliability of user follow-through."""

    FRAGILE = "fragile"
    INCONSISTENT = "inconsistent"
    REBUILDING = "rebuilding"
    MOSTLY_CONSISTENT = "mostly_consistent"
    HIGHLY_CONSISTENT = "highly_consistent"


class StressLevel(str, Enum):
    """Current life-stress band."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ConfidenceLevel(str, Enum):
    """Confidence level in training or body management."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class LifestyleType(str, Enum):
    """Lifestyle anchor used in the persona library."""

    STUDENT = "student"
    OFFICE_WORKER = "office_worker"
    PARENT = "parent"
    SHIFT_WORKER = "shift_worker"
    FREQUENT_TRAVELER = "frequent_traveler"
    RETIREE = "retiree"
    CREATIVE_FREELANCER = "creative_freelancer"


class DietStyle(str, Enum):
    """Dietary pattern that should shape coaching suggestions."""

    OMNIVORE = "omnivore"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    FLEXIBLE = "flexible"
    HIGH_PROTEIN = "high_protein"
    CULTURAL_RESTRICTION = "cultural_restriction"


class MemoryUsageMode(str, Enum):
    """Whether a memory reference should be surfaced or ignored."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    AVOID = "avoid"


class PersonaDefinition(BaseModel):
    """Structured persona used to condition future synthetic conversations."""

    persona_id: str
    display_name: str
    age: int = Field(ge=18, le=80)
    lifestyle: LifestyleType
    experience_level: ExperienceLevel
    motivation_style: MotivationStyle
    active_injuries: list[str] = Field(default_factory=list)
    historical_injuries: list[str] = Field(default_factory=list)
    schedule_constraints: list[str] = Field(default_factory=list)
    available_training_days_per_week: int = Field(ge=1, le=7)
    available_training_minutes: int = Field(ge=15, le=180)
    diet_style: DietStyle
    adherence_level: AdherenceLevel
    stress_level: StressLevel
    confidence_level: ConfidenceLevel
    lifestyle_notes: list[str] = Field(default_factory=list)
    preferred_training_modalities: list[str] = Field(default_factory=list)
    barriers: list[str] = Field(default_factory=list)

    @field_validator("persona_id", "display_name")
    @classmethod
    def non_empty_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Persona identity fields must be non-empty.")
        return value


class UserProfile(BaseModel):
    """Current user state attached to a synthetic conversation."""

    user_id: str
    age: int = Field(ge=18, le=80)
    experience_level: ExperienceLevel
    training_days_per_week: int = Field(ge=0, le=7)
    available_training_minutes: int = Field(ge=15, le=180)
    equipment_access: str
    diet_style: DietStyle
    adherence_level: AdherenceLevel
    stress_level: StressLevel
    confidence_level: ConfidenceLevel
    sleep_hours_average: float = Field(ge=3.0, le=10.0)
    active_injuries: list[str] = Field(default_factory=list)
    historical_injuries: list[str] = Field(default_factory=list)
    nutrition_constraints: list[str] = Field(default_factory=list)
    schedule_constraints: list[str] = Field(default_factory=list)
    current_stats: dict[str, str | float | int] = Field(default_factory=dict)
    priorities: list[str] = Field(default_factory=list)

    @field_validator("user_id", "equipment_access")
    @classmethod
    def non_empty_user_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("User profile text fields must be non-empty.")
        return value


class CoachingGoalSpec(BaseModel):
    """Primary coaching goal for a conversation."""

    goal_id: str
    domain: CoachingDomain
    summary: str
    timeframe_weeks: int = Field(ge=1, le=52)
    success_metrics: list[str] = Field(default_factory=list)
    primary_barriers: list[str] = Field(default_factory=list)

    @field_validator("goal_id", "summary")
    @classmethod
    def non_empty_goal_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Coaching goal fields must be non-empty.")
        return value


class ConversationTurn(BaseModel):
    """One user or assistant message in the synthetic specification."""

    role: MessageRole
    content: str

    @field_validator("content")
    @classmethod
    def non_empty_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Conversation content must be non-empty.")
        return value


class MemoryReference(BaseModel):
    """A memory event that may influence coaching behavior."""

    event_type: EventType
    source_event_id: str
    usage_mode: MemoryUsageMode
    reason: str
    facts: list[str] = Field(default_factory=list)
    influence_summary: str

    @field_validator("source_event_id", "reason", "influence_summary")
    @classmethod
    def non_empty_memory_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Memory reference text fields must be non-empty.")
        return value


class CoachingResponseSpec(BaseModel):
    """The expected assistant response contract for generation and validation."""

    response_text: str
    follow_up_questions: list[str] = Field(default_factory=list)
    cited_principles: list[str] = Field(default_factory=list)
    adaptation_summary: str | None = None

    @field_validator("response_text")
    @classmethod
    def non_empty_response(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Coaching response text must be non-empty.")
        return value

    @field_validator("follow_up_questions")
    @classmethod
    def non_empty_questions(cls, value: list[str]) -> list[str]:
        for question in value:
            if not question.strip():
                raise ValueError("Follow-up questions must be non-empty.")
        return value

    @field_validator("cited_principles")
    @classmethod
    def non_empty_principles(cls, value: list[str]) -> list[str]:
        for principle in value:
            if not principle.strip():
                raise ValueError("Scientific principles must be non-empty.")
        return value


class ExpectedCoachingBehavior(BaseModel):
    """Behavioral expectations that future synthetic generations must satisfy."""

    empathy_objectives: list[str] = Field(default_factory=list)
    coaching_objectives: list[str] = Field(default_factory=list)
    emotional_objectives: list[str] = Field(default_factory=list)
    scientific_grounding_requirements: list[str] = Field(default_factory=list)
    prohibited_behaviors: list[str] = Field(default_factory=list)
    required_follow_up_questions: int = Field(ge=0, le=4)
    must_use_memory: bool = False


class QualityMetadata(BaseModel):
    """Targets and annotations used by the synthetic quality rubric."""

    scenario_id: str
    validator_targets: dict[str, float] = Field(default_factory=dict)
    must_pass_validators: list[str] = Field(default_factory=list)
    rejection_rules: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("scenario_id")
    @classmethod
    def non_empty_scenario_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Quality metadata must include a scenario_id.")
        return value


class ScenarioDefinition(BaseModel):
    """Structured scenario template for future synthetic generation."""

    scenario_id: str
    title: str
    summary: str
    domains: list[CoachingDomain]
    typical_emotions: list[str] = Field(default_factory=list)
    coaching_objectives: list[str] = Field(default_factory=list)
    emotional_objectives: list[str] = Field(default_factory=list)
    allowed_memory_event_types: list[EventType] = Field(default_factory=list)
    required_follow_up_questions: int = Field(ge=0, le=4)
    response_keywords: list[str] = Field(default_factory=list)
    scientific_guardrails: list[str] = Field(default_factory=list)
    rejection_rules: list[str] = Field(default_factory=list)

    @field_validator("scenario_id", "title", "summary")
    @classmethod
    def non_empty_scenario_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Scenario identity fields must be non-empty.")
        return value

    @model_validator(mode="after")
    def require_domains_and_objectives(self) -> "ScenarioDefinition":
        if not self.domains:
            raise ValueError("Scenario definitions must declare at least one domain.")
        if not self.coaching_objectives:
            raise ValueError("Scenario definitions must declare coaching objectives.")
        return self


class SyntheticCoachingConversation(BaseModel):
    """Full structured synthetic-conversation contract used by validators."""

    conversation_id: str
    persona: PersonaDefinition
    user_profile: UserProfile
    coaching_goal: CoachingGoalSpec
    scenario: ScenarioDefinition
    conversation_history: list[ConversationTurn]
    memory_references: list[MemoryReference] = Field(default_factory=list)
    coaching_response: CoachingResponseSpec
    expected_coaching_behavior: ExpectedCoachingBehavior
    quality_metadata: QualityMetadata

    @field_validator("conversation_id")
    @classmethod
    def non_empty_conversation_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Conversation IDs must be non-empty.")
        return value

    @model_validator(mode="after")
    def validate_history_and_links(self) -> "SyntheticCoachingConversation":
        if not self.conversation_history:
            raise ValueError("Synthetic conversations must include conversation history.")
        if self.conversation_history[0].role is not MessageRole.USER:
            raise ValueError("Conversation history must begin with a user turn.")
        if self.conversation_history[-1].role is not MessageRole.USER:
            raise ValueError("Conversation history must end with the current user turn.")

        previous_role = None
        for turn in self.conversation_history:
            if previous_role is not None and turn.role is previous_role:
                raise ValueError("Conversation history must alternate between user and assistant turns.")
            if turn.role is MessageRole.SYSTEM:
                raise ValueError("System messages are not part of the synthetic conversation history.")
            previous_role = turn.role

        if self.quality_metadata.scenario_id != self.scenario.scenario_id:
            raise ValueError("Quality metadata scenario_id must match the structured scenario.")

        if self.expected_coaching_behavior.must_use_memory and not self.memory_references:
            raise ValueError("Memory-required conversations must include memory references.")

        if (
            len(self.coaching_response.follow_up_questions)
            < self.expected_coaching_behavior.required_follow_up_questions
        ):
            raise ValueError(
                "Coaching response does not include the minimum required follow-up questions."
            )
        return self

    def final_user_message(self) -> str:
        """Return the current user message that the coach is answering."""

        return self.conversation_history[-1].content

    def all_messages(self) -> list[ConversationTurn]:
        """Return the full conversation including the response."""

        return self.conversation_history + [
            ConversationTurn(role=MessageRole.ASSISTANT, content=self.coaching_response.response_text)
        ]
