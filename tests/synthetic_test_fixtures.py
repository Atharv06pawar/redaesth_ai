from __future__ import annotations

from memory.event_schema import EventType
from redaesth.synthetic_personas import build_user_profile_from_persona, persona_by_id
from redaesth.synthetic_scenarios import scenario_by_id
from redaesth.synthetic_schema import (
    CoachingGoalSpec,
    CoachingResponseSpec,
    CoachingDomain,
    ConversationTurn,
    ExpectedCoachingBehavior,
    MemoryReference,
    MemoryUsageMode,
    QualityMetadata,
    SyntheticCoachingConversation,
)


def build_valid_synthetic_conversation() -> SyntheticCoachingConversation:
    persona = persona_by_id("busy_fat_loss_professional")
    scenario = scenario_by_id("fat_loss_planning")
    profile = build_user_profile_from_persona(
        persona,
        user_id="user-001",
        equipment_access="commercial gym",
        sleep_hours_average=6.5,
        priorities=["sustainable fat loss", "keeping evening decisions simple"],
        nutrition_constraints=["late client dinners", "weekday lunch inconsistency"],
        current_stats={"weight_kg": 89, "waist_cm": 101},
    )
    return SyntheticCoachingConversation(
        conversation_id="conversation-001",
        persona=persona,
        user_profile=profile,
        coaching_goal=CoachingGoalSpec(
            goal_id="goal-001",
            domain=CoachingDomain.FAT_LOSS,
            summary="sustainable fat loss",
            timeframe_weeks=12,
            success_metrics=["waist reduction", "consistent calorie adherence"],
            primary_barriers=["late meetings", "takeout after work"],
        ),
        scenario=scenario,
        conversation_history=[
            ConversationTurn(
                role="user",
                content=(
                    "I've been tempted to slash calories because progress feels slow, "
                    "and late client meetings keep wrecking dinner."
                ),
            )
        ],
        memory_references=[
            MemoryReference(
                event_type=EventType.GOAL_SET,
                source_event_id="evt-goal",
                usage_mode=MemoryUsageMode.REQUIRED,
                reason="The current plan must stay tied to the user's fat-loss goal.",
                facts=["sustainable fat loss"],
                influence_summary="Keep the plan sustainable rather than aggressive.",
            ),
            MemoryReference(
                event_type=EventType.NUTRITION_TARGET_SET,
                source_event_id="evt-nutrition",
                usage_mode=MemoryUsageMode.REQUIRED,
                reason="The response should honor the active nutrition targets.",
                facts=["1800 calories", "150 grams of protein"],
                influence_summary="Meal guidance should fit the active macro targets.",
            ),
            MemoryReference(
                event_type=EventType.SCHEDULE_CHANGED,
                source_event_id="evt-schedule",
                usage_mode=MemoryUsageMode.REQUIRED,
                reason="Late meetings affect training and dinner decisions.",
                facts=["late client meetings"],
                influence_summary="Evening decisions should be simplified.",
            ),
        ],
        coaching_response=CoachingResponseSpec(
            response_text=(
                "That makes sense, and with your late client meetings it's smarter to tighten "
                "the plan than to crash diet. Because your goal is sustainable fat loss and "
                "you're already aiming for 1800 calories with 150 grams of protein, I'd keep "
                "the deficit moderate and build this week around three efficient gym sessions "
                "plus a daily step target. Start with a protein-forward lunch so weekday lunch "
                "inconsistency stops snowballing, prelog dinner on meeting days, and use one "
                "short incline walk after an upper-body session so the plan still fits your "
                "commercial gym routine. If progress is flat after two "
                "truly consistent weeks, then we can adjust calories instead of reacting early."
            ),
            follow_up_questions=[
                "Which meal is hardest to control on meeting days?",
                "How many days did you actually hit the 1800-calorie target last week?",
            ],
            cited_principles=[
                "Moderate calorie deficits are more sustainable than aggressive restriction.",
                "Protein intake supports satiety and lean-mass retention during fat loss.",
            ],
            adaptation_summary=(
                "The response adapts the plan to late client meetings and weekday lunch "
                "inconsistency by simplifying meals and using short gym sessions instead of a "
                "high-friction schedule."
            ),
        ),
        expected_coaching_behavior=ExpectedCoachingBehavior(
            empathy_objectives=["acknowledge frustration without exaggerating it"],
            coaching_objectives=["protect adherence", "keep the deficit moderate"],
            emotional_objectives=["reduce urgency", "restore confidence"],
            scientific_grounding_requirements=[
                "use sustainable fat-loss framing",
                "avoid crash-diet recommendations",
            ],
            prohibited_behaviors=["punitive cardio", "extreme calorie cuts"],
            required_follow_up_questions=2,
            must_use_memory=True,
        ),
        quality_metadata=QualityMetadata(
            scenario_id=scenario.scenario_id,
            validator_targets={"overall_quality": 0.75},
            must_pass_validators=[
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
            ],
            rejection_rules=["reject crash-diet coaching", "reject shame-based adherence advice"],
            notes=["canonical passing fixture"],
        ),
    )


def build_invalid_synthetic_conversation() -> SyntheticCoachingConversation:
    conversation = build_valid_synthetic_conversation()
    return conversation.model_copy(
        update={
            "coaching_response": conversation.coaching_response.model_copy(
                update={
                    "response_text": (
                        "You should always cut harder. Ignore the late client meetings, drop "
                        "to 1200 calories immediately, and push through every workout because "
                        "that guarantees fat loss."
                    ),
                    "follow_up_questions": [],
                    "cited_principles": [],
                    "adaptation_summary": None,
                }
            ),
            "memory_references": conversation.memory_references[:1],
            "expected_coaching_behavior": conversation.expected_coaching_behavior.model_copy(
                update={"required_follow_up_questions": 0}
            ),
            "quality_metadata": conversation.quality_metadata.model_copy(
                update={"must_pass_validators": ["scientific_consistency", "hallucination_detection"]}
            ),
        }
    )
