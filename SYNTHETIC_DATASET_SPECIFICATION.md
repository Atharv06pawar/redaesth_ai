# Synthetic Dataset Specification

## Philosophy

The synthetic coaching dataset is intended to become the primary LoRA training corpus for RedAesth, but this milestone does not generate any conversations. It defines what a valid synthetic coaching conversation must look like before generation begins.

The framework is designed around four principles:

- Structured first: synthetic examples must be described as typed objects before they are rendered into text.
- Behavior over style: conversations are judged by empathy, personalization, adaptation, safety, and scientific quality rather than by generic fluency alone.
- Memory aware: synthetic coaching must use long-term memory naturally, selectively, and only when behaviorally relevant.
- Deterministic validation: every future generated conversation must pass the same named validators and quality thresholds.

## Schema

The canonical schema lives in [src/redaesth/synthetic_schema.py](/D:/data/src/redaesth/synthetic_schema.py).

The main object is `SyntheticCoachingConversation`, which requires:

- `persona`: a typed `PersonaDefinition`
- `user_profile`: a typed `UserProfile` snapshot derived from the persona plus current state
- `coaching_goal`: a typed `CoachingGoalSpec`
- `scenario`: a typed `ScenarioDefinition`
- `conversation_history`: alternating user / assistant history that must start with a user turn and end with the current user turn
- `memory_references`: typed `MemoryReference` objects
- `coaching_response`: a typed `CoachingResponseSpec`
- `expected_coaching_behavior`: explicit empathy, coaching, emotional, and follow-up expectations
- `quality_metadata`: validator targets, must-pass validators, and rejection rules

Schema invariants:

- Empty conversation fields are invalid.
- System messages are not allowed inside `conversation_history`.
- History must alternate roles cleanly.
- `quality_metadata.scenario_id` must match `scenario.scenario_id`.
- Memory-required conversations cannot omit memory references.
- The response must include at least the declared minimum number of follow-up questions.

## Personas

The persona library lives in [src/redaesth/synthetic_personas.py](/D:/data/src/redaesth/synthetic_personas.py).

The current library covers:

- students under exam stress
- busy office professionals pursuing fat loss
- new parents rebuilding muscle-gain routines
- frequent travelers maintaining performance goals
- creators returning after long breaks
- shift workers with recovery problems
- older adults training for general health
- vegan body-recomposition users
- advanced competition-prep users
- low-confidence novice lifters trying to gain muscle

Each persona defines:

- age
- lifestyle type
- training experience
- motivation style
- injury history
- schedule constraints
- available training time
- diet style
- adherence level
- stress level
- confidence level
- barriers and preferred modalities

The library validator enforces:

- unique persona IDs
- non-empty library
- broad age coverage
- multiple lifestyle types
- all experience bands
- multiple diet styles

## Scenarios

The scenario library lives in [src/redaesth/synthetic_scenarios.py](/D:/data/src/redaesth/synthetic_scenarios.py).

Implemented scenarios:

- `beginner_onboarding`
- `fat_loss_planning`
- `muscle_gain_support`
- `plateau_review`
- `missed_workouts_reset`
- `injury_recovery_adjustment`
- `poor_sleep_recovery`
- `exam_stress_adjustment`
- `travel_continuity`
- `busy_professional_compliance`
- `returning_after_break`
- `competition_preparation`

Each scenario defines:

- high-level coaching domains
- common emotions
- coaching objectives
- emotional objectives
- allowed memory event types
- minimum follow-up count
- response keywords
- scientific guardrails
- rejection rules

The scenario validator enforces:

- unique scenario IDs
- non-empty coaching objectives
- at least one declared domain
- presence of all required milestone scenarios

## Memory Expectations

The synthetic memory contract lives in [src/redaesth/synthetic_memory.py](/D:/data/src/redaesth/synthetic_memory.py).

This layer builds on the existing memory engine in [memory/event_schema.py](/D:/data/memory/event_schema.py) and [memory/event_store.py](/D:/data/memory/event_store.py) without modifying it.

Specified event types currently include:

- `GOAL_SET`
- `SCHEDULE_CHANGED`
- `WORKOUT_MISSED`
- `INJURY_REPORTED`
- `SLEEP_PATTERN_CHANGED`
- `TRAVEL_STARTED`
- `NUTRITION_TARGET_SET`
- `STRESS_ELEVATED`

For each supported event type, the specification defines:

- creation rules
- ignore rules
- retrieval rules
- behavioral adaptation rules
- invalid memory usage
- expiration policy
- valid usage examples
- invalid usage examples

Core memory rules:

- Memory should only be created from durable user facts or recurring constraints.
- Transient noise, hypotheticals, and weakly implied facts should be ignored.
- Retrieved memory must adapt coaching behavior, not just decorate the response.
- Raw schema field names must never be surfaced in final coaching text.
- Superseded or resolved memories must not continue steering the conversation as current facts.

## Validation Rules

The validator framework lives in [src/redaesth/synthetic_validation.py](/D:/data/src/redaesth/synthetic_validation.py) and reuses the existing evaluation and scoring heuristics from:

- [src/redaesth/scoring.py](/D:/data/src/redaesth/scoring.py)
- [src/redaesth_ai/coaching_eval.py](/D:/data/src/redaesth_ai/coaching_eval.py)

Implemented validators:

- `empathy`
  - reuses the emotional acknowledgment heuristic
  - checks whether the response acknowledges emotion before advice when emotional support is required
- `coaching_quality`
  - reuses coaching signal, specificity, safety, and cliché penalties
- `personalization`
  - checks whether the response uses persona and user-profile facts
- `behavioral_adaptation`
  - checks whether advice adapts to injuries, schedule constraints, stress, sleep, or nutrition barriers
- `scientific_consistency`
  - checks cited principles, domain alignment, and safety penalties
- `long_term_memory_usage`
  - checks natural use of memory-reference facts and rejects raw memory-field leakage
- `follow_up_questioning`
  - checks required follow-up counts against scenario and expected behavior
- `hallucination_detection`
  - checks for risky absolutes, guarantees, cures, diagnoses, and injury-insensitive advice
- `repetitive_responses`
  - reuses the repetition penalty from the real-data scoring pipeline
- `scenario_consistency`
  - checks response keywords and domain fit against the declared scenario

## Quality Requirements

The quality rubric surface lives in [src/redaesth/synthetic_rubric.py](/D:/data/src/redaesth/synthetic_rubric.py).

Current deterministic minimum thresholds from [src/redaesth/config.py](/D:/data/src/redaesth/config.py):

- empathy: `0.70`
- coaching quality: `0.70`
- personalization: `0.70`
- behavioral adaptation: `0.65`
- scientific consistency: `0.75`
- long-term memory usage: `0.70`
- follow-up questioning: `0.60`
- hallucination safety: `0.85`
- repetition: `0.75`
- scenario consistency: `0.75`
- overall synthetic quality: `0.75`

PASS conditions:

- every validator score meets its named threshold
- every validator listed in `quality_metadata.must_pass_validators` passes
- weighted overall score meets `synthetic_quality_threshold`

FAIL conditions:

- any validator falls below its threshold
- any required validator is missing or fails
- the overall score falls below the configured global threshold

## Generation Contract

This milestone does not implement generation, but any future generator must obey the following contract:

- instantiate a valid `SyntheticCoachingConversation`
- choose a persona from the typed persona library
- choose a scenario from the typed scenario library
- use only allowed memory event types for that scenario
- fill `coaching_response` with grounded, personalized, adaptive coaching
- include follow-up questions when required
- include cited scientific principles when the scenario requires grounding
- run the full validator suite
- discard any sample that fails the quality rubric

## Rejection Rules

Future synthetic conversations must be rejected when they:

- violate the schema
- ignore required memory
- leak raw memory field names
- use unsafe absolutes or pseudo-medical certainty
- fail to adapt to injuries, schedules, sleep loss, or stress
- provide generic advice without personalization
- fail to include required follow-up questions
- conflict with the declared scenario
- fail any must-pass validator
- fail the weighted quality threshold

## Test Coverage

The milestone is covered by offline deterministic unittest modules:

- [tests/test_synthetic_schema.py](/D:/data/tests/test_synthetic_schema.py)
- [tests/test_synthetic_personas.py](/D:/data/tests/test_synthetic_personas.py)
- [tests/test_synthetic_scenarios.py](/D:/data/tests/test_synthetic_scenarios.py)
- [tests/test_synthetic_memory.py](/D:/data/tests/test_synthetic_memory.py)
- [tests/test_synthetic_validation.py](/D:/data/tests/test_synthetic_validation.py)
- [tests/test_synthetic_rubric.py](/D:/data/tests/test_synthetic_rubric.py)

The repository is now ready for the next milestone: synthetic conversation generation against this quality contract.
