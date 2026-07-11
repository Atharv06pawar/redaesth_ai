# Synthetic Generation Report

## Generation Summary

- Generated conversations: 100
- Accepted: 100
- Rejected: 0
- Acceptance rate: 100.00%
- JSONL export: `data/synthetic/validated/synthetic_coaching_pilot.jsonl`

## Rejection Reasons

None. Every exported conversation passed the full deterministic rubric.

## Persona Distribution

- `busy_fat_loss_professional`: 9 (9.00%)
- `competition_prep_advanced`: 9 (9.00%)
- `muscle_gain_new_parent`: 11 (11.00%)
- `muscle_gain_student_confidence_building`: 10 (10.00%)
- `retiree_general_fitness`: 10 (10.00%)
- `returning_after_break_creative`: 11 (11.00%)
- `shift_worker_recovery_focus`: 10 (10.00%)
- `student_beginner_exam_stress`: 11 (11.00%)
- `traveling_sales_runner`: 9 (9.00%)
- `vegan_body_recomp_office_worker`: 10 (10.00%)

## Scenario Distribution

- `beginner_onboarding`: 7 (7.00%)
- `busy_professional_compliance`: 6 (6.00%)
- `competition_preparation`: 10 (10.00%)
- `exam_stress_adjustment`: 9 (9.00%)
- `fat_loss_planning`: 7 (7.00%)
- `injury_recovery_adjustment`: 8 (8.00%)
- `missed_workouts_reset`: 8 (8.00%)
- `muscle_gain_support`: 9 (9.00%)
- `plateau_review`: 8 (8.00%)
- `poor_sleep_recovery`: 10 (10.00%)
- `returning_after_break`: 9 (9.00%)
- `travel_continuity`: 9 (9.00%)

## Coaching Objective Distribution

- `adapt volume temporarily`: 9 (9.00%)
- `adjust to available equipment`: 9 (9.00%)
- `adjust workload`: 10 (10.00%)
- `balance specificity with recovery`: 10 (10.00%)
- `decatastrophize the lapse`: 8 (8.00%)
- `diagnose plateau causes`: 8 (8.00%)
- `protect healing tissue`: 8 (8.00%)
- `rebuild gradually`: 9 (9.00%)
- `reduce overwhelm`: 7 (7.00%)
- `set a moderate deficit`: 7 (7.00%)
- `set surplus expectations`: 9 (9.00%)
- `simplify decisions`: 6 (6.00%)

## Memory Usage Statistics

- `goal_set`: 27 (27.00%)
- `injury_reported`: 8 (8.00%)
- `nutrition_target_set`: 3 (3.00%)
- `schedule_changed`: 12 (12.00%)
- `sleep_pattern_changed`: 9 (9.00%)
- `stress_elevated`: 22 (22.00%)
- `travel_started`: 7 (7.00%)
- `workout_missed`: 12 (12.00%)

## Validator Scores

- `behavioral_adaptation`: average 1.0000, minimum 1.0000, maximum 1.0000
- `coaching_quality`: average 0.9475, minimum 0.9475, maximum 0.9475
- `empathy`: average 1.0000, minimum 1.0000, maximum 1.0000
- `follow_up_questioning`: average 1.0000, minimum 1.0000, maximum 1.0000
- `hallucination_detection`: average 1.0000, minimum 1.0000, maximum 1.0000
- `long_term_memory_usage`: average 1.0000, minimum 1.0000, maximum 1.0000
- `personalization`: average 1.0000, minimum 1.0000, maximum 1.0000
- `repetitive_responses`: average 1.0000, minimum 1.0000, maximum 1.0000
- `scenario_consistency`: average 0.9896, minimum 0.9200, maximum 1.0000
- `scientific_consistency`: average 0.9922, minimum 0.9400, maximum 1.0000

## Rubric Score Distribution

- `0.90-1.00`: 100 (100.00%)

## Conversation Length

- Average conversation length: 3.86 messages
- Average turns: 1.93 user/coach exchanges
- Average coach-response length: 984.88 characters

## Quality Observations

- Every exported conversation uses at least one typed memory reference and turns it into an explicit plan adaptation.
- All 100 exported conversations passed empathy, coaching, personalization, behavioral-adaptation, scientific-consistency, memory, follow-up, safety, repetition, and scenario-consistency checks.
- The pilot is deterministic and intentionally capped at 100 validated samples for engineering review before any scale-up.
