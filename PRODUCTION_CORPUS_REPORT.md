# Production Corpus Report

## GO / NO GO

GO

## Generation Summary

- Generation timestamp: 2026-07-11T18:14:50.382740+00:00
- Schema version: 1.0
- Generator version: 1.0
- Accepted: 250
- Rejected: 2
- Retry count: 2
- Acceptance rate: 99.21%
- Candidate validator pass rate: 100.00%
- Export validator pass rate: 100.00%
- `synthetic_train.jsonl` SHA256: `e77ca4178f2225c6b66fa14e55da516a909eed89d8d102a5a784a57494689410`

## Average Validator Scores

- `behavioral_adaptation`: average 1.0000, minimum 1.0000, maximum 1.0000
- `coaching_quality`: average 0.9475, minimum 0.9475, maximum 0.9475
- `empathy`: average 1.0000, minimum 1.0000, maximum 1.0000
- `follow_up_questioning`: average 1.0000, minimum 1.0000, maximum 1.0000
- `hallucination_detection`: average 1.0000, minimum 1.0000, maximum 1.0000
- `long_term_memory_usage`: average 1.0000, minimum 1.0000, maximum 1.0000
- `personalization`: average 1.0000, minimum 1.0000, maximum 1.0000
- `repetitive_responses`: average 1.0000, minimum 1.0000, maximum 1.0000
- `scenario_consistency`: average 0.9891, minimum 0.9200, maximum 1.0000
- `scientific_consistency`: average 0.9918, minimum 0.9400, maximum 1.0000

## Rubric Score Histogram

- `0.90-1.00`: 250

## Persona Distribution

- `busy_fat_loss_professional`: 25 (10.00%)
- `competition_prep_advanced`: 25 (10.00%)
- `muscle_gain_new_parent`: 25 (10.00%)
- `muscle_gain_student_confidence_building`: 25 (10.00%)
- `retiree_general_fitness`: 25 (10.00%)
- `returning_after_break_creative`: 25 (10.00%)
- `shift_worker_recovery_focus`: 25 (10.00%)
- `student_beginner_exam_stress`: 25 (10.00%)
- `traveling_sales_runner`: 25 (10.00%)
- `vegan_body_recomp_office_worker`: 25 (10.00%)

## Scenario Distribution

- `beginner_onboarding`: 21 (8.40%)
- `busy_professional_compliance`: 21 (8.40%)
- `competition_preparation`: 21 (8.40%)
- `exam_stress_adjustment`: 21 (8.40%)
- `fat_loss_planning`: 21 (8.40%)
- `injury_recovery_adjustment`: 21 (8.40%)
- `missed_workouts_reset`: 21 (8.40%)
- `muscle_gain_support`: 21 (8.40%)
- `plateau_review`: 21 (8.40%)
- `poor_sleep_recovery`: 20 (8.00%)
- `returning_after_break`: 20 (8.00%)
- `travel_continuity`: 21 (8.40%)

## Memory Distribution

- `context`: 20 (8.00%)
- `goal`: 54 (21.60%)
- `health`: 21 (8.40%)
- `nutrition`: 21 (8.40%)
- `readiness`: 82 (32.80%)
- `schedule`: 52 (20.80%)

## Conversation Length Distribution

- `2`: 21 (8.40%)
- `4`: 114 (45.60%)
- `6`: 115 (46.00%)

## Quality Gates

- `validator_pass_rate`: PASS (actual 1.0, threshold 1.0)
- `duplicate_rate`: PASS (actual 0.0, threshold 0.0)
- `persona_balance`: PASS (actual 0.0, threshold 0.01)
- `scenario_balance`: PASS (actual 0.0, threshold 0.01)
- `memory_usage_balance`: PASS (actual 0.32799999999999996, threshold 0.4)

## Quality Observations

- Every exported record is schema-valid, memory-adaptive, and passed the complete deterministic quality rubric.
- Diversity is scheduled before generation and verified against the completed corpus before export.

## Remaining Issues

None. The proof corpus passed all configured production quality gates.
