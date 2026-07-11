# Research Report

## Milestone 1: Repository and constraint inventory
**Date:** 2026-06-28T17:39:04.0351112+05:30

### Objective

Build a memory-first AI coaching system whose engineering choices are optimized for 30-day user retention rather than raw benchmark performance alone.

### Local findings

- `D:\data` already contained collected assets and verifier metadata from earlier crawling work.
- No application repository, git history, top-level docs, or Python project scaffolding existed yet.
- No `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, or `HUGGINGFACE_HUB_TOKEN` environment variables are currently set.
- Python 3.13.5 is available locally.

### Implications

- Research automation must use public endpoints first and degrade gracefully when authenticated APIs are unavailable.
- Synthetic data generation should be implemented now but must tolerate missing LLM credentials.
- The repository should separate preserved corpus assets from application-controlled code and artifacts.

## Milestone 2: Initial model comparison
**Date:** 2026-06-28T17:54:26+05:30

### Sources queried

- Hugging Face model metadata API
- Hugging Face `open-llm-leaderboard/results` dataset

### Findings

- `HuggingFaceTB/SmolLM2-1.7B-Instruct` is the strongest initial candidate among the commercially usable models with public benchmark rows available through the queried leaderboard dataset.
- `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B` resolved successfully on Hugging Face, but the queried public leaderboard dataset did not expose matching benchmark rows at run time, so they remain viable but unselected pending additional evidence.
- `microsoft/Phi-4-mini-instruct` had strong benchmark signals but exceeded the repository's current approximately-2B parameter budget.
- `meta-llama/Llama-3.2-1B-Instruct`, `google/gemma-3-1b-it`, and `LiquidAI/LFM2-1.2B` did not pass the commercial-license filter used in this repository.

### Output artifacts

- `research/model_comparison/reports/model_comparison_20260628T122433Z.md`
- `DECISION_LOG.md` Decision 4

## Milestone 3: Literature scan
**Date:** 2026-06-28T17:56:08+05:30

### Query theme

- `event sourcing AI memory`

### Relevant paper captured

- `2026-06-25` [ConvMemory v3: A Validity Context Layer for Conversational Memory via Target-Conditioned Relation Verification](https://arxiv.org/abs/2606.26753v1)

### Why it matters

- The paper is directly relevant to RedAesth's memory architecture because it focuses on a failure mode that matters for retention: stale but still relevant memories that should be superseded gracefully rather than repeated blindly.
- Its emphasis on validity metadata, update evidence, and conservative demotion maps well onto RedAesth's planned event-sourcing model, where newer user facts need to override old ones without losing historical context.

### Output artifacts

- `research/literature/summaries/2026_06_25t08_36_43z_convmemory_v3_a_validity_context_layer_for_conversational_memory_via_target_conditioned_relation_verification.md`
- `research/literature/summaries/literature_index_20260628T122608Z.json`

### Next research actions

- Expand the literature search across coaching, retrieval, hallucination reduction, and emotional intelligence themes using the new topical filter.
- Run the dataset discovery workflow across the full query set and review the resulting approved license list.
- Add a second leaderboard source for models that are too new to appear in the currently queried Open LLM results dataset.

## Milestone 4: Phase 2 readiness backbone
**Date:** 2026-06-28T18:15:00+05:30

### Infrastructure completed

- Added `research/model_comparison/coaching_eval.py`, a 25-prompt coaching-specific evaluation harness covering emotional acknowledgment, memory usage, personalization quality, and scientific accuracy.
- Seeded `research/model_comparison/selected_model.txt` with the current Phase 1 winner so downstream components have a stable source of truth immediately.
- Added canonical typed settings in `src/redaesth/config.py` with compatibility re-exports from `src/redaesth_ai/config.py`.
- Expanded `.env.example`, dependency declarations, and the `Makefile` to support the new config and evaluation flow.

### Validation completed

- The coaching evaluation suite validates successfully with the required category counts: 8 emotional prompts, 6 memory prompts, 5 personalization prompts, and 6 scientific prompts.
- Config resolution now reads the selected model file correctly and falls back to `DECISION_LOG.md` when needed.
- Local unit tests pass for config resolution, coaching-eval scoring heuristics, and the memory event store.

### Constraint still present

- The full multi-model inference re-evaluation has not been run on this machine yet. `torch`, `transformers`, and `datasets` are installed, but the environment is CPU-only and `bitsandbytes` is unavailable, so the intended 4-bit local comparison would be slow and not representative of the target execution path.

### Next step

- Execute the full domain-specific model comparison on suitable hardware or with an adjusted CPU-safe subset run, then update `selected_model.txt` and `DECISION_LOG.md` from measured results rather than the seeded Phase 1 selection.

## Milestone 5: Dataset acquisition handoff
**Date:** 2026-06-28T23:58:00+05:30

### Findings

- Re-running the dataset license filter against the latest discovery report produced one commercially approved dataset: `ulysses531/fitness-conversation-dataset` under Apache-2.0.
- The earlier empty approved list was caused by a sequencing mistake during local execution: discovery and license filtering were run in parallel, so the filter consumed the older report before the new one finished writing.
- The approved inventory is still thin, but it is no longer empty, which is enough to build the first real raw-data pipeline stage.

### Infrastructure completed

- Added `src/redaesth/dataset_pipeline.py` as the shared raw-data acquisition module.
- Added `pipeline/download.py` as the Phase 2 dataset download entrypoint.
- Added checksum-backed manifest generation at `data/raw/raw_data_manifest.json`.
- Added local tests covering download manifest creation, reuse of existing downloads, and explicit failure on an empty approved list.

### Validation completed

- `python pipeline/download.py` successfully downloaded `ulysses531/fitness-conversation-dataset` into `data/raw/huggingface/ulysses531__fitness-conversation-dataset/`.
- The generated manifest recorded 3 materialized files totaling 13,954,494 bytes, with SHA-256 checksums for `.gitattributes`, `fitness-conversation.jsonl`, and `README.md`.

### Remaining constraint

- The approved dataset pool is currently only one Hugging Face dataset wide, so the next research-quality task is to expand the commercially usable inventory before final dataset composition targets are realistic.

### Next step

- Build the cleaning stage against `data/raw/raw_data_manifest.json`, keeping the raw dataset snapshot as the stable handoff between research and data preparation.

## Milestone 6: Locked final dataset assembly and readiness audit
**Date:** 2026-07-04T23:26:00+05:30

### Assembly findings

- The final assembly stage now emits a single locked artifact at `data/final/final_dataset.jsonl` with one JSONL sample per line.
- Each sample includes the required training fields: `text`, `source_id`, `language`, and `domain`, while preserving provenance and audit metadata such as `dataset_id`, `topic_tags`, `overall_quality_score`, and `normalized_sha256`.
- The `text` field is rendered with the official `HuggingFaceTB/SmolLM2-1.7B-Instruct` tokenizer chat template rather than a hand-rolled formatter. The verified template prepends the model's default system message and wraps each turn in `<|im_start|>...<|im_end|>` markers.

### Locked artifact and split results

- Final assembled corpus: `27,181` samples
- Train split: `24,463` samples
- Val split: `1,359` samples
- Test split: `1,359` samples
- Final artifact SHA-256: `fcceade07717dcc2223f5fc977105fb06e104fedf9c4bd725f48cd5b7bad7722`
- Train SHA-256: `8e1369d33675ad4f7010e2489b297edfdbdb46051c765142b34cd4f11befe640`
- Val SHA-256: `7fe8cae6e5fdcd8bf46afd0a3fefdc1cb484e5447cf6b64b687152ed0929bf55`
- Test SHA-256: `fcc6e73b149bc922d368f082763e0538494d92926c670131e00084b650c8765b`

### Composition audit

- Language distribution from existing pipeline labels:
  - `mostly_ascii`: `66.37%`
  - `majority_non_ascii`: `33.63%`
- Domain distribution from preserved source/topic metadata:
  - `fitness-coaching-adjacent`: `33.46%`
  - `mental-health-adjacent`: `66.54%`
  - `off-domain`: `0.00%`
- Source contribution:
  - `hizardev/MentalHealth-Counseling`: `66.37%`
  - `ulysses531/fitness-conversation-dataset`: `33.63%`
- Exact duplicate rate from preserved `normalized_sha256`: `37.77%`

### Spot-check validation

- Tokenizer validation sample size: `50`
- Malformed samples: `0`
- Samples exceeding `2048` tokens: `0`
- Truncation rate at `2048` tokens: `0.00%`
- Average observed token length in the spot check: `450.22`
- Maximum observed token length in the spot check: `1263`

### Readiness outcome

- Overall decision: `NO GO`
- Blocking metric 1: `mental-health-adjacent` share exceeded the configured `35%` ceiling by reaching `66.54%`
- Blocking metric 2: exact duplicate rate was `37.77%`, showing that upstream deduplication remains incomplete
- The split artifact itself is healthy and deterministic, but the current corpus composition is not yet suitable for the first calibration LoRA run

### Operational note

- The recommended Kaggle training entrypoint in `TRAINING_READINESS_REPORT.md` uses notebook-local installs of `peft`, `bitsandbytes`, and `trl`. Those dependencies were not added to the repository during this data-readiness pass because the task was limited to final assembly and readiness review rather than training-system implementation.

## Milestone 7: Synthetic coaching quality contract
**Date:** 2026-07-10T23:16:00+05:30

### Objective

- Define exactly what a valid synthetic RedAesth coaching conversation looks like before any generator or prompt-template work begins.

### Infrastructure completed

- Added a typed schema in `src/redaesth/synthetic_schema.py` for personas, user profiles, goals, scenarios, memory references, response contracts, and full synthetic conversations.
- Added a structured persona library in `src/redaesth/synthetic_personas.py` covering students, busy professionals, new parents, travelers, shift workers, retirees, body-recomposition users, competition-prep users, and return-after-break cases.
- Added a structured scenario library in `src/redaesth/synthetic_scenarios.py` covering beginner onboarding, fat loss, muscle gain, plateaus, missed workouts, injury recovery, poor sleep, exam stress, travel, busy professionals, returning after a break, and competition preparation.
- Added a memory-specification layer in `src/redaesth/synthetic_memory.py` that builds on the existing memory engine without modifying it, including creation, ignore, retrieval, adaptation, invalid-usage, and expiration rules.
- Added a deterministic validator suite plus rubric in `src/redaesth/synthetic_validation.py` and `src/redaesth/synthetic_rubric.py`, reusing the established coaching-eval and real-data scoring heuristics for empathy, coaching quality, specificity, memory usage, safety, and repetition.
- Added the operator-facing specification document `SYNTHETIC_DATASET_SPECIFICATION.md`.

### Validation completed

- Added new offline tests for:
  - schema validation
  - persona validation
  - scenario validation
  - memory-spec validation
  - validator behavior
  - quality-rubric PASS / FAIL outcomes
- Full repository validation passes:
  - `44` unit tests
  - `python -m redaesth_ai.cli smoke-test`

### Quality contract summary

- Synthetic conversations are now required to pass named thresholds for:
  - empathy
  - coaching quality
  - personalization
  - behavioral adaptation
  - scientific consistency
  - long-term memory usage
  - follow-up questioning
  - hallucination safety
  - repetition control
  - scenario consistency
- Future generators must create structured conversations first and may only emit training data after those samples pass the deterministic rubric.

### Boundary of this milestone

- No synthetic conversations were generated.
- No prompt templates were implemented.
- No training, retrieval, or memory-engine redesign work was started.
- The repository is now ready for the next milestone: controlled synthetic conversation generation against the new contract.

## Milestone 8: Deterministic synthetic conversation generation pilot
**Date:** 2026-07-11T23:13:10+05:30

### Objective

- Produce the first production-ready synthetic coaching pilot from the completed typed framework, without changing the real-data, memory-engine, retrieval, or training subsystems.

### Infrastructure completed

- Added `src/redaesth/synthetic_generator.py`, which composes typed personas, scenarios, goals, profile snapshots, conversation history, memory references, behavioral adaptations, and coaching responses into `SyntheticCoachingConversation` objects.
- Integrated every candidate with the existing synthetic validators and quality rubric. The generator fails closed if it cannot reach the configured accepted count.
- Reused `build_training_record` from final assembly for the established training JSONL schema, plus an offline SmolLM2 chat-template adapter that produces the locked `<|im_start|>...<|im_end|>` format.
- Added `pipeline/generate_synthetic.py` as the deterministic executable entrypoint and typed configuration for the pilot count, generation seed, JSONL path, and report path.
- Added generator unit coverage for deterministic output, typed-conversation validity, validator and memory integration, locked JSONL export, and the exact configured pilot size.

### Pilot results

- Generated candidates: `100`
- Accepted: `100`
- Rejected: `0`
- Acceptance rate: `100.00%`
- JSONL artifact: `data/synthetic/validated/synthetic_coaching_pilot.jsonl`
- Report artifact: `SYNTHETIC_GENERATION_REPORT.md`
- Average conversation length: `3.86` messages
- Average coach-response length: `984.88` characters
- All exported conversations passed every named synthetic validator and the aggregate rubric.

### Validation completed

- `48` unit tests pass, including the new synthetic generator suite.
- `python -m redaesth_ai.cli smoke-test` passes.
- `python -m compileall src tests pipeline redaesth` passes.

### Boundary of this milestone

- The pilot is intentionally limited to 100 validated conversations for engineering review.
- No synthetic-corpus scale-up, model training, Kaggle workflow, retrieval change, or memory-engine change was started.
