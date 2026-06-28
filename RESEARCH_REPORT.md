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
