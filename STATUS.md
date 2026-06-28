# RedAesth Phase 2 Status

Last updated: 2026-06-28

## Inspection summary

The repository has a solid Phase 1 foundation but is still early in implementation. The current codebase contains:

- A documented repository scaffold with required top-level docs.
- A working shared CLI under `src/redaesth_ai/`.
- A partial research automation layer for model comparison, literature search, and dataset discovery.
- A partial memory subsystem with an event schema, SQLite-backed event store, and passing unit tests.

The repository does not yet contain the systems that make training immediately runnable at the end of Phase 2: dataset pipeline, retrieval system, prompt orchestration, evaluation framework, synthetic generation, or training smoke-test code.

## What exists

### Repository and documentation

Complete:
- `README.md`
- `ARCHITECTURE.md`
- `DECISION_LOG.md`
- `RESEARCH_REPORT.md`
- Directory README files across the spec structure

Partial:
- `Makefile` now includes install, lint, model-eval, pipeline, train, and eval targets, but several downstream targets still point at scripts that have not been implemented yet
- `pyproject.toml` now declares core and dev dependencies, but requirements generation is still not automated

### Research

Complete:
- `research/model_comparison/search_models.py`
- `research/model_comparison/benchmark_models.py`
- `research/model_comparison/compare_results.py`
- `research/model_comparison/coaching_eval.py`
- `research/model_comparison/selected_model.txt`
- `research/dataset_discovery/search_datasets.py`
- `research/dataset_discovery/license_checker.py`
- `research/literature/search_papers.py`

Partial:
- Model comparison currently relies on public leaderboard artifacts and heuristic proxies for the actual winner, although a domain-specific coaching evaluation harness now exists
- Dataset discovery now produces a machine-readable approved list, but only one dataset currently passes the commercial-use filter
- Literature search works, but only one relevant paper has been captured so far

Missing:
- `research/model_comparison/embedding_model_eval.py`
- `research/model_comparison/selected_embedding_model.txt`

### Configuration

Complete:
- Canonical typed settings model in `src/redaesth/config.py`
- Compatibility re-export in `src/redaesth_ai/config.py`
- Annotated `.env.example`
- Root shim package for `redaesth`

Partial:
- Existing runtime modules have not yet been fully refactored to consume the shared config object everywhere

### Memory

Complete:
- `memory/event_schema.py`
- `memory/event_store.py`
- `memory/tests/test_event_store.py`
- `python -c "from memory.event_store import EventStore; print('memory OK')"` passes
- `python -m unittest memory.tests.test_event_store` passes

Partial:
- Event store uses SQLite with FTS, but lacks the Phase 2 index set recommended by the architecture review
- No migration wiring yet despite `alembic/` directory existing

Missing:
- `memory/event_processor.py`
- `memory/profile_builder.py`
- `memory/memory_injector.py`
- CLI coverage for append/query/profile/clear flows

### Data state

Current state:
- `data/raw/` now contains a checksum-backed manifest at `data/raw/raw_data_manifest.json`
- `data/raw/huggingface/ulysses531__fitness-conversation-dataset/` now contains one real approved dataset snapshot downloaded through `pipeline/download.py`
- `data/synthetic/` contains only `README.md` and `validated/README.md`

Missing:
- No cleaned dataset outputs
- No scored dataset outputs
- No final train/validation/test dataset

### Pipeline

Partial:
- `pipeline/download.py`
- `tests/test_dataset_pipeline.py`
- Real download validation completed for one approved Apache-2.0 Hugging Face dataset, with 3 files and a 13.95 MB raw snapshot recorded in the manifest

Missing:
- `pipeline/clean.py`
- `pipeline/deduplicate.py`
- `pipeline/score.py`
- `pipeline/augment.py`
- `pipeline/build_final_dataset.py`
- Pipeline test suite

### Retrieval

Missing:
- Retrieval corpus documents
- `retrieval/corpus_builder.py`
- `retrieval/embedder.py`
- `retrieval/vector_store.py`
- `retrieval/retriever.py`
- `retrieval/reranker.py`
- Retrieval tests

### Prompts, evaluation, synthetic generation, training, export

Missing:
- Prompt orchestration implementation
- Evaluator implementations and benchmarks
- Synthetic generator implementations
- Training configuration and smoke-test trainer
- Export pipeline implementation

## Environment and tooling findings

Working:
- Python 3.13.5
- Local imports for the current memory subsystem

Blocked or incomplete:
- `python -m pytest --collect-only` fails because `pytest` is not installed in the current environment
- No OpenAI, Anthropic, or Hugging Face tokens are currently set in the environment
- Full local coaching-model evaluation has not been run yet because the machine is CPU-only and `bitsandbytes` is unavailable for the intended 4-bit comparison path

## Key gaps against the Phase 2 review

1. Base model selection has not yet been re-evaluated with actual model outputs from the new coaching-specific prompt suite.
2. The raw-data path now exists and has been exercised against a real snapshot, but the approved dataset pool is still only one dataset wide.
3. The retrieval system is entirely unbuilt.
4. The memory subsystem exists only at the persistence layer, not the profile-building layer.
5. The test surface is still too narrow for Phase 2.

## Working plan

### Immediate next steps

1. Execute the full coaching-model comparison with actual responses and update `selected_model.txt` from measured results.
2. Refactor existing runtime modules to consume the typed settings model consistently.
3. Build the cleaning stage against `data/raw/raw_data_manifest.json` rather than reading research artifacts directly.
4. Improve test collection by installing dev dependencies or validating via the new `make install` path.

### After those land

1. Build the data cleaning pipeline on top of the downloaded raw snapshot and manifest.
2. Implement embedding-model selection and deduplication.
3. Build retrieval corpus infrastructure.
4. Complete the memory profile and injection layer.
5. Build prompt orchestration, evaluators, and training smoke test.

## Current verdict

Phase 1 foundation: usable.

Phase 2 readiness: improving, but still not yet.

The repository is best described as "well-scaffolded, partially researched, and beginning to gain real Phase 2 execution paths." The highest-value move now is to validate the new raw-data acquisition stage and then build cleaning, retrieval, and evaluation layers on top of it.
