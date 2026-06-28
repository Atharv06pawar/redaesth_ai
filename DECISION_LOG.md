# Decision Log

## Decision 1: Use `D:\data` as the canonical repository root
**Date:** 2026-06-28T17:39:04.0351112+05:30
**Phase:** Phase 0 - Initialization
**Decision:** Normalize all new repository work to `D:\data`, matching the user instruction rather than the shell's starting directory.
**Alternatives Considered:** Continue working from `D:\AI\data`; nest a new repository under `D:\data\redaesth-ai`.
**Justification:** A single canonical root reduces path drift, keeps artifacts discoverable, and supports a one-command workflow. That lowers engineering friction and accelerates the memory and retrieval features that matter for retention.
**Impact:** All subsequent paths, configs, logs, and scripts assume `D:\data` as the project root.
---

## Decision 2: Preserve preexisting collected assets under `legacy/`
**Date:** 2026-06-28T17:39:04.0351112+05:30
**Phase:** Phase 0 - Initialization
**Decision:** Move the preexisting corpus and collector state out of the application surface into `legacy/preexisting_assets/`.
**Alternatives Considered:** Delete the old assets; leave the old assets mixed into the root; copy them elsewhere outside the repository root.
**Justification:** Keeping those materials preserves potentially useful retrieval seed content without polluting the working repository. A cleaner root makes the build easier to reason about and reduces the chance of mistakes that would slow delivery of the memory-first coaching experience.
**Impact:** The root now cleanly represents the application repository while still retaining earlier assets for later review.
---

## Decision 3: Use a shared `src/redaesth_ai` package with spec-compatible top-level scripts
**Date:** 2026-06-28T17:39:04.0351112+05:30
**Phase:** Phase 0 - Initialization
**Decision:** Keep the specification's required script locations, but place shared utilities and orchestration logic in `src/redaesth_ai/`.
**Alternatives Considered:** Put all logic directly into the top-level scripts; create a different package name unrelated to the product.
**Justification:** Reusable shared code reduces duplication across research, pipeline, memory, and evaluation modules. That keeps the system easier to evolve as research findings change, which is important for sustaining a better coaching product over time.
**Impact:** New scripts can remain thin entrypoints while complex logic stays modular and testable.
---

## Decision 4: Initial base model selection
**Date:** 2026-06-28T12:24:33.905004+00:00
**Phase:** Phase 1 - Model Comparison
**Decision:** Select `HuggingFaceTB/SmolLM2-1.7B-Instruct` as the initial BASE_MODEL.
**Alternatives Considered:** HuggingFaceTB/SmolLM2-1.7B-Instruct, Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B
**Justification:** The selected model passed the commercial-license filter and had the strongest combined instruction-following and reasoning signal among the queried candidates, while remaining small enough to support low-cost personalization workflows that matter for retention.
**Impact:** Training configuration, prompt formatting, export validation, and downstream tokenizer assumptions should use the selected model until later research overturns it.
---

## Decision 5: Use SQLite with FTS-backed append-only event storage for the MVP
**Date:** 2026-06-28T12:31:35.332955+00:00
**Phase:** Phase 5 - Memory System
**Decision:** Implement the first memory persistence layer as a local SQLite event log with an FTS5 search table and explicit supersession markers.
**Alternatives Considered:** Skip persistence for now; jump directly to Postgres; store memory as mutable profile documents instead of events.
**Justification:** SQLite keeps the memory layer offline-friendly and easy to run while preserving the append-only event history needed for durable user continuity. That makes it faster to iterate on the retention-critical personalization loop before adding infrastructure.
**Impact:** Memory extraction, profile building, and prompt injection can now rely on a concrete event store contract, and later database backends can preserve the same append-only model.
---

## Decision 6: Add a canonical typed configuration layer and persist selected models in files
**Date:** 2026-06-28T18:15:00+05:30
**Phase:** Phase 2 - Configuration Consolidation
**Decision:** Introduce `src/redaesth/config.py` as the canonical typed settings surface, keep `redaesth_ai` as a compatibility layer, and persist the active base model in `research/model_comparison/selected_model.txt`.
**Alternatives Considered:** Keep configuration scattered across modules; continue using only `redaesth_ai`; rely on `DECISION_LOG.md` parsing alone for model selection.
**Justification:** A typed config layer reduces drift across subsystems and makes later pipeline stages easier to wire safely. Persisting the selected model in a dedicated file avoids brittle parsing and gives downstream training, retrieval, and evaluation steps a single stable source of truth.
**Impact:** New Phase 2 components can read configuration and model-selection state consistently without hardcoding values, while older code continues to function during the migration.
---

## Decision 7: Use Hugging Face snapshot downloads plus a local checksum manifest for raw dataset acquisition
**Date:** 2026-06-28T23:58:00+05:30
**Phase:** Phase 2 - Dataset Download Pipeline
**Decision:** Implement the first data acquisition stage as `pipeline/download.py`, backed by `src/redaesth/dataset_pipeline.py`, using Hugging Face snapshot downloads into `data/raw/huggingface/` and a generated `data/raw/raw_data_manifest.json` with SHA-256 checksums.
**Alternatives Considered:** Download datasets ad hoc without a manifest; postpone downloads until the cleaning stage exists; store only dataset IDs and fetch lazily at cleaning time.
**Justification:** A deterministic raw-data manifest turns the approved research output into a concrete, reusable pipeline input. Computing local checksums immediately gives later cleaning and dataset-card steps stable provenance while keeping the first acquisition stage simple and offline-inspectable.
**Impact:** The repository now has a real Phase 2 data-pipeline entrypoint, and subsequent cleaning, scoring, and final assembly stages can build against a stable raw-data contract instead of querying research artifacts directly.
---
