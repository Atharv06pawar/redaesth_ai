# RedAesth AI

RedAesth AI is an autonomous engineering repository for building a memory-first AI coaching system. The product goal is not to answer generic fitness questions. It is to coach a specific person over time by combining durable user memory, evidence retrieval, synthetic coaching data, and evaluation centered on 30-day retention.

## Current state

This repository was initialized inside `D:\data`, which already contained collected corpus assets from earlier exploration. Those assets have been preserved under `legacy/` so the application repository can stay clean and runnable while still retaining useful source material.

## Working principles

- Every meaningful technical choice is recorded in `DECISION_LOG.md`.
- Research findings and open questions are recorded in `RESEARCH_REPORT.md`.
- Code directories are documented before code is added to them.
- The repository should stay runnable at every milestone.

## Repository map

- `research/`: live internet-backed model, literature, and dataset discovery.
- `data/`: raw, cleaned, scored, synthetic, retrieval, and final training artifacts.
- `pipeline/`: dataset download, cleaning, scoring, augmentation, and assembly steps.
- `synthetic/`: synthetic coaching data generation inputs and validators.
- `memory/`: event-sourced long-term memory system.
- `retrieval/`: scientific corpus, embeddings, vector storage, and reranking.
- `prompts/`: prompt templates and orchestration logic.
- `training/`: fine-tuning configuration and execution.
- `evaluation/`: behavior, safety, and benchmark evaluation.
- `export/`: GGUF export, quantization, and validation.
- `docs/`: operator and architecture documentation.
- `src/redaesth/`: canonical typed runtime package for new Phase 2 subsystems.
- `src/redaesth_ai/`: shared runtime package used by the phase scripts.
- `legacy/`: preserved preexisting corpus assets kept outside the tracked app surface.

## Intended entrypoints

- `python -m redaesth_ai.cli bootstrap`
- `python -m redaesth_ai.cli research`
- `python pipeline/download.py`
- `python -m redaesth_ai.cli full-pipeline`

The shared CLI is implemented in `src/redaesth_ai/`, while new reusable Phase 2 runtime code is landing in `src/redaesth/` and phase-specific scripts remain at the top-level paths required by the specification.
