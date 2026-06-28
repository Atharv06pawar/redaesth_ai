# Architecture Overview

## Product thesis

RedAesth wins on retention by remembering the user better than a general-purpose assistant. That requires three distinct layers:

1. Behavior layer: a tuned model that responds like a coach rather than a search engine.
2. Memory layer: an event-sourced profile that carries user history across months.
3. Knowledge layer: a retrieval system that injects current scientific evidence without hardcoding it into model weights.

## System layout

### Research and dataset strategy

`research/` automates live discovery of candidate base models, licensing constraints, public datasets, and supporting literature. Those outputs drive model selection, scoring rules, and synthetic data design.

### Data pipeline

`pipeline/` converts approved datasets into a unified conversational format, removes weak or unsafe samples, scores quality, augments memory-aware variants, and constructs the final train/validation/test splits.

### Synthetic generation

`synthetic/` defines scenarios, persona libraries, and validation rules for generating emotionally intelligent, memory-aware coaching conversations when API credentials are available.

### Memory architecture

`memory/` uses an append-only event store, extraction heuristics, and profile-building utilities to turn conversations into durable user state and prompt-ready memory blocks.

### Retrieval architecture

`retrieval/` ingests scientific documents, chunks them, embeds them, stores them in a swappable vector backend, and retrieves evidence using hybrid dense plus sparse ranking.

### Prompt orchestration

`prompts/` assembles the base coaching prompt, injected memory, retrieved evidence, prior conversation history, and the current user message while enforcing a token budget.

### Training and evaluation

`training/` owns fine-tuning execution and configuration. `evaluation/` scores safety, coaching quality, memory use, hallucination resistance, emotional intelligence, and baseline comparisons.

### Export

`export/` packages the trained model into deployable artifacts, validates quantization output, and documents the delivery constraints for mobile or CPU inference.

## Shared runtime package

The repository keeps phase scripts at the exact paths required by the specification, but shared logic lives under `src/redaesth_ai/` so code can be reused cleanly across research, pipeline, memory, retrieval, and evaluation steps.
