# Pipeline

This directory contains executable dataset-processing stages from acquisition through final assembly. Each script should be runnable independently and composable through the shared CLI.

Current implemented stage:

- `download.py`: reads `research/dataset_discovery/reports/approved_datasets.json`, downloads the approved Hugging Face datasets into `data/raw/huggingface/`, and writes `data/raw/raw_data_manifest.json` with local SHA-256 checksums.
- `clean.py`: reads `data/raw/raw_data_manifest.json`, normalizes supported raw dataset files into `data/cleaned/cleaned_dataset.jsonl`, and writes `data/cleaned/cleaning_report.json` with rejection reasons, language hints, and provenance.
- `score.py`: reads `data/cleaned/cleaned_dataset.jsonl`, scores each conversation for domain relevance and coaching usefulness, and writes `data/scored/scored_dataset.jsonl` plus `data/scored/scoring_report.json`.
- `build_final_dataset.py`: assembles train/validation/test splits from scored records and any validated synthetic or augmented JSONL sources that exist at build time.
