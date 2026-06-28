# Pipeline

This directory contains executable dataset-processing stages from acquisition through final assembly. Each script should be runnable independently and composable through the shared CLI.

Current implemented stage:

- `download.py`: reads `research/dataset_discovery/reports/approved_datasets.json`, downloads the approved Hugging Face datasets into `data/raw/huggingface/`, and writes `data/raw/raw_data_manifest.json` with local SHA-256 checksums.
