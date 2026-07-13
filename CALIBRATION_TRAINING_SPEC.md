# Calibration Training Specification

## Selected Model

The calibration run uses the repository-selected base model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`. The training package reads `base_model_id` from `RedAesthConfig`, so Qwen3-1.7B and Gemma-3-1B-IT are supported when a future formal model-selection decision changes that field. LoRA projection targets are inferred from the loaded architecture rather than selected from a hard-coded model-ID map.

## Dataset

The immutable source is `data/synthetic/production/synthetic_train.jsonl`: 250 validated synthetic coaching conversations, SHA-256 `e77ca4178f2225c6b66fa14e55da516a909eed89d8d102a5a784a57494689410`. The loader validates each record, re-renders the selected tokenizer chat template, tokenizes to `max_seq_length: 2048`, and forms a seeded 90/10 in-memory train/validation split of 225/25 records. No dataset file is modified.

## LoRA Configuration

```yaml
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_bias: none
lora_task_type: CAUSAL_LM
quantization: 4-bit NF4 with double quantization
gradient_checkpointing: true
optimizer: paged_adamw_8bit
num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.0002
warmup_ratio: 0.05
max_seq_length: 2048
fp16: true
bf16: false
```

## Kaggle Execution

Use Kaggle T4 x2 or a stronger accelerator. The expected 4-bit QLoRA footprint is approximately 12 GB per GPU. With the 225-record calibration train partition, one epoch is expected to finish comfortably within a Kaggle session; runtime telemetry records the actual duration, samples per second, tokens per second, and GPU memory after the run.

Install requirements and run the launcher:

```bash
pip install -r training/kaggle_requirements.txt
python training/kaggle_launcher.py
```

## Outputs

The run writes checkpoints under `training/outputs/calibration_lora_run/checkpoint-*`, `calibration_metrics.json`, `CALIBRATION_REPORT.md`, adapter weights under `training/outputs/calibration_lora_run/adapter/`, the selected tokenizer, generation configuration, training configuration, and checkpoint metadata prepared for later GGUF conversion. GGUF conversion is intentionally not executed.

## Resume Procedure

After an interruption, rerun `python training/train.py --resume` to select the highest available checkpoint. To select a specific checkpoint, pass `--resume training/outputs/calibration_lora_run/checkpoint-N`. The trainer preserves the typed seed, checkpoint cadence, evaluation cadence, and best-model tracking settings.
