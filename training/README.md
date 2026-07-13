# RedAesth Calibration Training

`python training/train.py` runs the complete QLoRA calibration pipeline against the locked production synthetic corpus. It loads `data/synthetic/production/synthetic_train.jsonl`, forms a deterministic in-memory validation holdout, validates every record against the selected tokenizer chat template, and writes checkpoints, metrics, adapter artifacts, and `CALIBRATION_REPORT.md`.

## Kaggle

1. Upload this repository as a Kaggle Dataset or add it to the notebook working directory.
2. Enable a GPU accelerator.
3. Run `!pip install -r training/kaggle_requirements.txt`.
4. Run `!python training/kaggle_launcher.py`.

The launcher uses the selected model and typed defaults. Resume after an interruption with `python training/train.py --resume`; pass a specific checkpoint path with `--resume /path/to/checkpoint-N`.

## Overrides

The CLI accepts `--config`, `--output-dir`, `--epochs`, `--batch-size`, and `--learning-rate`. Values are merged into `RedAesthConfig`; no separate training configuration format is introduced.
