"""Executable calibration QLoRA training entrypoint for Kaggle or a compatible GPU host."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from redaesth.config import RedAesthConfig
from training.dataset import load_calibration_datasets
from training.evaluation import (
    build_calibration_metrics,
    write_calibration_metrics,
    write_calibration_report,
)
from training.export import export_adapter_bundle
from training.model import (
    apply_lora_adapter,
    build_lora_settings,
    create_peft_lora_config,
    load_quantized_model,
    load_selected_tokenizer,
)
from training.trainer import (
    build_calibration_trainer,
    resolve_resume_checkpoint,
    run_calibration_trainer,
)
from training.utils import load_calibration_config, set_reproducible_seed, with_calibration_output_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the required calibration training command-line interface."""

    parser = argparse.ArgumentParser(description="Run the RedAesth calibration QLoRA training job.")
    parser.add_argument("--config", type=Path, help="Optional JSON or YAML typed-config overrides.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume from a checkpoint path, or use the latest checkpoint when passed without a value.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override the configured calibration output root.")
    parser.add_argument("--epochs", type=int, help="Override typed num_train_epochs.")
    parser.add_argument("--batch-size", type=int, help="Override typed per_device_train_batch_size.")
    parser.add_argument("--learning-rate", type=float, help="Override typed learning_rate.")
    return parser


def resolve_cli_config(args: argparse.Namespace) -> RedAesthConfig:
    """Apply allowed CLI overrides through the existing typed configuration system."""

    calibration_config = load_calibration_config(
        config_path=args.config,
        overrides={
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
    )
    return with_calibration_output_dir(calibration_config, args.output_dir)


def execute_calibration(
    calibration_config: RedAesthConfig,
    *,
    resume: str | None = None,
) -> dict[str, Path]:
    """Execute one complete calibration run and write all post-training artifacts."""

    set_reproducible_seed(calibration_config.seed)
    tokenizer = load_selected_tokenizer(calibration_config)
    datasets = load_calibration_datasets(tokenizer=tokenizer, config=calibration_config)
    model = load_quantized_model(calibration_config)
    lora_settings = build_lora_settings(model, config=calibration_config)
    model = apply_lora_adapter(model, create_peft_lora_config(lora_settings))
    trainer, telemetry = build_calibration_trainer(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        config=calibration_config,
    )
    resume_checkpoint = resolve_resume_checkpoint(resume, calibration_config.calibration_output_dir)
    train_metrics = run_calibration_trainer(trainer, resume_checkpoint=resume_checkpoint)
    evaluation_metrics = dict(trainer.evaluate())
    metrics = build_calibration_metrics(
        train_metrics=train_metrics,
        evaluation_metrics=evaluation_metrics,
        data_report=datasets.report,
        telemetry=telemetry,
        output_dir=calibration_config.calibration_output_dir,
        config=calibration_config,
    )
    metrics_path = write_calibration_metrics(metrics, config=calibration_config)
    export_paths = export_adapter_bundle(
        model=model,
        tokenizer=tokenizer,
        metrics=metrics,
        config=calibration_config,
    )
    report_path = write_calibration_report(metrics, config=calibration_config)
    return {"metrics": metrics_path, "report": report_path, **export_paths}


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute the calibration training run."""

    args = build_parser().parse_args(argv)
    outputs = execute_calibration(resolve_cli_config(args), resume=args.resume)
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
