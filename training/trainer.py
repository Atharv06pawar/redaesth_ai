"""Hugging Face Trainer assembly and checkpoint-resume support for calibration LoRA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redaesth.config import RedAesthConfig, config as default_config

from training.callbacks import TrainingTelemetry, build_training_callbacks
from training.dataset import CalibrationDatasets


@dataclass(slots=True, frozen=True)
class TrainerSettings:
    """Resolved Trainer options derived from typed repository configuration."""

    output_dir: Path
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    fp16: bool
    bf16: bool
    seed: int


def build_trainer_settings(config: RedAesthConfig = default_config) -> TrainerSettings:
    """Expose the effective calibration Trainer settings for tests and documentation."""

    return TrainerSettings(
        output_dir=config.calibration_output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.calibration_logging_steps,
        eval_steps=config.calibration_eval_steps,
        save_steps=config.calibration_save_steps,
        save_total_limit=config.calibration_save_total_limit,
        fp16=config.calibration_fp16,
        bf16=config.calibration_bf16,
        seed=config.seed,
    )


def resolve_resume_checkpoint(resume: str | Path | None, output_dir: Path) -> Path | None:
    """Resolve an explicit checkpoint or the most recent checkpoint under one output directory."""

    if resume is None:
        return None
    if str(resume) != "latest":
        path = Path(resume)
        if not path.exists():
            raise FileNotFoundError(f"Requested resume checkpoint does not exist: {path}")
        return path
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if path.is_dir() and suffix.isdigit():
            checkpoints.append((int(suffix), path))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints are available under {output_dir}")
    return max(checkpoints, key=lambda item: item[0])[1]


def build_training_arguments(config: RedAesthConfig = default_config) -> Any:
    """Build TrainingArguments lazily for the selected mixed-precision calibration run."""

    from transformers import TrainingArguments

    settings = build_trainer_settings(config)
    return TrainingArguments(
        output_dir=str(settings.output_dir),
        num_train_epochs=settings.num_train_epochs,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        per_device_eval_batch_size=settings.per_device_train_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        warmup_ratio=settings.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.calibration_optimizer,
        logging_steps=settings.logging_steps,
        eval_strategy="steps",
        eval_steps=settings.eval_steps,
        save_strategy="steps",
        save_steps=settings.save_steps,
        save_total_limit=settings.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=settings.fp16,
        bf16=settings.bf16,
        seed=settings.seed,
        data_seed=settings.seed,
        dataloader_num_workers=config.calibration_dataloader_num_workers,
        gradient_checkpointing=config.calibration_gradient_checkpointing,
        remove_unused_columns=False,
        report_to="none",
    )


def build_calibration_trainer(
    *,
    model: Any,
    tokenizer: Any,
    datasets: CalibrationDatasets,
    config: RedAesthConfig = default_config,
) -> tuple[Any, TrainingTelemetry]:
    """Create a standard Hugging Face Trainer with telemetry and early stopping callbacks."""

    from transformers import DataCollatorForLanguageModeling, Trainer

    telemetry = TrainingTelemetry()
    trainer = Trainer(
        model=model,
        args=build_training_arguments(config),
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.validation_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=build_training_callbacks(telemetry, config=config),
    )
    return trainer, telemetry


def run_calibration_trainer(trainer: Any, *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    """Run training with optional checkpoint resume and return Trainer's metrics payload."""

    result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    return dict(result.metrics)
