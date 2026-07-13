"""Calibration evaluation, runtime measurement, and report generation."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from redaesth.config import RedAesthConfig, config as default_config

from training.callbacks import TrainingTelemetry
from training.dataset import DatasetLoadReport
from training.utils import write_json


def calculate_perplexity(eval_loss: float | None) -> float | None:
    """Convert validation loss into a finite perplexity value when available."""

    if eval_loss is None:
        return None
    return round(math.exp(min(float(eval_loss), 20.0)), 6)


def gpu_memory_statistics() -> dict[str, float | None]:
    """Capture CUDA memory metrics when a GPU is available, otherwise return nulls."""

    try:
        import torch

        if torch.cuda.is_available():
            return {
                "gpu_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
                "gpu_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 4),
            }
    except ImportError:
        pass
    return {"gpu_memory_allocated_gb": None, "gpu_memory_reserved_gb": None}


def checkpoint_statistics(output_dir: Path) -> dict[str, Any]:
    """Summarize Trainer checkpoints without altering them."""

    checkpoints = sorted(path.name for path in output_dir.glob("checkpoint-*") if path.is_dir())
    return {"checkpoint_count": len(checkpoints), "checkpoints": checkpoints}


def build_calibration_metrics(
    *,
    train_metrics: dict[str, Any],
    evaluation_metrics: dict[str, Any],
    data_report: DatasetLoadReport,
    telemetry: TrainingTelemetry,
    output_dir: Path,
    config: RedAesthConfig = default_config,
) -> dict[str, Any]:
    """Build the complete JSON metrics payload required after one calibration run."""

    eval_loss = evaluation_metrics.get("eval_loss")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_model_id": config.base_model_id,
        "train_samples": data_report.train_count,
        "validation_samples": data_report.validation_count,
        "validation_source": data_report.validation_source,
        "validation_loss": eval_loss,
        "perplexity": calculate_perplexity(eval_loss),
        "training_duration_seconds": train_metrics.get("train_runtime"),
        "samples_per_second": train_metrics.get("train_samples_per_second"),
        "tokens_per_second": train_metrics.get("train_tokens_per_second"),
        "train_metrics": train_metrics,
        "evaluation_metrics": evaluation_metrics,
        "telemetry": telemetry.to_dict(),
        "gpu_memory": gpu_memory_statistics(),
        "checkpoint_statistics": checkpoint_statistics(output_dir),
    }


def write_calibration_metrics(
    metrics: dict[str, Any],
    *,
    config: RedAesthConfig = default_config,
) -> Path:
    """Persist calibration evaluation metrics as JSON."""

    return write_json(config.calibration_metrics_path, metrics)


def write_calibration_report(
    metrics: dict[str, Any],
    *,
    config: RedAesthConfig = default_config,
) -> Path:
    """Write CALIBRATION_REPORT.md after a completed Trainer run."""

    lines = [
        "# Calibration Report",
        "",
        f"- Base model: `{metrics['base_model_id']}`",
        f"- Train samples: {metrics['train_samples']}",
        f"- Validation samples: {metrics['validation_samples']}",
        f"- Validation source: {metrics['validation_source']}",
        f"- Validation loss: {metrics['validation_loss']}",
        f"- Perplexity: {metrics['perplexity']}",
        f"- Training duration (seconds): {metrics['training_duration_seconds']}",
        f"- Samples per second: {metrics['samples_per_second']}",
        f"- Tokens per second: {metrics['tokens_per_second']}",
        "",
        "## GPU Memory",
        "",
        f"- Allocated GB: {metrics['gpu_memory']['gpu_memory_allocated_gb']}",
        f"- Reserved GB: {metrics['gpu_memory']['gpu_memory_reserved_gb']}",
        "",
        "## Checkpoints",
        "",
        f"- Count: {metrics['checkpoint_statistics']['checkpoint_count']}",
        *[f"- `{checkpoint}`" for checkpoint in metrics["checkpoint_statistics"]["checkpoints"]],
    ]
    config.calibration_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.calibration_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config.calibration_report_path
