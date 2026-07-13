"""Adapter-only export and GGUF-conversion preparation for calibration LoRA runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from redaesth.config import RedAesthConfig, config as default_config

from training.utils import write_json


def export_adapter_bundle(
    *,
    model: Any,
    tokenizer: Any,
    metrics: dict[str, Any],
    config: RedAesthConfig = default_config,
) -> dict[str, Path]:
    """Save LoRA adapters and conversion metadata without performing GGUF conversion."""

    adapter_dir = config.calibration_adapter_dir
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir / "tokenizer")
    generation_config_path = adapter_dir / "generation_config"
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(generation_config_path)

    training_config_path = write_json(
        adapter_dir / "training_configuration.json",
        {
            "base_model_id": config.base_model_id,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_bias": config.lora_bias,
            "lora_task_type": config.lora_task_type,
            "max_seq_length": config.max_seq_length,
        },
    )
    checkpoint_metadata_path = write_json(
        config.calibration_checkpoint_metadata_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "base_model_id": config.base_model_id,
            "adapter_dir": str(adapter_dir),
            "metrics_path": str(config.calibration_metrics_path),
            "metrics": metrics,
            "gguf_conversion": {
                "status": "not_performed",
                "reason": "This milestone exports adapter artifacts only.",
            },
        },
    )
    return {
        "adapter_dir": adapter_dir,
        "training_configuration": training_config_path,
        "checkpoint_metadata": checkpoint_metadata_path,
    }
