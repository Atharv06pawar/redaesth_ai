"""Shared configuration, reproducibility, and artifact helpers for calibration training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from redaesth.config import RedAesthConfig, config as default_config


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a stable UTF-8 JSON artifact and create its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_config_overrides(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML override mapping without introducing another config system."""

    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(content)
    else:
        import yaml

        payload = yaml.safe_load(content)
    if not isinstance(payload, dict):
        raise ValueError("Calibration config files must contain a top-level mapping.")
    return payload


def load_calibration_config(
    *,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
    base_config: RedAesthConfig = default_config,
) -> RedAesthConfig:
    """Merge optional file and CLI values into the repository's typed settings model."""

    values = base_config.model_dump(mode="python")
    if config_path is not None:
        values.update(read_config_overrides(config_path))
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    return RedAesthConfig(**values)


def with_calibration_output_dir(
    calibration_config: RedAesthConfig,
    output_dir: Path | None,
) -> RedAesthConfig:
    """Apply a single output-root override while preserving typed artifact subpaths."""

    if output_dir is None:
        return calibration_config
    values = calibration_config.model_dump(mode="python")
    values.update(
        {
            "calibration_output_dir": output_dir,
            "calibration_adapter_dir": output_dir / "adapter",
            "calibration_metrics_path": output_dir / "calibration_metrics.json",
            "calibration_checkpoint_metadata_path": output_dir / "checkpoint_metadata.json",
        }
    )
    return RedAesthConfig(**values)


def set_reproducible_seed(seed: int) -> None:
    """Seed available local RNGs without requiring CUDA during offline tests."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
