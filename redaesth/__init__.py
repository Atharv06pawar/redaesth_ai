"""Root package shim that exposes the canonical `src/redaesth` package."""

from __future__ import annotations

import sys
from pathlib import Path
from pkgutil import extend_path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent / "src"

if SRC_ROOT.is_dir():
    src_string = str(SRC_ROOT)
    if src_string not in sys.path:
        sys.path.insert(0, src_string)

__path__ = extend_path(__path__, __name__)

from .config import RedAesthConfig, config, resolve_base_model_id
from .dataset_pipeline import download_approved_datasets, load_approved_datasets

__all__ = [
    "RedAesthConfig",
    "config",
    "resolve_base_model_id",
    "download_approved_datasets",
    "load_approved_datasets",
]
