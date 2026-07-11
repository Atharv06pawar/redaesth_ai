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

from .cleaning import clean_raw_datasets
from .config import RedAesthConfig, config, resolve_base_model_id
from .dataset_pipeline import download_approved_datasets, load_approved_datasets
from .final_dataset import build_final_dataset
from .scoring import score_cleaned_dataset
from .synthetic_memory import get_memory_event_specifications, validate_memory_specifications
from .synthetic_generator import generate_pilot_dataset, generate_validated_conversations
from .synthetic_personas import get_persona_library, validate_persona_library
from .synthetic_rubric import evaluate_synthetic_conversation, synthetic_quality_contract
from .synthetic_scenarios import get_scenario_library, validate_scenario_library

__all__ = [
    "RedAesthConfig",
    "build_final_dataset",
    "clean_raw_datasets",
    "config",
    "evaluate_synthetic_conversation",
    "generate_pilot_dataset",
    "generate_validated_conversations",
    "get_memory_event_specifications",
    "get_persona_library",
    "get_scenario_library",
    "resolve_base_model_id",
    "download_approved_datasets",
    "load_approved_datasets",
    "score_cleaned_dataset",
    "synthetic_quality_contract",
    "validate_memory_specifications",
    "validate_persona_library",
    "validate_scenario_library",
]
