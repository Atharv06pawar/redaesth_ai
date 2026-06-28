"""Ensure the local `src/` package directory is importable from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.is_dir():
    src_string = str(SRC_PATH)
    if src_string not in sys.path:
        sys.path.insert(0, src_string)
