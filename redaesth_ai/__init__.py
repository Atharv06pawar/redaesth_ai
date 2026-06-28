"""Root package shim that exposes the `src/redaesth_ai` implementation."""

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
