"""Generate the fixed, validated synthetic coaching pilot dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from redaesth.config import config
from redaesth.synthetic_generator import generate_pilot_dataset


def main() -> int:
    """Run deterministic generation and print the two produced artifact paths."""

    result = generate_pilot_dataset(config=config)
    print(result.dataset_path)
    print(result.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
