"""Assemble the final training dataset splits from scored and optional synthetic inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from redaesth import build_final_dataset, config
from redaesth_ai.common import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for final dataset assembly."""

    parser = argparse.ArgumentParser(
        description="Build final train/validation/test splits from the scored corpus."
    )
    parser.add_argument("--scored-dataset", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run final dataset assembly and print the output paths."""

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    result = build_final_dataset(
        config=config,
        scored_dataset_path=args.scored_dataset,
    )
    for output in (
        result.final_dataset_path,
        result.train_path,
        result.validation_path,
        result.test_path,
        result.manifest_path,
        result.composition_audit_path,
        result.spot_check_report_path,
        result.dataset_card_path,
        result.training_readiness_report_path,
    ):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
