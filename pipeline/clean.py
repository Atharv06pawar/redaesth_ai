"""Clean and normalize raw dataset snapshots into a validated training corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from redaesth import clean_raw_datasets, config
from redaesth_ai.common import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the cleaning stage."""

    parser = argparse.ArgumentParser(
        description="Clean and normalize raw datasets into a validated JSONL corpus."
    )
    parser.add_argument("--raw-manifest", type=Path)
    parser.add_argument("--dataset-id", action="append", dest="dataset_ids")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the cleaning stage and print the produced output paths."""

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    output_path, report_path = clean_raw_datasets(
        config=config,
        raw_manifest_path=args.raw_manifest,
        dataset_ids=args.dataset_ids,
        output_path=args.output_path,
        report_path=args.report_path,
    )
    print(output_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
