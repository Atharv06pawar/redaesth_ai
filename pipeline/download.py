"""Download approved datasets into the raw-data staging area."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from redaesth import config, download_approved_datasets
from redaesth_ai.common import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the dataset download stage."""

    parser = argparse.ArgumentParser(description="Download approved datasets into data/raw.")
    parser.add_argument("--approved-report", type=Path)
    parser.add_argument("--dataset-id", action="append", dest="dataset_ids")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dataset download stage."""

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    output = download_approved_datasets(
        config=config,
        approved_report_path=args.approved_report,
        dataset_ids=args.dataset_ids,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
