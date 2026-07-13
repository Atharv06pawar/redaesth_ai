"""Generate the fixed, validated synthetic coaching pilot dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from redaesth.config import config
from redaesth.synthetic_generator import generate_pilot_dataset, generate_production_corpus


def build_parser() -> argparse.ArgumentParser:
    """Build the production synthetic-factory command-line interface."""

    parser = argparse.ArgumentParser(description="Generate deterministic RedAesth synthetic data.")
    parser.add_argument("--pilot", action="store_true", help="Run the legacy 100-conversation pilot.")
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Stop after this many batches; rerun the command to resume from checkpoint.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected deterministic generation path and print its materialized artifacts."""

    args = build_parser().parse_args(argv)
    if args.pilot:
        if args.target_count or args.batch_size or args.max_batches:
            raise ValueError("Batch controls apply only to the production corpus factory.")
        result = generate_pilot_dataset(config=config)
        print(result.dataset_path)
        print(result.report_path)
        return 0

    result = generate_production_corpus(
        config=config,
        target_count=args.target_count,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    print(result.accepted_staging_path)
    print(result.rejection_log_path)
    if result.completed:
        print(result.train_path)
        print(result.manifest_path)
        print(result.dataset_card_path)
        print(result.statistics_path)
        print(result.report_path)
    else:
        print("Generation checkpoint saved; rerun the command to resume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
