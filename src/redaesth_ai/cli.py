from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import LOGGER, PROJECT_ROOT, configure_logging
from .research_tasks import (
    DEFAULT_ARXIV_QUERIES,
    DEFAULT_DATASET_QUERIES,
    DEFAULT_MODEL_IDS,
    compare_models,
    run_dataset_search,
    run_license_checker,
    run_literature_search,
)


def bootstrap_command(_: argparse.Namespace) -> int:
    """Validate the repository root and required core files."""

    required_paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "ARCHITECTURE.md",
        PROJECT_ROOT / "DECISION_LOG.md",
        PROJECT_ROOT / "RESEARCH_REPORT.md",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "research",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "memory",
        PROJECT_ROOT / "retrieval",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        for item in missing:
            LOGGER.error("Missing required path: %s", item)
        return 1

    LOGGER.info("Repository bootstrap validation passed for %s", PROJECT_ROOT)
    return 0


def research_command(args: argparse.Namespace) -> int:
    """Run research tasks and print the generated output paths."""

    outputs: list[Path] = []

    if not args.only or args.only == "models":
        outputs.append(compare_models(model_ids=args.models or DEFAULT_MODEL_IDS))
    if not args.only or args.only == "datasets":
        dataset_report = run_dataset_search(
            queries=args.dataset_queries or DEFAULT_DATASET_QUERIES,
            limit_per_query=args.dataset_limit,
        )
        outputs.append(dataset_report)
        outputs.append(run_license_checker(source_report=dataset_report))
    if not args.only or args.only == "literature":
        outputs.append(
            run_literature_search(
                queries=args.paper_queries or DEFAULT_ARXIV_QUERIES,
                max_results_per_query=args.paper_limit,
            )
        )

    for output in outputs:
        print(output)
    return 0


def smoke_test_command(_: argparse.Namespace) -> int:
    """Run a small local-only validation pass."""

    if bootstrap_command(argparse.Namespace()) != 0:
        return 1
    LOGGER.info("Smoke test passed")
    return 0


def full_pipeline_command(args: argparse.Namespace) -> int:
    """Run the implemented end-to-end automation slice."""

    LOGGER.info(
        "Running the current end-to-end slice: bootstrap plus research automation. "
        "Downstream pipeline, memory, retrieval, and training phases will be added in subsequent milestones."
    )
    if bootstrap_command(args) != 0:
        return 1
    return research_command(args)


def build_parser() -> argparse.ArgumentParser:
    """Build the shared CLI parser."""

    parser = argparse.ArgumentParser(prog="redaesth", description="RedAesth AI repository CLI")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Validate the repository scaffold")
    bootstrap.set_defaults(func=bootstrap_command)

    research = subparsers.add_parser("research", help="Run research automation")
    research.add_argument("--only", choices=["models", "datasets", "literature"])
    research.add_argument("--model", dest="models", action="append")
    research.add_argument("--dataset-query", dest="dataset_queries", action="append")
    research.add_argument("--dataset-limit", type=int, default=5)
    research.add_argument("--paper-query", dest="paper_queries", action="append")
    research.add_argument("--paper-limit", type=int, default=3)
    research.set_defaults(func=research_command)

    smoke = subparsers.add_parser("smoke-test", help="Run local smoke checks")
    smoke.set_defaults(func=smoke_test_command)

    full_pipeline = subparsers.add_parser("full-pipeline", help="Run the implemented end-to-end slice")
    full_pipeline.add_argument("--only", choices=["models", "datasets", "literature"])
    full_pipeline.add_argument("--model", dest="models", action="append")
    full_pipeline.add_argument("--dataset-query", dest="dataset_queries", action="append")
    full_pipeline.add_argument("--dataset-limit", type=int, default=5)
    full_pipeline.add_argument("--paper-query", dest="paper_queries", action="append")
    full_pipeline.add_argument("--paper-limit", type=int, default=3)
    full_pipeline.set_defaults(func=full_pipeline_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
