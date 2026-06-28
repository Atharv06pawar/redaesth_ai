from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from redaesth_ai.common import configure_logging
from redaesth_ai.research_tasks import (
    DEFAULT_DATASET_QUERIES,
    build_argument_parser,
    run_dataset_search,
)


def main() -> int:
    """CLI entrypoint for dataset discovery."""

    parser = build_argument_parser("Search Hugging Face for candidate datasets.")
    parser.add_argument("--query", dest="queries", action="append")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    output = run_dataset_search(
        queries=args.queries or DEFAULT_DATASET_QUERIES,
        limit_per_query=args.limit,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
