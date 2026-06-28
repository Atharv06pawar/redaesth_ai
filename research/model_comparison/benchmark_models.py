from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from redaesth_ai.cli import DEFAULT_MODEL_IDS
from redaesth_ai.common import configure_logging, read_json
from redaesth_ai.research_tasks import (
    build_argument_parser,
    run_model_benchmarks,
    run_model_search,
)


def main() -> int:
    """CLI entrypoint for leaderboard benchmark retrieval."""

    parser = build_argument_parser("Fetch Open LLM leaderboard metrics for candidate models.")
    parser.add_argument("--model", dest="models", action="append")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    metadata_report = run_model_search(model_ids=args.models or DEFAULT_MODEL_IDS)
    model_ids = [
        item["resolved_id"]
        for item in read_json(metadata_report)["models"]
        if item.get("resolved_id")
    ]
    output = run_model_benchmarks(model_ids=model_ids)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
