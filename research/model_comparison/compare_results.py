from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from redaesth_ai.cli import DEFAULT_MODEL_IDS
from redaesth_ai.common import configure_logging
from redaesth_ai.research_tasks import build_argument_parser, compare_models


def main() -> int:
    """CLI entrypoint for the weighted model comparison report."""

    parser = build_argument_parser("Compare candidate base models and write a ranked markdown report.")
    parser.add_argument("--model", dest="models", action="append")
    parser.add_argument("--skip-decision-log", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    output = compare_models(
        model_ids=args.models or DEFAULT_MODEL_IDS,
        write_decision=not args.skip_decision_log,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
