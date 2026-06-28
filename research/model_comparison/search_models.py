from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from redaesth_ai.cli import DEFAULT_MODEL_IDS
from redaesth_ai.common import configure_logging
from redaesth_ai.research_tasks import build_argument_parser, run_model_search


def main() -> int:
    """CLI entrypoint for model metadata search."""

    parser = build_argument_parser("Search Hugging Face model metadata for candidate base models.")
    parser.add_argument("--model", dest="models", action="append")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    output = run_model_search(model_ids=args.models or DEFAULT_MODEL_IDS)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
