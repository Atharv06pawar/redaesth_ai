from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from redaesth_ai.common import configure_logging
from redaesth_ai.research_tasks import build_argument_parser, run_license_checker


def main() -> int:
    """CLI entrypoint for dataset license filtering."""

    parser = build_argument_parser("Filter discovered datasets by commercial-friendly license.")
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    output = run_license_checker(source_report=args.source_report)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
