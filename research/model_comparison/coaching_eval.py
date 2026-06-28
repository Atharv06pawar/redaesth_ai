from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from redaesth_ai.common import configure_logging
from redaesth_ai.coaching_eval import (
    CoachingEvaluationRunner,
    default_candidate_models,
    run_coaching_evaluation,
    validate_prompt_suite,
)
from redaesth_ai.research_tasks import build_argument_parser


def main() -> int:
    """CLI entrypoint for domain-specific coaching model evaluation."""

    parser = build_argument_parser("Run coaching-specific base model evaluation.")
    parser.add_argument("--model", dest="models", action="append")
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--skip-decision-log", action="store_true")
    parser.add_argument("--validate-suite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    runner = CoachingEvaluationRunner()
    counts = validate_prompt_suite(runner.prompts)
    if args.validate_suite:
        print(counts)
        return 0

    outputs = run_coaching_evaluation(
        model_ids=args.models or default_candidate_models(),
        prompt_limit=args.prompt_limit,
        max_new_tokens=args.max_new_tokens,
        write_decision=not args.skip_decision_log,
    )
    for value in outputs.values():
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
