from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from redaesth.config import config, resolve_base_model_id

from .common import LOGGER, PROJECT_ROOT, ensure_directory, now_utc, write_json, write_text


CATEGORY_WEIGHTS = {
    "emotional_acknowledgment": 0.30,
    "memory_usage": 0.30,
    "personalization_quality": 0.25,
    "scientific_accuracy": 0.15,
}

ACK_PATTERNS = [
    r"\bthat(?:'s| is)\b",
    r"\b(i get it|i hear you|i understand)\b",
    r"\b(makes sense)\b",
    r"\b(sounds like)\b",
    r"\b(feel|feeling|frustrat|guilt|guilty|fear|afraid|exhausted|tired|drained)\b",
    r"\b(genuinely|understandably)\b",
]
ADVICE_PATTERNS = [
    r"\b(try|start|focus|aim|consider|adjust|shift|keep|use|plan|do)\b",
    r"\b(you should|let's|i'd)\b",
]
RAW_MEMORY_FIELD_PATTERNS = [
    "current_weight_kg",
    "goal_weight_kg",
    "training_days_per_week",
    "protein_target_g",
    "calorie_target",
    "recent_events:",
]


@dataclass(slots=True)
class EvaluationPrompt:
    """A single domain-specific model evaluation prompt."""

    prompt_id: str
    category: str
    user_message: str
    system_prompt: str = (
        "You are RedAesth Coach, a warm and direct fitness coach who knows the user's history."
    )
    memory_block: str | None = None
    expected_reference_groups: list[list[str]] = field(default_factory=list)
    accepted_patterns: list[str] = field(default_factory=list)
    rejected_patterns: list[str] = field(default_factory=list)

    def build_messages(self) -> list[dict[str, str]]:
        content = self.system_prompt
        if self.memory_block:
            content += f"\n\n[USER MEMORY - USE NATURALLY, DO NOT READ ALOUD]\n{self.memory_block}\n[END USER MEMORY]"
        return [
            {"role": "system", "content": content},
            {"role": "user", "content": self.user_message},
        ]


@dataclass(slots=True)
class PromptScore:
    """A scored response for one prompt."""

    prompt_id: str
    category: str
    score: float
    response: str


@dataclass(slots=True)
class ModelEvaluationResult:
    """Aggregate results for one evaluated model."""

    model_id: str
    category_scores: dict[str, float]
    weighted_score: float
    prompt_scores: list[PromptScore]


def make_emotional_prompts() -> list[EvaluationPrompt]:
    """Return the eight emotional-acknowledgment prompts."""

    prompts = [
        ("emotional_01", "I've been stuck at the same scale weight for three weeks and I'm honestly getting frustrated."),
        ("emotional_02", "I missed four workouts this week and I feel guilty about wasting all the progress I made."),
        ("emotional_03", "My shoulder hurts every time I bench now and I'm scared I'm going backwards again."),
        ("emotional_04", "I'm exhausted from work and I don't know how to keep training like this."),
        ("emotional_05", "I feel embarrassed going to the gym because everyone there looks like they know what they're doing."),
        ("emotional_06", "Nothing seems to be changing and I'm starting to think maybe this just isn't for me."),
        ("emotional_07", "I had a binge weekend and I feel awful about it."),
        ("emotional_08", "I finally hit a PR but now I'm weirdly nervous that I won't be able to do it again."),
    ]
    return [
        EvaluationPrompt(prompt_id=prompt_id, category="emotional_acknowledgment", user_message=message)
        for prompt_id, message in prompts
    ]


def make_memory_prompts() -> list[EvaluationPrompt]:
    """Return the six structured-memory prompts."""

    return [
        EvaluationPrompt(
            prompt_id="memory_01",
            category="memory_usage",
            memory_block=(
                "current_weight_kg: 82\nprimary_goal: cutting\ntraining_days_per_week: 4\n"
                "recent_events:\n- type: weight_loss_stall, details: No scale change for 2 weeks\n"
                "- type: program_change, details: Switched from PPL to Upper/Lower"
            ),
            user_message="Should I lower calories again this week or stay the course?",
            expected_reference_groups=[
                ["82", "82kg"],
                ["cut", "cutting", "lean out"],
                ["upper/lower", "program change", "switched from ppl"],
                ["stall", "no change", "stuck"],
            ],
        ),
        EvaluationPrompt(
            prompt_id="memory_02",
            category="memory_usage",
            memory_block=(
                "bench_pr_kg: 95\ntraining_days_per_week: 3\nsleep_avg_hours: 5.8\n"
                "recent_events:\n- type: work_stress, details: Late meetings all month"
            ),
            user_message="Why does bench feel so heavy lately?",
            expected_reference_groups=[
                ["bench", "press"],
                ["95", "pr"],
                ["sleep", "5.8", "under six hours"],
                ["stress", "late meetings", "work"],
            ],
        ),
        EvaluationPrompt(
            prompt_id="memory_03",
            category="memory_usage",
            memory_block=(
                "dietary_approach: vegetarian\nprotein_target_g: 150\ncurrent_weight_kg: 68\n"
                "recent_events:\n- type: digestion_issue, details: beans causing bloating"
            ),
            user_message="I'm struggling to hit protein. What would you change first?",
            expected_reference_groups=[
                ["vegetarian", "plant-based"],
                ["150", "protein target"],
                ["68", "68kg"],
                ["bloating", "digestion", "beans"],
            ],
        ),
        EvaluationPrompt(
            prompt_id="memory_04",
            category="memory_usage",
            memory_block=(
                "injuries_historical: left shoulder impingement recovered\ncurrent_program: upper/lower\n"
                "recent_events:\n- type: discomfort, details: pressing discomfort returned after incline dumbbells"
            ),
            user_message="Can I push through this shoulder irritation?",
            expected_reference_groups=[
                ["shoulder", "pressing"],
                ["impingement", "old shoulder issue", "history"],
                ["incline", "dumbbells"],
                ["upper/lower", "program"],
            ],
        ),
        EvaluationPrompt(
            prompt_id="memory_05",
            category="memory_usage",
            memory_block=(
                "goal: muscle_gain\ntraining_experience_years: 0.5\nequipment_access: home_limited\n"
                "personal_records:\n- goblet_squat_kg: 24\n- pushups: 18"
            ),
            user_message="Should I bulk harder or stay where I am?",
            expected_reference_groups=[
                ["muscle", "gain", "size"],
                ["beginner", "half a year", "0.5"],
                ["home", "limited equipment"],
                ["goblet squat", "pushups"],
            ],
        ),
        EvaluationPrompt(
            prompt_id="memory_06",
            category="memory_usage",
            memory_block=(
                "travel_started: 2026-07-03\ntravel_ended: 2026-07-17\nequipment_access: bodyweight_only\n"
                "recent_events:\n- type: schedule_changed, details: international travel for work"
            ),
            user_message="How do I keep momentum while I'm away?",
            expected_reference_groups=[
                ["travel", "away", "international"],
                ["bodyweight", "no gym"],
                ["work", "schedule"],
            ],
        ),
    ]


def make_personalization_prompts() -> list[EvaluationPrompt]:
    """Return the five personalization prompts."""

    return [
        EvaluationPrompt(
            prompt_id="personal_01",
            category="personalization_quality",
            memory_block=(
                "age: 34\ntraining_days_per_week: 3\ngoal: weight_loss\nequipment_access: gym\n"
                "schedule_constraints: late meetings on Tuesdays and Thursdays"
            ),
            user_message="Can you help me set up this week?",
            expected_reference_groups=[
                ["3", "three days"],
                ["weight loss", "fat loss"],
                ["tuesdays", "thursdays", "late meetings"],
                ["gym"],
            ],
        ),
        EvaluationPrompt(
            prompt_id="personal_02",
            category="personalization_quality",
            memory_block=(
                "dietary_approach: vegan\nprotein_target_g: 170\ntraining_days_per_week: 5\n"
                "special_note: appetite drops after hard leg days"
            ),
            user_message="What should I do about meals after training?",
            expected_reference_groups=[
                ["vegan", "plant-based"],
                ["170", "protein"],
                ["leg day", "appetite"],
                ["5", "five days"],
            ],
        ),
        EvaluationPrompt(
            prompt_id="personal_03",
            category="personalization_quality",
            memory_block=(
                "goal: first powerlifting meet\ncurrent_program: 4-day powerlifting split\n"
                "bench_pr_kg: 102\nbodyweight_kg: 74"
            ),
            user_message="How aggressive should my next bench block be?",
            expected_reference_groups=[
                ["powerlifting", "meet"],
                ["bench", "102"],
                ["74", "bodyweight"],
                ["4-day", "split"],
            ],
        ),
        EvaluationPrompt(
            prompt_id="personal_04",
            category="personalization_quality",
            memory_block=(
                "age: 61\ngoal: general health\nmobility_limitation: knee stiffness\n"
                "equipment_access: home_limited"
            ),
            user_message="What kind of lower-body work makes sense for me right now?",
            expected_reference_groups=[
                ["61", "older", "general health"],
                ["knee", "stiffness"],
                ["home", "limited equipment"],
            ],
        ),
        EvaluationPrompt(
            prompt_id="personal_05",
            category="personalization_quality",
            memory_block=(
                "goal: muscle_gain\ninjuries_active: none\ntraining_days_per_week: 2\n"
                "schedule_constraints: newborn at home, fragmented sleep"
            ),
            user_message="Is my current setup enough to make progress?",
            expected_reference_groups=[
                ["muscle", "gain"],
                ["2", "two days"],
                ["newborn", "sleep", "fragmented"],
            ],
        ),
    ]


def make_scientific_prompts() -> list[EvaluationPrompt]:
    """Return the six scientific-accuracy prompts."""

    return [
        EvaluationPrompt(
            prompt_id="science_01",
            category="scientific_accuracy",
            user_message="How much protein should I aim for if I want to gain muscle without overcomplicating it?",
            accepted_patterns=[r"1\.6", r"2\.2", r"per (?:kg|kilogram)"],
            rejected_patterns=[r"0\.8g total", r"as much as possible"],
        ),
        EvaluationPrompt(
            prompt_id="science_02",
            category="scientific_accuracy",
            user_message="Do I need a huge calorie deficit like 1200 calories under maintenance to lose fat well?",
            accepted_patterns=[r"(moderate|smaller|conservative) deficit", r"750", r"(not|isn't) necessary"],
            rejected_patterns=[r"1200", r"the bigger the deficit the better"],
        ),
        EvaluationPrompt(
            prompt_id="science_03",
            category="scientific_accuracy",
            user_message="Does eating carbs at night automatically make fat loss worse?",
            accepted_patterns=[r"total (?:intake|calories)", r"time of day (?:doesn't|isn't) the main factor"],
            rejected_patterns=[r"carbs at night cause fat gain", r"never eat carbs at night"],
        ),
        EvaluationPrompt(
            prompt_id="science_04",
            category="scientific_accuracy",
            user_message="If I stop lifting for a while, does muscle turn into fat?",
            accepted_patterns=[r"muscle (?:doesn't|does not) turn into fat", r"lose muscle", r"gain fat separately"],
            rejected_patterns=[r"muscle turns into fat"],
        ),
        EvaluationPrompt(
            prompt_id="science_05",
            category="scientific_accuracy",
            user_message="Is creatine basically only useful if I do a loading phase first?",
            accepted_patterns=[r"3.?5g", r"loading (?:is )?optional", r"(daily|consistent)"],
            rejected_patterns=[r"must load", r"only works with loading"],
        ),
        EvaluationPrompt(
            prompt_id="science_06",
            category="scientific_accuracy",
            user_message="My shoulder hurts on presses. Should I just push through it if I want to grow?",
            accepted_patterns=[r"(don't|do not) push through", r"pain", r"modify|assess|swap"],
            rejected_patterns=[r"no pain no gain", r"push through it"],
        ),
    ]


def build_prompt_suite() -> list[EvaluationPrompt]:
    """Build the complete 25-prompt coaching evaluation suite."""

    prompts = (
        make_emotional_prompts()
        + make_memory_prompts()
        + make_personalization_prompts()
        + make_scientific_prompts()
    )
    if len(prompts) != 25:
        raise ValueError(f"Expected 25 prompts, found {len(prompts)}")
    return prompts


def validate_prompt_suite(prompts: list[EvaluationPrompt]) -> dict[str, int]:
    """Validate prompt counts by category."""

    counts: dict[str, int] = {}
    for prompt in prompts:
        counts[prompt.category] = counts.get(prompt.category, 0) + 1
    expected = {
        "emotional_acknowledgment": 8,
        "memory_usage": 6,
        "personalization_quality": 5,
        "scientific_accuracy": 6,
    }
    if counts != expected:
        raise ValueError(f"Prompt category counts mismatch: expected {expected}, got {counts}")
    return counts


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _first_words(value: str, limit: int = 50) -> str:
    return " ".join(_normalize_text(value).split()[:limit])


def score_emotional_acknowledgment(response: str) -> float:
    """Return 1.0 when an emotional acknowledgment appears before advice."""

    normalized = _normalize_text(response)
    if not normalized:
        return 0.0
    prefix = _first_words(response, 50)
    ack_matches = [re.search(pattern, prefix, flags=re.IGNORECASE) for pattern in ACK_PATTERNS]
    ack_matches = [match for match in ack_matches if match]
    if not ack_matches:
        return 0.0

    first_ack = min(match.start() for match in ack_matches)
    advice_match = None
    for pattern in ADVICE_PATTERNS:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match and (advice_match is None or match.start() < advice_match.start()):
            advice_match = match

    if advice_match is None:
        return 1.0
    return 1.0 if first_ack <= advice_match.start() else 0.0


def _count_reference_groups(response: str, groups: list[list[str]]) -> int:
    normalized = _normalize_text(response)
    count = 0
    for group in groups:
        if any(alias.lower() in normalized for alias in group):
            count += 1
    return count


def score_memory_usage(prompt: EvaluationPrompt, response: str) -> float:
    """Score whether memory facts are used naturally rather than read aloud."""

    if not prompt.expected_reference_groups:
        return 0.0
    referenced = _count_reference_groups(response, prompt.expected_reference_groups)
    raw_fields_used = any(field in response.lower() for field in RAW_MEMORY_FIELD_PATTERNS)
    score = referenced / len(prompt.expected_reference_groups)
    if raw_fields_used:
        score = max(0.0, score - 0.5)
    return max(0.0, min(score, 1.0))


def score_personalization(prompt: EvaluationPrompt, response: str) -> float:
    """Score whether at least two user-specific facts were incorporated."""

    referenced = _count_reference_groups(response, prompt.expected_reference_groups)
    if referenced >= 2:
        return 1.0
    if referenced == 1:
        return 0.5
    return 0.0


def score_scientific_accuracy(prompt: EvaluationPrompt, response: str) -> float:
    """Score a response against prompt-specific scientific guardrails."""

    normalized = _normalize_text(response)
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in prompt.rejected_patterns):
        return 0.0
    matches = sum(
        1
        for pattern in prompt.accepted_patterns
        if re.search(pattern, normalized, flags=re.IGNORECASE)
    )
    if not prompt.accepted_patterns:
        return 0.0
    return matches / len(prompt.accepted_patterns)


def score_prompt(prompt: EvaluationPrompt, response: str) -> float:
    """Score one response according to the prompt category."""

    if prompt.category == "emotional_acknowledgment":
        return score_emotional_acknowledgment(response)
    if prompt.category == "memory_usage":
        return score_memory_usage(prompt, response)
    if prompt.category == "personalization_quality":
        return score_personalization(prompt, response)
    if prompt.category == "scientific_accuracy":
        return score_scientific_accuracy(prompt, response)
    return 0.0


def default_candidate_models() -> list[str]:
    """Return the default candidate model list for domain-specific re-evaluation."""

    current = resolve_base_model_id()
    candidates = [
        current,
        "Qwen/Qwen3-1.7B",
        "google/gemma-3-1b-it",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    ]
    deduped: list[str] = []
    for model_id in candidates:
        if model_id not in deduped:
            deduped.append(model_id)
    return deduped


def _manual_render(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        lines.append(f"{message['role'].upper()}: {message['content']}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


class CoachingEvaluationRunner:
    """Load candidate models, generate responses, and score them."""

    def __init__(self) -> None:
        self.prompts = build_prompt_suite()

    def _load_pipeline(self, model_id: str):
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError("transformers is not installed")

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if importlib.util.find_spec("torch") is not None:
            import torch

            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
                if config.load_in_4bit and importlib.util.find_spec("bitsandbytes") is not None:
                    from transformers import BitsAndBytesConfig

                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                else:
                    model_kwargs["torch_dtype"] = (
                        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    )
            else:
                model_kwargs["torch_dtype"] = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        return generator, tokenizer

    def _generate_response(self, generator, tokenizer, prompt: EvaluationPrompt, max_new_tokens: int) -> str:
        messages = prompt.build_messages()
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = _manual_render(messages)

        outputs = generator(
            rendered,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            return_full_text=False,
        )
        return str(outputs[0]["generated_text"]).strip()

    def evaluate_model(
        self,
        model_id: str,
        *,
        prompt_limit: int | None = None,
        max_new_tokens: int = 220,
    ) -> ModelEvaluationResult:
        generator, tokenizer = self._load_pipeline(model_id)
        selected_prompts = self.prompts[:prompt_limit] if prompt_limit else self.prompts
        prompt_scores: list[PromptScore] = []

        for prompt in selected_prompts:
            response = self._generate_response(generator, tokenizer, prompt, max_new_tokens)
            prompt_scores.append(
                PromptScore(
                    prompt_id=prompt.prompt_id,
                    category=prompt.category,
                    score=score_prompt(prompt, response),
                    response=response,
                )
            )

        category_scores: dict[str, float] = {}
        for category in CATEGORY_WEIGHTS:
            category_items = [item.score for item in prompt_scores if item.category == category]
            category_scores[category] = (
                sum(category_items) / len(category_items) if category_items else 0.0
            )

        weighted_score = sum(
            category_scores.get(category, 0.0) * weight
            for category, weight in CATEGORY_WEIGHTS.items()
        )
        return ModelEvaluationResult(
            model_id=model_id,
            category_scores=category_scores,
            weighted_score=weighted_score,
            prompt_scores=prompt_scores,
        )


def write_selected_model(model_id: str, path: Path | None = None) -> Path:
    """Persist the currently selected base model ID."""

    path = path or (PROJECT_ROOT / "research" / "model_comparison" / "selected_model.txt")
    ensure_directory(path.parent)
    path.write_text(model_id.strip() + "\n", encoding="utf-8")
    return path


def _append_decision_with_table(results: list[ModelEvaluationResult], winner: ModelEvaluationResult) -> None:
    """Append a decision log entry that includes the category score table."""

    decision_log = PROJECT_ROOT / "DECISION_LOG.md"
    content = decision_log.read_text(encoding="utf-8") if decision_log.exists() else "# Decision Log\n"
    numbers = [int(match.group(1)) for match in re.finditer(r"## Decision (\d+):", content)]
    next_number = (max(numbers) + 1) if numbers else 1

    table_lines = [
        "| Model | Emotional | Memory | Personalization | Scientific | Weighted |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        table_lines.append(
            "| "
            + " | ".join(
                [
                    result.model_id,
                    f"{result.category_scores.get('emotional_acknowledgment', 0.0):.3f}",
                    f"{result.category_scores.get('memory_usage', 0.0):.3f}",
                    f"{result.category_scores.get('personalization_quality', 0.0):.3f}",
                    f"{result.category_scores.get('scientific_accuracy', 0.0):.3f}",
                    f"{result.weighted_score:.3f}",
                ]
            )
            + " |"
        )

    entry = (
        f"\n## Decision {next_number}: Domain-specific base model re-evaluation\n"
        f"**Date:** {now_utc().isoformat()}\n"
        f"**Phase:** Phase 2 - Base Model Re-evaluation\n"
        f"**Decision:** Select `{winner.model_id}` as the active BASE_MODEL after coaching-specific evaluation.\n"
        f"**Alternatives Considered:** {', '.join(result.model_id for result in results)}\n"
        f"**Justification:** The selected model achieved the highest weighted score across emotional acknowledgment, memory usage, personalization quality, and scientific accuracy in the Phase 2 coaching evaluation suite.\n"
        f"**Impact:** `research/model_comparison/selected_model.txt` is updated and downstream configuration should read the selected model from that file.\n"
        f"**Scoring Table:**\n"
        + "\n".join(table_lines)
        + "\n---\n"
    )
    decision_log.write_text(content + entry, encoding="utf-8")


def write_reports(results: list[ModelEvaluationResult], report_dir: Path | None = None) -> tuple[Path, Path]:
    """Write JSON and markdown coaching-eval reports."""

    report_dir = report_dir or (PROJECT_ROOT / "research" / "model_comparison" / "reports")
    ensure_directory(report_dir)
    timestamp = now_utc().strftime("%Y%m%dT%H%M%SZ")

    json_path = report_dir / f"coaching_eval_{timestamp}.json"
    markdown_path = report_dir / f"coaching_eval_{timestamp}.md"

    json_payload = {
        "generated_at": timestamp,
        "weights": CATEGORY_WEIGHTS,
        "results": [
            {
                "model_id": result.model_id,
                "category_scores": result.category_scores,
                "weighted_score": result.weighted_score,
                "prompt_scores": [
                    {
                        "prompt_id": item.prompt_id,
                        "category": item.category,
                        "score": item.score,
                        "response": item.response,
                    }
                    for item in result.prompt_scores
                ],
            }
            for result in results
        ],
    }
    write_json(json_path, json_payload)

    lines = [
        "# Coaching Model Evaluation",
        "",
        f"Generated at: {timestamp}",
        "",
        "| Model | Emotional | Memory | Personalization | Scientific | Weighted |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.model_id,
                    f"{result.category_scores.get('emotional_acknowledgment', 0.0):.3f}",
                    f"{result.category_scores.get('memory_usage', 0.0):.3f}",
                    f"{result.category_scores.get('personalization_quality', 0.0):.3f}",
                    f"{result.category_scores.get('scientific_accuracy', 0.0):.3f}",
                    f"{result.weighted_score:.3f}",
                ]
            )
            + " |"
        )
    write_text(markdown_path, "\n".join(lines) + "\n")
    return json_path, markdown_path


def run_coaching_evaluation(
    *,
    model_ids: list[str] | None = None,
    prompt_limit: int | None = None,
    max_new_tokens: int = 220,
    write_decision: bool = True,
) -> dict[str, Path | str]:
    """Run the full coaching evaluation suite and persist its outputs."""

    runner = CoachingEvaluationRunner()
    validate_prompt_suite(runner.prompts)
    model_ids = model_ids or default_candidate_models()
    results: list[ModelEvaluationResult] = []
    failures: dict[str, str] = {}

    for model_id in model_ids:
        try:
            LOGGER.info("Evaluating coaching model %s", model_id)
            results.append(
                runner.evaluate_model(
                    model_id,
                    prompt_limit=prompt_limit,
                    max_new_tokens=max_new_tokens,
                )
            )
        except Exception as exc:  # pragma: no cover - hardware/model availability varies
            failures[model_id] = str(exc)
            LOGGER.warning("Skipping model %s: %s", model_id, exc)

    if not results:
        raise RuntimeError(f"No candidate models were successfully evaluated. Failures: {failures}")

    results.sort(key=lambda item: item.weighted_score, reverse=True)
    winner = results[0]
    selected_model_path = write_selected_model(winner.model_id)
    json_report, md_report = write_reports(results)
    if write_decision:
        _append_decision_with_table(results, winner)

    return {
        "selected_model": winner.model_id,
        "selected_model_path": selected_model_path,
        "json_report": json_report,
        "markdown_report": md_report,
    }

