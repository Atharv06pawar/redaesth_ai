"""Quality scoring helpers for cleaned coaching conversations."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from redaesth_ai.coaching_eval import score_emotional_acknowledgment

from .config import RedAesthConfig, config


SCORING_REPORT_VERSION = 1
EMOTIONAL_CONTEXT_PATTERNS = (
    "frustrated",
    "anxious",
    "overwhelmed",
    "discouraged",
    "stressed",
    "burned out",
    "ashamed",
    "worried",
    "guilty",
    "疲惫",
    "焦虑",
    "沮丧",
    "压力",
    "担心",
    "崩溃",
    "难受",
    "不舒服",
    "烦",
)
COACHING_ACTION_PATTERNS = (
    "recommend",
    "suggest",
    "start with",
    "focus on",
    "keep",
    "track",
    "adjust",
    "plan",
    "建议",
    "推荐",
    "保持",
    "记录",
    "调整",
    "计划",
    "先",
    "逐步",
    "安排",
)
COLLABORATIVE_PATTERNS = ("let's", "we can", "together", "我们", "可以先", "可以把")
ABSOLUTE_RISK_PATTERNS = (
    "always",
    "never",
    "guarantee",
    "cure",
    "must avoid",
    "all you need",
    "一定",
    "绝对",
    "保证",
    "治愈",
    "完全不用",
)
CICHE_OPENERS = (
    "of course",
    "sure",
    "great question",
    "as an ai",
    "当然可以",
    "你好",
    "您好",
    "很高兴",
)
TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "strength_training": (
        "strength",
        "hypertrophy",
        "bench",
        "squat",
        "deadlift",
        "reps",
        "sets",
        "progressive overload",
        "力量训练",
        "深蹲",
        "硬拉",
        "卧推",
        "增肌",
        "训练计划",
    ),
    "running_cardio": (
        "running",
        "runner",
        "marathon",
        "cardio",
        "jog",
        "有氧",
        "跑步",
        "马拉松",
    ),
    "nutrition": (
        "protein",
        "calorie",
        "diet",
        "carb",
        "fat",
        "meal prep",
        "supplement",
        "nutrition",
        "蛋白质",
        "热量",
        "饮食",
        "碳水",
        "脂肪",
        "营养",
        "补剂",
        "减脂",
    ),
    "recovery_sleep": (
        "recovery",
        "sleep",
        "soreness",
        "rest day",
        "deload",
        "恢复",
        "睡眠",
        "酸痛",
        "休息",
        "疲劳",
    ),
    "injury_pain": (
        "injury",
        "pain",
        "rehab",
        "shoulder",
        "knee",
        "back pain",
        "疼痛",
        "受伤",
        "肩",
        "膝",
        "腰",
    ),
    "motivation_adherence": (
        "motivation",
        "habit",
        "consistency",
        "plateau",
        "stress",
        "坚持",
        "动力",
        "习惯",
        "平台期",
        "压力",
    ),
    "gym_etiquette": (
        "share equipment",
        "gym etiquette",
        "bench press station",
        "交替使用",
        "器械",
        "健身房礼仪",
    ),
    "bodybuilding_prep": (
        "competition prep",
        "bodybuilding",
        "stage",
        "peak week",
        "备赛",
        "健美",
        "比赛",
    ),
    "emotional_support": (
        "mental health",
        "depression",
        "anxiety",
        "anxious",
        "therapist",
        "counselor",
        "lonely",
        "worthless",
        "panic",
        "suicid",
        "心理",
        "焦虑",
        "抑郁",
        "情绪",
        "难过",
        "沮丧",
        "咨询",
        "治疗",
    ),
    "programming_offdomain": (
        "python",
        "javascript",
        "java",
        "html",
        "css",
        "sql",
        "module.exports",
        "```",
        "代码",
        "程序",
        "函数",
    ),
}
SPECIFICITY_UNIT_PATTERN = re.compile(
    r"(\d+\s?(kg|g|calories?|sets?|reps?|minutes?|hours?|days?|weeks?))|"
    r"(\d+\s?(组|次|秒|分钟|小时|天|周|公斤|千卡|克))",
    flags=re.IGNORECASE,
)
LIST_PATTERN = re.compile(r"(^|\n)\s*(?:[-*]|\d+[.)、])\s+", flags=re.MULTILINE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?。！？\n]+")


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def write_json(path: Path, payload: Any) -> Path:
    """Write a UTF-8 JSON file with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write a list of dictionaries to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


def normalized_text(value: str) -> str:
    """Normalize text for substring-based heuristics."""

    return re.sub(r"\s+", " ", value.strip().lower())


def final_exchange(record: dict[str, Any]) -> tuple[str, str]:
    """Return the final user message and final assistant message."""

    messages = record["conversations"]
    final_assistant = ""
    final_user = ""
    for message in reversed(messages):
        if not final_assistant and message["role"] == "assistant":
            final_assistant = message["content"]
            continue
        if final_assistant and message["role"] == "user":
            final_user = message["content"]
            break
    return final_user, final_assistant


def detect_topic_tags(text: str) -> list[str]:
    """Assign coarse topic tags using transparent keyword heuristics."""

    normalized = normalized_text(text)
    tags = [
        tag
        for tag, patterns in TOPIC_PATTERNS.items()
        if any(pattern in normalized for pattern in patterns)
    ]
    return sorted(tags)


def domain_relevance_score(topic_tags: list[str], text: str) -> tuple[float, list[str]]:
    """Score how relevant the conversation is to coaching and fitness."""

    exclusion_reasons: list[str] = []
    if "programming_offdomain" in topic_tags:
        exclusion_reasons.append("off_domain_programming")
        return 0.0, exclusion_reasons

    domain_tags = [tag for tag in topic_tags if not tag.endswith("_offdomain")]
    if len(domain_tags) >= 2:
        return 1.0, exclusion_reasons
    if len(domain_tags) == 1:
        return 0.8, exclusion_reasons

    normalized = normalized_text(text)
    fallback_patterns = ("coach", "goal", "habit", "plan", "训练", "健身", "目标", "计划")
    if any(pattern in normalized for pattern in fallback_patterns):
        return 0.55, exclusion_reasons

    exclusion_reasons.append("weak_domain_signal")
    return 0.2, exclusion_reasons


def specificity_score(assistant_text: str) -> float:
    """Score whether the assistant gives concrete and structured advice."""

    sentences = [part for part in SENTENCE_SPLIT_PATTERN.split(assistant_text) if part.strip()]
    score = 0.0
    if len(assistant_text) >= 80:
        score += 0.35
    elif len(assistant_text) >= 40:
        score += 0.20
    if SPECIFICITY_UNIT_PATTERN.search(assistant_text):
        score += 0.35
    if LIST_PATTERN.search(assistant_text):
        score += 0.15
    if len(sentences) >= 2:
        score += 0.15
    return min(score, 1.0)


def coaching_signal_score(assistant_text: str) -> float:
    """Score whether the response sounds like active coaching rather than vague chat."""

    normalized = normalized_text(assistant_text)
    score = 0.0
    if any(pattern in normalized for pattern in COACHING_ACTION_PATTERNS):
        score += 0.45
    if any(pattern in normalized for pattern in COLLABORATIVE_PATTERNS):
        score += 0.20
    if "?" in assistant_text or "？" in assistant_text:
        score += 0.15
    if any(token in normalized for token in ("you", "your", "你", "你的", "我们")):
        score += 0.20
    return min(score, 1.0)


def cliche_penalty(assistant_text: str) -> float:
    """Penalize generic assistant-y openings lightly."""

    normalized = normalized_text(assistant_text)
    if any(normalized.startswith(pattern) for pattern in CICHE_OPENERS):
        return 0.10
    return 0.0


def safety_penalty(assistant_text: str) -> float:
    """Penalize risky absolutes that are poor fits for coaching."""

    normalized = normalized_text(assistant_text)
    if any(pattern in normalized for pattern in ABSOLUTE_RISK_PATTERNS):
        return 0.20
    return 0.0


def repetition_penalty(assistant_text: str) -> float:
    """Penalize repeated sentence fragments."""

    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(assistant_text) if part.strip()]
    if len(sentences) < 2:
        return 0.0
    duplicates = len(sentences) - len(set(sentences))
    if duplicates <= 0:
        return 0.0
    return min(0.15, 0.05 * duplicates)


def emotional_score(
    user_text: str,
    assistant_text: str,
    *,
    language_hint: str,
) -> float | None:
    """Score emotional acknowledgment only when the rubric is appropriate."""

    if language_hint != "mostly_ascii":
        return None
    normalized_user = normalized_text(user_text)
    if not any(pattern in normalized_user for pattern in EMOTIONAL_CONTEXT_PATTERNS):
        return None
    return score_emotional_acknowledgment(assistant_text)


def aggregate_quality_score(
    *,
    domain_score: float,
    coaching_score: float,
    specificity: float,
    emotional: float | None,
    cliche: float,
    safety: float,
    repetition: float,
) -> float:
    """Combine quality dimensions into a final 0-1 score."""

    weighted_scores = [
        (domain_score, 0.40),
        (coaching_score, 0.30),
        (specificity, 0.30),
    ]
    if emotional is not None:
        weighted_scores = [
            (domain_score, 0.35),
            (coaching_score, 0.25),
            (specificity, 0.25),
            (emotional, 0.15),
        ]

    base_score = sum(score * weight for score, weight in weighted_scores) / sum(
        weight for _, weight in weighted_scores
    )
    penalty = cliche + safety + repetition
    return max(0.0, min(base_score - penalty, 1.0))


def score_record(
    record: dict[str, Any],
    *,
    config: RedAesthConfig,
) -> dict[str, Any]:
    """Score one cleaned conversation and attach training-eligibility metadata."""

    user_text, assistant_text = final_exchange(record)
    combined_text = f"{user_text}\n{assistant_text}"
    topic_tags = detect_topic_tags(combined_text)
    domain_score, exclusion_reasons = domain_relevance_score(topic_tags, combined_text)
    coaching_score = coaching_signal_score(assistant_text)
    specificity = specificity_score(assistant_text)
    emotional = emotional_score(
        user_text,
        assistant_text,
        language_hint=record["language_hint"],
    )
    cliche = cliche_penalty(assistant_text)
    safety = safety_penalty(assistant_text)
    repetition = repetition_penalty(assistant_text)
    overall = aggregate_quality_score(
        domain_score=domain_score,
        coaching_score=coaching_score,
        specificity=specificity,
        emotional=emotional,
        cliche=cliche,
        safety=safety,
        repetition=repetition,
    )

    if domain_score < config.minimum_domain_relevance_score and "off_domain_programming" not in exclusion_reasons:
        exclusion_reasons.append("low_domain_relevance")
    if overall < config.minimum_training_quality_score:
        exclusion_reasons.append("low_quality_score")

    scored = dict(record)
    scored.update(
        {
            "topic_tags": topic_tags,
            "domain_relevance_score": round(domain_score, 4),
            "coaching_signal_score": round(coaching_score, 4),
            "specificity_score": round(specificity, 4),
            "emotional_acknowledgment_score": None if emotional is None else round(emotional, 4),
            "cliche_penalty": round(cliche, 4),
            "safety_penalty": round(safety, 4),
            "repetition_penalty": round(repetition, 4),
            "overall_quality_score": round(overall, 4),
            "passes_training_filter": not exclusion_reasons,
            "exclusion_reasons": exclusion_reasons,
        }
    )
    return scored


def average(values: list[float]) -> float:
    """Return the average of a non-empty list or zero for empty input."""

    return fmean(values) if values else 0.0


def score_cleaned_dataset(
    *,
    config: RedAesthConfig = config,
    cleaned_dataset_path: Path | None = None,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> tuple[Path, Path]:
    """Score the cleaned dataset and write a scored corpus plus report."""

    source_path = config.resolve_path(cleaned_dataset_path or config.cleaned_dataset_path)
    scored_path = config.resolve_path(output_path or config.scored_dataset_path)
    scoring_report_path = config.resolve_path(report_path or config.scoring_report_path)

    records = read_jsonl(source_path)
    if not records:
        raise RuntimeError(f"No cleaned records were found in {source_path}")

    scored_records = [score_record(record, config=config) for record in records]
    write_jsonl(scored_path, scored_records)

    domain_scores = [record["domain_relevance_score"] for record in scored_records]
    coaching_scores = [record["coaching_signal_score"] for record in scored_records]
    specificity_scores = [record["specificity_score"] for record in scored_records]
    overall_scores = [record["overall_quality_score"] for record in scored_records]
    emotional_scores = [
        record["emotional_acknowledgment_score"]
        for record in scored_records
        if record["emotional_acknowledgment_score"] is not None
    ]
    topic_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    for record in scored_records:
        topic_counts.update(record["topic_tags"])
        exclusion_counts.update(record["exclusion_reasons"])
        language_counts.update([record["language_hint"]])
        quality_flag_counts.update(record["quality_flags"])

    report = {
        "report_version": SCORING_REPORT_VERSION,
        "generated_at": utc_timestamp(),
        "source_dataset": str(source_path),
        "output_dataset": str(scored_path),
        "thresholds": {
            "minimum_training_quality_score": config.minimum_training_quality_score,
            "minimum_domain_relevance_score": config.minimum_domain_relevance_score,
        },
        "totals": {
            "records": len(scored_records),
            "passes_training_filter": sum(record["passes_training_filter"] for record in scored_records),
            "excluded_records": sum(not record["passes_training_filter"] for record in scored_records),
            "average_domain_relevance_score": round(average(domain_scores), 4),
            "average_coaching_signal_score": round(average(coaching_scores), 4),
            "average_specificity_score": round(average(specificity_scores), 4),
            "average_overall_quality_score": round(average(overall_scores), 4),
            "emotional_examples_scored": len(emotional_scores),
            "average_emotional_acknowledgment_score": (
                round(average(emotional_scores), 4) if emotional_scores else None
            ),
            "topic_tags": dict(topic_counts),
            "exclusion_reasons": dict(exclusion_counts),
            "language_hints": dict(language_counts),
            "quality_flags": dict(quality_flag_counts),
        },
    }
    write_json(scoring_report_path, report)
    return scored_path, scoring_report_path
