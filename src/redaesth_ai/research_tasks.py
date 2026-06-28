from __future__ import annotations

import argparse
import html
import re
import statistics
import textwrap
import urllib.parse
import xml.etree.ElementTree as etree
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .common import (
    LOGGER,
    PROJECT_ROOT,
    append_decision,
    append_text,
    ensure_directory,
    fetch_json,
    fetch_text,
    human_params,
    license_from_metadata,
    latest_matching_file,
    parameter_count_from_metadata,
    quote_path_component,
    read_json,
    safe_slug,
    timestamp_slug,
    write_json,
    write_text,
)


HF_MODELS_API = "https://huggingface.co/api/models"
HF_DATASETS_API = "https://huggingface.co/api/datasets"
HF_DATASET_ROWS_API = "https://datasets-server.huggingface.co/rows"
HF_DATASET_SPLITS_API = "https://datasets-server.huggingface.co/splits"
OPEN_LLM_RESULTS_DATASET = "open-llm-leaderboard/results"
ARXIV_API = "http://export.arxiv.org/api/query"
MIN_PAPER_YEAR = 2023

DEFAULT_MODEL_IDS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "google/gemma-3-1b-it",
    "microsoft/Phi-4-mini-instruct",
    "LiquidAI/LFM2-1.2B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "allenai/OLMo-1B-0725-Instruct",
]
MAX_PARAMETER_COUNT = 2_200_000_000

MODEL_WEIGHTS = {
    "instruction_following": 0.25,
    "reasoning_quality": 0.15,
    "hallucination_proxy": 0.20,
    "emotional_awareness": 0.15,
    "mobile_deployment": 0.10,
    "fine_tuning_efficiency": 0.10,
}

APPROVED_MODEL_LICENSES = {"apache-2.0", "mit"}

DEFAULT_DATASET_QUERIES = [
    "fitness conversation",
    "health coaching dialogue",
    "nutrition advice",
    "personal trainer",
    "workout instruction",
    "emotional support conversation",
    "mental health counseling",
    "motivational interviewing",
    "behavior change conversation",
    "exercise science",
    "sports nutrition",
    "strength training",
]

APPROVED_DATASET_LICENSES = {
    "apache-2.0",
    "cc-by-4.0",
    "cc0-1.0",
    "mit",
    "odc-by",
    "pddl",
    "bsd-2-clause",
    "bsd-3-clause",
}
REJECTED_LICENSE_TOKENS = ("arr", "nc", "non-commercial", "other", "rail", "unknown")

DEFAULT_ARXIV_QUERIES = [
    "fitness coaching language model",
    "emotional intelligence language model fine-tuning",
    "personalized AI coaching retention",
    "retrieval augmented generation health",
    "event sourcing AI memory",
    "small language model instruction following",
    "coaching behavior synthetic data generation",
    "long-term user personalization LLM",
    "hallucination reduction retrieval augmentation",
]


@dataclass(slots=True)
class ModelCandidate:
    """Combined metadata and benchmark representation for a candidate base model."""

    requested_id: str
    resolved_id: str | None
    found: bool
    license: str | None
    parameter_count: int | None
    downloads: int | None
    likes: int | None
    last_modified: str | None
    pipeline_tag: str | None
    eq_bench: float | None
    model_card_url: str | None
    metrics: dict[str, float | None]
    filter_reason: str | None = None
    weighted_score: float | None = None


def build_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create a consistent CLI parser."""

    return argparse.ArgumentParser(description=description)


def resolve_model_metadata(requested_id: str) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve a model ID directly or through a search fallback."""

    direct_url = f"{HF_MODELS_API}/{quote_path_component(requested_id)}"
    try:
        return requested_id, fetch_json(direct_url)
    except Exception:
        search_url = (
            f"{HF_MODELS_API}?search={urllib.parse.quote(requested_id)}&limit=10"
        )
        search_results = fetch_json(search_url)
        for result in search_results:
            if result.get("pipeline_tag") == "text-generation":
                candidate_id = str(result["id"])
                try:
                    return candidate_id, fetch_json(
                        f"{HF_MODELS_API}/{quote_path_component(candidate_id)}"
                    )
                except Exception:
                    continue
    return None, None


def extract_eq_bench_score(model_id: str) -> float | None:
    """Best-effort extraction of EQ-Bench-like scores from the model card text."""

    url = f"https://huggingface.co/{quote_path_component(model_id)}/raw/main/README.md"
    try:
        readme = fetch_text(url)
    except Exception:
        return None

    match = re.search(r"EQ[- ]?Bench[^0-9]*(\d+(?:\.\d+)?)", readme, flags=re.IGNORECASE)
    if not match:
        return None

    value = float(match.group(1))
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(value, 1.0))


def model_card_url(model_id: str) -> str:
    """Return a canonical model card URL."""

    return f"https://huggingface.co/{quote_path_component(model_id)}"


def run_model_search(
    *,
    model_ids: list[str],
    reports_dir: Path | None = None,
) -> Path:
    """Fetch model metadata and write a timestamped JSON report."""

    reports_dir = reports_dir or PROJECT_ROOT / "research" / "model_comparison" / "reports"
    ensure_directory(reports_dir)

    records: list[dict[str, Any]] = []
    for requested_id in model_ids:
        resolved_id, metadata = resolve_model_metadata(requested_id)
        found = metadata is not None and resolved_id is not None
        LOGGER.info("Resolved model", extra={"requested_id": requested_id, "found": found})

        if not found:
            records.append(
                {
                    "requested_id": requested_id,
                    "resolved_id": None,
                    "found": False,
                    "error": "Model could not be resolved via Hugging Face API.",
                }
            )
            continue

        assert metadata is not None
        assert resolved_id is not None

        records.append(
            {
                "requested_id": requested_id,
                "resolved_id": resolved_id,
                "found": True,
                "license": license_from_metadata(metadata),
                "parameter_count": parameter_count_from_metadata(metadata),
                "downloads": metadata.get("downloads"),
                "likes": metadata.get("likes"),
                "last_modified": metadata.get("lastModified"),
                "pipeline_tag": metadata.get("pipeline_tag"),
                "tags": metadata.get("tags", []),
                "eq_bench": extract_eq_bench_score(resolved_id),
                "model_card_url": model_card_url(resolved_id),
            }
        )

    output_path = reports_dir / f"model_search_{timestamp_slug()}.json"
    write_json(output_path, {"generated_at": timestamp_slug(), "models": records})
    return output_path


def fetch_open_llm_result_index() -> dict[str, str]:
    """Map model IDs to the latest available Open LLM leaderboard result JSON."""

    dataset_info = fetch_json(f"{HF_DATASETS_API}/{quote_path_component(OPEN_LLM_RESULTS_DATASET)}")
    mapping: dict[str, str] = {}
    for sibling in dataset_info.get("siblings", []):
        path = sibling.get("rfilename")
        if not isinstance(path, str):
            continue
        match = re.match(r"(.+)/results_[^/]+\.json$", path)
        if not match:
            continue
        model_id = match.group(1)
        current = mapping.get(model_id)
        if current is None or path > current:
            mapping[model_id] = path
    return mapping


def fetch_open_llm_metrics(model_id: str, index: dict[str, str]) -> dict[str, float | None]:
    """Fetch the latest benchmark metrics for a model if available."""

    default_metrics = {
        "ifeval_strict": None,
        "ifeval_loose": None,
        "bbh": None,
        "gpqa": None,
        "math_hard": None,
        "musr": None,
        "mmlu_pro": None,
    }
    result_path = index.get(model_id)
    if result_path is None:
        return default_metrics

    url = (
        "https://huggingface.co/datasets/open-llm-leaderboard/results/resolve/main/"
        f"{quote_path_component(result_path)}"
    )
    try:
        payload = fetch_json(url)
    except Exception:
        return default_metrics

    results = payload.get("results", {})
    leaderboard = results.get("leaderboard", {})
    default_metrics["ifeval_strict"] = leaderboard.get("inst_level_strict_acc,none")
    default_metrics["ifeval_loose"] = leaderboard.get("inst_level_loose_acc,none")
    default_metrics["bbh"] = (results.get("leaderboard_bbh") or {}).get("acc_norm,none")
    default_metrics["gpqa"] = (results.get("leaderboard_gpqa") or {}).get("acc_norm,none")
    default_metrics["math_hard"] = (results.get("leaderboard_math_hard") or {}).get(
        "exact_match,none"
    )
    default_metrics["musr"] = (results.get("leaderboard_musr") or {}).get("acc_norm,none")
    default_metrics["mmlu_pro"] = (results.get("leaderboard_mmlu_pro") or {}).get("acc,none")
    return default_metrics


def run_model_benchmarks(
    *,
    model_ids: list[str],
    reports_dir: Path | None = None,
) -> Path:
    """Fetch benchmark data for the candidate models and persist it as JSON."""

    reports_dir = reports_dir or PROJECT_ROOT / "research" / "model_comparison" / "reports"
    ensure_directory(reports_dir)
    index = fetch_open_llm_result_index()

    benchmarks: list[dict[str, Any]] = []
    for model_id in model_ids:
        metrics = fetch_open_llm_metrics(model_id, index)
        benchmarks.append({"model_id": model_id, "metrics": metrics})

    output_path = reports_dir / f"model_benchmarks_{timestamp_slug()}.json"
    write_json(output_path, {"generated_at": timestamp_slug(), "benchmarks": benchmarks})
    return output_path


def min_max_normalize(values: dict[str, float]) -> dict[str, float]:
    """Normalize values into the [0, 1] range while preserving order."""

    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def inverse_min_max_normalize(values: dict[str, float]) -> dict[str, float]:
    """Normalize values into the [0, 1] range where smaller is better."""

    direct = min_max_normalize(values)
    return {key: 1.0 - value for key, value in direct.items()}


def average_available(values: list[float | None]) -> float | None:
    """Average the non-null values in a list."""

    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return statistics.fmean(filtered)


def format_metric(value: float | None) -> str:
    """Format a metric as a percentage-like value when present."""

    if value is None:
        return "n/a"
    return f"{value:.3f}"


def model_candidates_from_reports(
    metadata_report: Path,
    benchmark_report: Path,
) -> list[ModelCandidate]:
    """Merge model search and benchmark reports into structured candidates."""

    metadata_payload = read_json(metadata_report)
    benchmark_payload = read_json(benchmark_report)
    benchmark_lookup = {
        item["model_id"]: item["metrics"] for item in benchmark_payload.get("benchmarks", [])
    }

    candidates: list[ModelCandidate] = []
    for record in metadata_payload.get("models", []):
        resolved_id = record.get("resolved_id")
        metrics = benchmark_lookup.get(resolved_id or "", {})
        candidates.append(
            ModelCandidate(
                requested_id=str(record["requested_id"]),
                resolved_id=resolved_id,
                found=bool(record.get("found")),
                license=record.get("license"),
                parameter_count=record.get("parameter_count"),
                downloads=record.get("downloads"),
                likes=record.get("likes"),
                last_modified=record.get("last_modified"),
                pipeline_tag=record.get("pipeline_tag"),
                eq_bench=record.get("eq_bench"),
                model_card_url=record.get("model_card_url"),
                metrics=metrics,
            )
        )
    return candidates


def score_model_candidates(candidates: list[ModelCandidate]) -> list[ModelCandidate]:
    """Apply license filtering and weighted scoring to candidate models."""

    eligible = [candidate for candidate in candidates if candidate.found and candidate.resolved_id]

    mobile_inputs = {
        candidate.resolved_id: float(candidate.parameter_count)
        for candidate in eligible
        if candidate.parameter_count is not None and candidate.resolved_id is not None
    }
    mobile_scores = inverse_min_max_normalize(mobile_inputs)
    finetune_scores = inverse_min_max_normalize(mobile_inputs)

    dimension_maps: dict[str, dict[str, float]] = {
        "instruction_following": {
            candidate.resolved_id: candidate.metrics["ifeval_strict"]
            for candidate in eligible
            if candidate.resolved_id and candidate.metrics.get("ifeval_strict") is not None
        },
        "reasoning_quality": {
            candidate.resolved_id: average_available(
                [
                    candidate.metrics.get("bbh"),
                    candidate.metrics.get("gpqa"),
                    candidate.metrics.get("math_hard"),
                    candidate.metrics.get("musr"),
                    candidate.metrics.get("mmlu_pro"),
                ]
            )
            for candidate in eligible
            if candidate.resolved_id
        },
        "hallucination_proxy": {
            candidate.resolved_id: average_available(
                [candidate.metrics.get("gpqa"), candidate.metrics.get("mmlu_pro")]
            )
            for candidate in eligible
            if candidate.resolved_id
        },
        "emotional_awareness": {
            candidate.resolved_id: candidate.eq_bench
            for candidate in eligible
            if candidate.resolved_id and candidate.eq_bench is not None
        },
        "mobile_deployment": mobile_scores,
        "fine_tuning_efficiency": finetune_scores,
    }

    available_dimensions = {
        name for name, values in dimension_maps.items() if any(value is not None for value in values.values())
    }

    for candidate in candidates:
        if not candidate.found or candidate.resolved_id is None:
            candidate.filter_reason = "not_found"
            continue

        if candidate.license not in APPROVED_MODEL_LICENSES:
            candidate.filter_reason = f"license={candidate.license or 'unknown'}"
            continue

        if candidate.parameter_count is not None and candidate.parameter_count > MAX_PARAMETER_COUNT:
            candidate.filter_reason = "over_parameter_budget"
            continue

        if candidate.metrics.get("ifeval_strict") is None:
            candidate.filter_reason = "missing_leaderboard_metrics"
            continue

        contributions: list[float] = []
        total_weight = 0.0

        for dimension, weight in MODEL_WEIGHTS.items():
            if dimension not in available_dimensions:
                continue

            score_map = dimension_maps[dimension]
            raw_value = score_map.get(candidate.resolved_id)
            if raw_value is None:
                raw_value = 0.0
            contributions.append(weight * raw_value)
            total_weight += weight

        candidate.weighted_score = (sum(contributions) / total_weight) if total_weight else 0.0

    return candidates


def compare_models(
    *,
    model_ids: list[str],
    reports_dir: Path | None = None,
    write_decision: bool = True,
) -> Path:
    """Produce a ranked markdown comparison report and optionally log the winner."""

    reports_dir = reports_dir or PROJECT_ROOT / "research" / "model_comparison" / "reports"
    ensure_directory(reports_dir)

    metadata_report = run_model_search(model_ids=model_ids, reports_dir=reports_dir)
    resolved_ids = [
        item.get("resolved_id")
        for item in read_json(metadata_report).get("models", [])
        if item.get("resolved_id")
    ]
    benchmark_report = run_model_benchmarks(model_ids=resolved_ids, reports_dir=reports_dir)
    candidates = score_model_candidates(
        model_candidates_from_reports(metadata_report=metadata_report, benchmark_report=benchmark_report)
    )

    ranked = sorted(
        candidates,
        key=lambda item: (item.weighted_score is not None, item.weighted_score or -1.0),
        reverse=True,
    )
    winner = next(
        (candidate for candidate in ranked if candidate.filter_reason is None),
        None,
    )

    lines = [
        "# Model Comparison Report",
        "",
        f"Generated at: {timestamp_slug()}",
        "",
        "## Method notes",
        "",
        "- License is treated as a hard filter and limited to Apache-2.0 or MIT.",
        "- IFEval, BBH, GPQA, MATH Hard, MUSR, and MMLU-Pro are fetched from the public `open-llm-leaderboard/results` dataset when available.",
        "- Emotional awareness is a best-effort model-card extraction using EQ-Bench mentions when present.",
        "- Mobile deployment and fine-tuning efficiency are inferred from parameter counts, which is explicitly an inference rather than a benchmark.",
        "- Hallucination rate is proxied with knowledge-intensive benchmark performance because a consistent public small-model hallucination benchmark was not available from the queried leaderboard source.",
        "",
        "## Ranked candidates",
        "",
        "| Rank | Requested ID | Resolved ID | License | Params | IFEval | BBH | GPQA | MATH | MUSR | MMLU-Pro | EQ | Score | Filter |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for index, candidate in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    candidate.requested_id,
                    candidate.resolved_id or "n/a",
                    candidate.license or "n/a",
                    human_params(candidate.parameter_count),
                    format_metric(candidate.metrics.get("ifeval_strict")),
                    format_metric(candidate.metrics.get("bbh")),
                    format_metric(candidate.metrics.get("gpqa")),
                    format_metric(candidate.metrics.get("math_hard")),
                    format_metric(candidate.metrics.get("musr")),
                    format_metric(candidate.metrics.get("mmlu_pro")),
                    format_metric(candidate.eq_bench),
                    format_metric(candidate.weighted_score),
                    candidate.filter_reason or "eligible",
                ]
            )
            + " |"
        )

    if winner is not None:
        lines.extend(
            [
                "",
                "## Selected base model",
                "",
                f"`{winner.resolved_id}` with weighted score `{winner.weighted_score:.3f}`.",
                "",
            ]
        )

        if write_decision:
            alternatives = ", ".join(
                candidate.resolved_id or candidate.requested_id for candidate in ranked[:3]
            )
            append_decision(
                title="Initial base model selection",
                phase="Phase 1 - Model Comparison",
                decision=f"Select `{winner.resolved_id}` as the initial BASE_MODEL.",
                alternatives=alternatives,
                justification=(
                    "The selected model passed the commercial-license filter and had the strongest "
                    "combined instruction-following and reasoning signal among the queried candidates, "
                    "while remaining small enough to support low-cost personalization workflows that "
                    "matter for retention."
                ),
                impact=(
                    "Training configuration, prompt formatting, export validation, and downstream "
                    "tokenizer assumptions should use the selected model until later research overturns it."
                ),
            )

    output_path = reports_dir / f"model_comparison_{timestamp_slug()}.md"
    write_text(output_path, "\n".join(lines) + "\n")
    return output_path


def dataset_detail_url(dataset_id: str) -> str:
    """Return the Hugging Face dataset detail URL."""

    return f"{HF_DATASETS_API}/{quote_path_component(dataset_id)}"


def fetch_dataset_samples(dataset_id: str) -> list[dict[str, Any]]:
    """Fetch a small sample of rows from the dataset server when available."""

    try:
        split_payload = fetch_json(
            f"{HF_DATASET_SPLITS_API}?dataset={urllib.parse.quote(dataset_id, safe='')}"
        )
    except Exception:
        return []

    splits = split_payload.get("splits", [])
    if not splits:
        return []

    sample_split = splits[0]
    params = urllib.parse.urlencode(
        {
            "dataset": dataset_id,
            "config": sample_split["config"],
            "split": sample_split["split"],
            "offset": 0,
            "length": 2,
        }
    )
    try:
        row_payload = fetch_json(f"{HF_DATASET_ROWS_API}?{params}")
    except Exception:
        return []

    return [row.get("row", {}) for row in row_payload.get("rows", [])]


def run_dataset_search(
    *,
    queries: list[str],
    limit_per_query: int = 5,
    reports_dir: Path | None = None,
) -> Path:
    """Search Hugging Face datasets for the requested queries and persist a report."""

    reports_dir = reports_dir or PROJECT_ROOT / "research" / "dataset_discovery" / "reports"
    ensure_directory(reports_dir)
    discovered: dict[str, dict[str, Any]] = {}

    for query in queries:
        url = f"{HF_DATASETS_API}?search={urllib.parse.quote(query)}&limit={limit_per_query}"
        for result in fetch_json(url):
            dataset_id = str(result["id"])
            if dataset_id not in discovered:
                try:
                    detail = fetch_json(dataset_detail_url(dataset_id))
                except Exception:
                    continue

                discovered[dataset_id] = {
                    "id": dataset_id,
                    "queries": [query],
                    "license": license_from_metadata(detail),
                    "tags": detail.get("tags", []),
                    "description": detail.get("description"),
                    "downloads": detail.get("downloads"),
                    "likes": detail.get("likes"),
                    "last_modified": detail.get("lastModified"),
                    "size_categories": [
                        tag.removeprefix("size_categories:")
                        for tag in detail.get("tags", [])
                        if isinstance(tag, str) and tag.startswith("size_categories:")
                    ],
                    "formats": [
                        tag.removeprefix("format:")
                        for tag in detail.get("tags", [])
                        if isinstance(tag, str) and tag.startswith("format:")
                    ],
                    "sample_rows": fetch_dataset_samples(dataset_id),
                    "dataset_url": f"https://huggingface.co/datasets/{quote_path_component(dataset_id)}",
                }
            else:
                discovered[dataset_id]["queries"].append(query)

    output_path = reports_dir / f"discovered_datasets_{timestamp_slug()}.json"
    write_json(
        output_path,
        {"generated_at": timestamp_slug(), "datasets": list(discovered.values())},
    )
    return output_path


def dataset_license_decision(license_name: str | None) -> tuple[bool, str]:
    """Return whether a dataset license should pass the commercial-use filter."""

    if not license_name:
        return False, "missing_license"

    normalized = license_name.lower()
    if normalized in APPROVED_DATASET_LICENSES:
        return True, "approved"
    if any(token in normalized for token in REJECTED_LICENSE_TOKENS):
        return False, f"rejected:{normalized}"
    return False, f"manual_review_needed:{normalized}"


def run_license_checker(
    *,
    source_report: Path | None = None,
    reports_dir: Path | None = None,
) -> Path:
    """Filter the latest discovered dataset report into an approved list."""

    reports_dir = reports_dir or PROJECT_ROOT / "research" / "dataset_discovery" / "reports"
    ensure_directory(reports_dir)
    source_report = source_report or latest_matching_file(
        reports_dir, "discovered_datasets_", ".json"
    )
    if source_report is None:
        raise FileNotFoundError("No discovered dataset report was found.")

    payload = read_json(source_report)
    approved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for dataset in payload.get("datasets", []):
        is_approved, rationale = dataset_license_decision(dataset.get("license"))
        annotated = dict(dataset)
        annotated["license_rationale"] = rationale
        if is_approved:
            approved.append(annotated)
        else:
            rejected.append(annotated)

    output_path = reports_dir / "approved_datasets.json"
    write_json(
        output_path,
        {
            "generated_at": timestamp_slug(),
            "source_report": str(source_report),
            "approved": approved,
            "rejected": rejected,
        },
    )
    return output_path


def arxiv_query_url(query: str, max_results: int) -> str:
    """Build an arXiv API URL with phrase-style search semantics."""

    encoded_query = urllib.parse.quote(f'all:"{query}"')
    return (
        f"{ARXIV_API}?search_query={encoded_query}&start=0&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )


def arxiv_fallback_query_url(query: str, max_results: int) -> str:
    """Build a broader arXiv API query when the exact phrase is too restrictive."""

    terms = [token for token in re.split(r"[^A-Za-z0-9]+", query.lower()) if len(token) >= 3]
    deduped_terms: list[str] = []
    for term in terms:
        if term not in deduped_terms:
            deduped_terms.append(term)
    boolean_query = " AND ".join(f"all:{term}" for term in deduped_terms[:5])
    encoded_query = urllib.parse.quote(boolean_query)
    return (
        f"{ARXIV_API}?search_query={encoded_query}&start=0&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )


def strip_html_tags(value: str) -> str:
    """Strip tags and collapse whitespace."""

    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_arxiv_conclusion(arxiv_id_with_version: str) -> tuple[str | None, str]:
    """Fetch the conclusion section from the HTML rendering when available."""

    html_url = f"https://arxiv.org/html/{arxiv_id_with_version}"
    try:
        body = fetch_text(html_url)
    except Exception:
        return None, "html_unavailable"

    match = re.search(
        r"<h[1-6][^>]*>\s*(?:\d+\.?\s*)?(?:Conclusion|Conclusions)\s*</h[1-6]>(.*?)<h[1-6]",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, "conclusion_not_found"

    conclusion = strip_html_tags(match.group(1))
    conclusion = textwrap.shorten(conclusion, width=1400, placeholder="...")
    return conclusion, "html_section"


def extract_sentences(text: str, limit: int = 2) -> str:
    """Return the first few sentence-like spans from a block of text."""

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:limit]).strip()


def relevance_statement(query: str, summary: str) -> str:
    """Generate a brief retention-oriented relevance statement."""

    query_lower = query.lower()
    summary_lower = summary.lower()

    if "emotion" in query_lower or "motivation" in query_lower:
        return "Relevant to retention because emotionally aware responses are critical in dropout-risk moments."
    if "memory" in query_lower or "personalization" in query_lower:
        return "Relevant to retention because user-specific continuity is the core product differentiator."
    if "retrieval" in query_lower or "hallucination" in query_lower:
        return "Relevant to retention because safer and better-grounded advice increases trust over time."
    if "coaching" in summary_lower:
        return "Relevant to retention because it informs how the system should coach rather than merely answer questions."
    return "Relevant to retention because it informs model behavior or scientific grounding that affects long-term trust."


def paper_matches_query_theme(query: str, paper: dict[str, Any]) -> bool:
    """Filter obviously off-domain papers from broad arXiv fallback queries."""

    haystack = f"{paper['title']} {paper['summary']}".lower()
    query_lower = query.lower()

    theme_requirements: list[str] = []
    if any(token in query_lower for token in ("memory", "personalization", "event sourcing")):
        theme_requirements = [
            "convers",
            "dialog",
            "language model",
            "llm",
            "assistant",
            "agent",
            "user",
            "retrieval",
        ]
    elif any(token in query_lower for token in ("health", "fitness", "nutrition", "exercise")):
        theme_requirements = [
            "health",
            "fitness",
            "exercise",
            "nutrition",
            "coaching",
            "medical",
        ]
    elif any(token in query_lower for token in ("hallucination", "retrieval", "generation")):
        theme_requirements = [
            "language model",
            "llm",
            "generation",
            "retrieval",
            "question answering",
            "reasoning",
        ]
    elif any(token in query_lower for token in ("emotion", "coaching", "motivation", "retention")):
        theme_requirements = [
            "emotion",
            "empathy",
            "coaching",
            "dialog",
            "conversation",
            "motivation",
            "behavior",
            "counsel",
        ]

    if theme_requirements and not any(token in haystack for token in theme_requirements):
        return False

    banned_terms = (
        "visual tracking",
        "rgb-event",
        "video tracking",
        "object tracking",
        "sensor fusion",
        "image generation",
    )
    return not any(term in haystack for term in banned_terms)


def parse_arxiv_feed(xml_text: str) -> list[dict[str, Any]]:
    """Parse a subset of the arXiv Atom feed into dictionaries."""

    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    root = etree.fromstring(xml_text)
    papers: list[dict[str, Any]] = []

    for entry in root.findall("atom:entry", namespace):
        identifier = entry.findtext("atom:id", default="", namespaces=namespace)
        published = entry.findtext("atom:published", default="", namespaces=namespace)
        if not published.startswith(tuple(str(year) for year in range(MIN_PAPER_YEAR, 2100))):
            continue

        title = entry.findtext("atom:title", default="", namespaces=namespace).strip()
        summary = entry.findtext("atom:summary", default="", namespaces=namespace).strip()
        authors = [
            author.findtext("atom:name", default="", namespaces=namespace).strip()
            for author in entry.findall("atom:author", namespace)
        ]
        arxiv_id_with_version = identifier.rsplit("/", 1)[-1]
        papers.append(
            {
                "id": identifier,
                "arxiv_id": arxiv_id_with_version,
                "title": title,
                "published": published,
                "authors": authors,
                "summary": summary,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id_with_version}.pdf",
                "abs_url": f"https://arxiv.org/abs/{arxiv_id_with_version}",
            }
        )

    return papers


def run_literature_search(
    *,
    queries: list[str],
    max_results_per_query: int = 3,
    summaries_dir: Path | None = None,
) -> Path:
    """Search arXiv, write per-paper summaries, and append an aggregate report."""

    summaries_dir = summaries_dir or PROJECT_ROOT / "research" / "literature" / "summaries"
    ensure_directory(summaries_dir)

    unique_papers: dict[str, dict[str, Any]] = {}
    for query in queries:
        feeds = [fetch_text(arxiv_query_url(query, max_results=max_results_per_query))]
        primary_papers = parse_arxiv_feed(feeds[0])
        if not primary_papers:
            feeds.append(fetch_text(arxiv_fallback_query_url(query, max_results=max_results_per_query)))
        query_papers: list[dict[str, Any]] = []
        for feed in feeds:
            query_papers.extend(parse_arxiv_feed(feed))

        filtered_papers = [paper for paper in query_papers if paper_matches_query_theme(query, paper)]
        for paper in filtered_papers:
            if paper["arxiv_id"] in unique_papers:
                unique_papers[paper["arxiv_id"]]["queries"].append(query)
                continue

            conclusion, conclusion_source = fetch_arxiv_conclusion(paper["arxiv_id"])
            if conclusion is None:
                conclusion = extract_sentences(paper["summary"], limit=2)

            record = dict(paper)
            record["queries"] = [query]
            record["conclusion"] = conclusion
            record["conclusion_source"] = conclusion_source
            record["key_findings"] = extract_sentences(paper["summary"], limit=2)
            record["relevance"] = relevance_statement(query, paper["summary"])
            unique_papers[paper["arxiv_id"]] = record

    for paper in unique_papers.values():
        filename = safe_slug(f"{paper['published']}_{paper['title']}") + ".md"
        content = "\n".join(
            [
                f"# {paper['title']}",
                "",
                f"- Published: {paper['published']}",
                f"- Authors: {', '.join(paper['authors'])}",
                f"- Queries matched: {', '.join(paper['queries'])}",
                f"- Abstract page: {paper['abs_url']}",
                f"- PDF: {paper['pdf_url']}",
                "",
                "## Abstract",
                "",
                paper["summary"],
                "",
                "## Key findings",
                "",
                paper["key_findings"],
                "",
                "## Conclusion",
                "",
                paper["conclusion"],
                "",
                "## Relevance to 30-day retention",
                "",
                paper["relevance"],
                "",
            ]
        )
        write_text(summaries_dir / filename, content)

    report_lines = [
        "",
        f"## Milestone {date.today().isoformat()}: Literature scan",
        "",
        f"Generated {len(unique_papers)} unique paper summaries from {len(queries)} query themes.",
        "",
    ]
    for paper in sorted(unique_papers.values(), key=lambda item: item["published"], reverse=True):
        report_lines.append(
            f"- `{paper['published'][:10]}` [{paper['title']}]({paper['abs_url']}): {paper['relevance']}"
        )
    report_lines.append("")
    append_text(PROJECT_ROOT / "RESEARCH_REPORT.md", "\n".join(report_lines))

    output_path = summaries_dir / f"literature_index_{timestamp_slug()}.json"
    write_json(
        output_path,
        {"generated_at": timestamp_slug(), "papers": list(unique_papers.values())},
    )
    return output_path
