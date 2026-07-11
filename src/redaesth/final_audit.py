"""Composition audit helpers for the locked final training dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import RedAesthConfig, config
from .dataset_pipeline import write_json_file


AUDIT_REPORT_VERSION = 1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def distribution(counter: Counter[str], total_records: int) -> dict[str, dict[str, float | int]]:
    """Convert a counter into counts plus percentage shares."""

    result: dict[str, dict[str, float | int]] = {}
    for key, count in sorted(counter.items()):
        share = (count / total_records) if total_records else 0.0
        result[key] = {
            "count": count,
            "share": round(share, 4),
            "percentage": round(share * 100, 2),
        }
    return result


def metric_result(
    *,
    actual: float,
    threshold: float,
    comparison: str,
) -> dict[str, Any]:
    """Build a normalized PASS/FAIL payload for one audit check."""

    if comparison == "minimum":
        passed = actual >= threshold
    elif comparison == "maximum":
        passed = actual <= threshold
    else:
        raise ValueError(f"Unsupported comparison: {comparison}")

    return {
        "status": "PASS" if passed else "FAIL",
        "comparison": comparison,
        "actual_share": round(actual, 4),
        "actual_percentage": round(actual * 100, 2),
        "threshold_share": round(threshold, 4),
        "threshold_percentage": round(threshold * 100, 2),
    }


def share_for(counter: Counter[str], key: str, total_records: int) -> float:
    """Look up one label share from a counter."""

    if not total_records:
        return 0.0
    return counter.get(key, 0) / total_records


def exact_duplicate_rate(records: list[dict[str, Any]]) -> float | None:
    """Estimate exact duplicate rate from preserved normalized hashes."""

    hashes = [str(record["normalized_sha256"]) for record in records if record.get("normalized_sha256")]
    if not hashes:
        return None
    return (len(hashes) - len(set(hashes))) / len(hashes)


def audit_final_dataset(
    *,
    config: RedAesthConfig = config,
    final_dataset_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Audit the locked final dataset against configured readiness thresholds."""

    source_path = config.resolve_path(final_dataset_path or config.final_dataset_path)
    report_path = config.resolve_path(output_path or config.final_composition_audit_path)
    records = read_jsonl(source_path)
    if not records:
        raise RuntimeError(f"No final dataset records were found in {source_path}")

    total_records = len(records)
    language_counts = Counter(str(record.get("language", "unknown")) for record in records)
    domain_counts = Counter(str(record.get("domain", "unknown")) for record in records)
    source_counts = Counter(str(record.get("source_id", "unknown")) for record in records)

    duplicate_rate = exact_duplicate_rate(records)
    max_source_share = max((count / total_records) for count in source_counts.values()) if source_counts else 0.0

    checks = {
        "language_mostly_ascii": metric_result(
            actual=share_for(language_counts, "mostly_ascii", total_records),
            threshold=config.minimum_mostly_ascii_share,
            comparison="minimum",
        ),
        "language_majority_non_ascii": metric_result(
            actual=share_for(language_counts, "majority_non_ascii", total_records),
            threshold=config.maximum_majority_non_ascii_share,
            comparison="maximum",
        ),
        "domain_mental_health_adjacent": metric_result(
            actual=share_for(domain_counts, "mental-health-adjacent", total_records),
            threshold=config.maximum_mental_health_adjacent_share,
            comparison="maximum",
        ),
        "domain_off_domain": metric_result(
            actual=share_for(domain_counts, "off-domain", total_records),
            threshold=config.maximum_off_domain_share,
            comparison="maximum",
        ),
        "source_max_single_source": metric_result(
            actual=max_source_share,
            threshold=config.maximum_single_source_share,
            comparison="maximum",
        ),
    }

    if duplicate_rate is None:
        checks["exact_duplicate_rate"] = {
            "status": "UNVERIFIED",
            "actual_share": None,
            "actual_percentage": None,
            "threshold_share": round(config.maximum_exact_duplicate_rate, 4),
            "threshold_percentage": round(config.maximum_exact_duplicate_rate * 100, 2),
        }
    else:
        checks["exact_duplicate_rate"] = metric_result(
            actual=duplicate_rate,
            threshold=config.maximum_exact_duplicate_rate,
            comparison="maximum",
        )

    pass_fail_checks = [
        payload["status"] == "PASS"
        for payload in checks.values()
        if payload["status"] in {"PASS", "FAIL"}
    ]
    overall_status = "PASS" if all(pass_fail_checks) else "FAIL"

    report = {
        "report_version": AUDIT_REPORT_VERSION,
        "final_dataset_path": str(source_path),
        "total_samples": total_records,
        "distributions": {
            "language": distribution(language_counts, total_records),
            "domain": distribution(domain_counts, total_records),
            "source": distribution(source_counts, total_records),
        },
        "deduplication": {
            "signal_present": duplicate_rate is not None,
            "exact_duplicate_rate": None if duplicate_rate is None else round(duplicate_rate, 4),
            "exact_duplicate_percentage": None if duplicate_rate is None else round(duplicate_rate * 100, 2),
        },
        "checks": checks,
        "overall_status": overall_status,
    }
    return write_json_file(report_path, report), report
