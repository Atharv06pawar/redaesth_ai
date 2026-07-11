"""Split and validation helpers for the locked final training dataset."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RedAesthConfig, config
from .dataset_pipeline import (
    build_manifest_file_record,
    count_jsonl_records,
    utc_timestamp,
    write_json_file,
)


FINAL_MANIFEST_VERSION = 1
SPOT_CHECK_REPORT_VERSION = 1


@dataclass(slots=True, frozen=True)
class FinalSplitResult:
    """Concrete output paths and counts for the final dataset split."""

    train_path: Path
    validation_path: Path
    test_path: Path
    train_count: int
    validation_count: int
    test_count: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write records to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


def deterministic_shuffle(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Return a deterministically shuffled copy of the input records."""

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def stable_group_seed(source_id: str, seed: int) -> int:
    """Derive a deterministic integer seed for one source group."""

    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def compute_split_targets(total_records: int, config: RedAesthConfig) -> tuple[int, int, int]:
    """Compute deterministic split targets using configured ratios and minimums."""

    validation_target = max(
        config.final_min_validation_examples,
        round(total_records * config.final_validation_ratio),
    )
    test_target = max(
        config.final_min_test_examples,
        round(total_records * config.final_test_ratio),
    )
    train_target = total_records - validation_target - test_target
    if train_target < config.final_min_train_examples:
        raise RuntimeError(
            "Final dataset does not meet minimum size requirements: "
            f"train={train_target}, validation={validation_target}, test={test_target}"
        )
    return train_target, validation_target, test_target


def allocate_counts(
    *,
    group_sizes: dict[str, int],
    target_total: int,
    ratio: float,
    reserved_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    """Allocate one split count across sources using largest-remainder apportionment."""

    reserved = reserved_counts or {}
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    total_allocated = 0

    for source_id, group_size in group_sizes.items():
        available = group_size - reserved.get(source_id, 0)
        if available <= 0:
            allocations[source_id] = 0
            remainders.append((0.0, source_id))
            continue
        raw_target = group_size * ratio
        base_count = min(int(raw_target), available)
        allocations[source_id] = base_count
        total_allocated += base_count
        remainders.append((raw_target - base_count, source_id))

    for _, source_id in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if total_allocated >= target_total:
            break
        available = group_sizes[source_id] - reserved.get(source_id, 0) - allocations[source_id]
        if available <= 0:
            continue
        allocations[source_id] += 1
        total_allocated += 1

    if total_allocated < target_total:
        for source_id, _ in sorted(group_sizes.items(), key=lambda item: (-item[1], item[0])):
            if total_allocated >= target_total:
                break
            available = group_sizes[source_id] - reserved.get(source_id, 0) - allocations[source_id]
            while available > 0 and total_allocated < target_total:
                allocations[source_id] += 1
                total_allocated += 1
                available -= 1

    if total_allocated != target_total:
        raise RuntimeError(
            "Could not allocate deterministic split counts for the final dataset: "
            f"target_total={target_total}, allocated={total_allocated}"
        )

    return allocations


def stratified_split_records(
    records: list[dict[str, Any]],
    *,
    config: RedAesthConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records deterministically while preserving source proportions."""

    if not records:
        raise RuntimeError("Cannot split an empty final dataset.")

    grouped_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        source_id = str(record.get("source_id", "unknown"))
        grouped_records.setdefault(source_id, []).append(record)

    shuffled_groups = {
        source_id: deterministic_shuffle(group, stable_group_seed(source_id, config.seed))
        for source_id, group in sorted(grouped_records.items())
    }
    group_sizes = {source_id: len(group) for source_id, group in shuffled_groups.items()}
    train_target, validation_target, test_target = compute_split_targets(len(records), config)

    validation_counts = allocate_counts(
        group_sizes=group_sizes,
        target_total=validation_target,
        ratio=config.final_validation_ratio,
    )
    test_counts = allocate_counts(
        group_sizes=group_sizes,
        target_total=test_target,
        ratio=config.final_test_ratio,
        reserved_counts=validation_counts,
    )

    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    for source_id, group in shuffled_groups.items():
        validation_count = validation_counts[source_id]
        test_count = test_counts[source_id]
        validation_records.extend(group[:validation_count])
        test_records.extend(group[validation_count : validation_count + test_count])
        train_records.extend(group[validation_count + test_count :])

    train_records = deterministic_shuffle(train_records, config.seed + 11)
    validation_records = deterministic_shuffle(validation_records, config.seed + 17)
    test_records = deterministic_shuffle(test_records, config.seed + 23)

    if (
        len(train_records) != train_target
        or len(validation_records) != validation_target
        or len(test_records) != test_target
    ):
        raise RuntimeError(
            "Final split sizes do not match configured targets: "
            f"train={len(train_records)}/{train_target}, "
            f"validation={len(validation_records)}/{validation_target}, "
            f"test={len(test_records)}/{test_target}"
        )

    return train_records, validation_records, test_records


def split_final_dataset(
    *,
    config: RedAesthConfig = config,
    final_dataset_path: Path | None = None,
) -> FinalSplitResult:
    """Split the locked final dataset and write train/val/test JSONL files."""

    source_path = config.resolve_path(final_dataset_path or config.final_dataset_path)
    records = read_jsonl(source_path)
    train_records, validation_records, test_records = stratified_split_records(records, config=config)

    train_path = write_jsonl(config.final_train_path, train_records)
    validation_path = write_jsonl(config.final_validation_path, validation_records)
    test_path = write_jsonl(config.final_test_path, test_records)
    return FinalSplitResult(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        train_count=len(train_records),
        validation_count=len(validation_records),
        test_count=len(test_records),
    )


def build_final_manifest(
    *,
    config: RedAesthConfig = config,
    final_dataset_path: Path | None = None,
    split_result: FinalSplitResult,
    composition_audit_path: Path | None = None,
    spot_check_report_path: Path | None = None,
) -> Path:
    """Write a SHA256-backed manifest for the locked dataset artifact and splits."""

    locked_dataset_path = config.resolve_path(final_dataset_path or config.final_dataset_path)
    audit_path = config.resolve_path(composition_audit_path or config.final_composition_audit_path)
    spot_check_path = config.resolve_path(spot_check_report_path or config.final_spot_check_report_path)

    manifest = {
        "manifest_version": FINAL_MANIFEST_VERSION,
        "generated_at": utc_timestamp(),
        "project_root": str(config.project_root),
        "base_model_id": config.base_model_id,
        "final_dataset": build_manifest_file_record(
            locked_dataset_path,
            relative_to=config.project_root,
            sample_count=count_jsonl_records(locked_dataset_path),
        ),
        "splits": {
            "train": build_manifest_file_record(
                split_result.train_path,
                relative_to=config.project_root,
                sample_count=split_result.train_count,
            ),
            "val": build_manifest_file_record(
                split_result.validation_path,
                relative_to=config.project_root,
                sample_count=split_result.validation_count,
            ),
            "test": build_manifest_file_record(
                split_result.test_path,
                relative_to=config.project_root,
                sample_count=split_result.test_count,
            ),
        },
        "composition_audit": str(audit_path.relative_to(config.project_root)).replace("\\", "/"),
        "spot_check_report": str(spot_check_path.relative_to(config.project_root)).replace("\\", "/"),
    }
    return write_json_file(config.final_manifest_path, manifest)


def token_count_for_text(tokenizer: Any, text: str) -> int:
    """Count tokens for one formatted training sample."""

    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def spot_check_training_split(
    *,
    config: RedAesthConfig = config,
    train_path: Path | None = None,
    tokenizer: Any,
) -> tuple[Path, dict[str, Any]]:
    """Validate a deterministic sample of training rows with the selected tokenizer."""

    source_path = config.resolve_path(train_path or config.final_train_path)
    records = read_jsonl(source_path)
    if not records:
        raise RuntimeError(f"No training split records were found in {source_path}")

    sample_size = min(config.final_spot_check_sample_size, len(records))
    sample_indices = sorted(random.Random(config.seed + 101).sample(range(len(records)), sample_size))
    sampled_records = [records[index] for index in sample_indices]

    malformed_record_ids: list[str] = []
    over_length_record_ids: list[str] = []
    max_observed_tokens = 0
    observed_token_counts: list[int] = []

    for record in sampled_records:
        record_id = str(record.get("record_id", "unknown"))
        text = record.get("text")
        conversations = record.get("conversations")
        if not isinstance(text, str) or not text.strip():
            malformed_record_ids.append(record_id)
            continue
        if not isinstance(conversations, list) or not conversations:
            malformed_record_ids.append(record_id)
            continue

        expected_text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )
        if text != expected_text:
            malformed_record_ids.append(record_id)
            continue

        token_count = token_count_for_text(tokenizer, text)
        observed_token_counts.append(token_count)
        max_observed_tokens = max(max_observed_tokens, token_count)
        if token_count > config.max_seq_length:
            over_length_record_ids.append(record_id)

    truncation_rate = len(over_length_record_ids) / sample_size if sample_size else 0.0
    report = {
        "report_version": SPOT_CHECK_REPORT_VERSION,
        "train_path": str(source_path),
        "sample_size": sample_size,
        "max_seq_length": config.max_seq_length,
        "average_observed_tokens": (
            round(sum(observed_token_counts) / len(observed_token_counts), 2)
            if observed_token_counts
            else 0.0
        ),
        "max_observed_tokens": max_observed_tokens,
        "malformed_record_count": len(malformed_record_ids),
        "malformed_record_ids": malformed_record_ids,
        "over_length_record_count": len(over_length_record_ids),
        "over_length_record_ids": over_length_record_ids,
        "truncation_rate": round(truncation_rate, 4),
        "truncation_percentage": round(truncation_rate * 100, 2),
        "warning": truncation_rate > config.final_spot_check_truncation_warning_threshold,
        "status": "PASS" if not malformed_record_ids else "FAIL",
    }
    return write_json_file(config.final_spot_check_report_path, report), report
