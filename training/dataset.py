"""Schema validation and deterministic tokenization for calibration JSONL artifacts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from redaesth.config import RedAesthConfig, config as default_config


REQUIRED_RECORD_FIELDS = (
    "record_id",
    "source_id",
    "language",
    "domain",
    "conversations",
    "text",
)
VALID_ROLES = {"system", "user", "assistant"}


@dataclass(slots=True, frozen=True)
class TrainingExample:
    """One schema-valid, tokenizer-ready calibration sample."""

    record_id: str
    conversations: list[dict[str, str]]
    text: str


@dataclass(slots=True)
class DatasetLoadReport:
    """Counts and malformed-record details from a training-data load."""

    train_path: Path
    validation_path: Path | None
    malformed_records: list[str] = field(default_factory=list)
    train_count: int = 0
    validation_count: int = 0
    validation_source: str = "provided"


class DatasetSchemaError(ValueError):
    """Raised when training JSONL contains malformed or chat-template-drifted samples."""

    def __init__(self, report: DatasetLoadReport) -> None:
        super().__init__(f"Malformed calibration records: {', '.join(report.malformed_records)}")
        self.report = report


class TokenizedTrainingDataset(Sequence[dict[str, list[int]]]):
    """Small in-memory dataset compatible with the Hugging Face Trainer data collator."""

    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)


@dataclass(slots=True)
class CalibrationDatasets:
    """Tokenized train/validation datasets plus the data-quality load report."""

    train_dataset: TokenizedTrainingDataset
    validation_dataset: TokenizedTrainingDataset
    report: DatasetLoadReport


def validate_training_record(record: dict[str, Any], *, tokenizer: Any) -> TrainingExample:
    """Validate the locked training schema and verify tokenizer-template consistency."""

    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    record_id = record["record_id"]
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("record_id must be a non-empty string")
    conversations = record["conversations"]
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations must be a non-empty list")
    normalized_messages: list[dict[str, str]] = []
    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError("conversation entries must be objects")
        role = message.get("role")
        content = message.get("content")
        if role not in VALID_ROLES or not isinstance(content, str) or not content.strip():
            raise ValueError("conversation entries require a supported role and non-empty content")
        normalized_messages.append({"role": role, "content": content})
    if normalized_messages[-1]["role"] != "assistant":
        raise ValueError("training conversations must end with an assistant response")
    text = record["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    expected_text = tokenizer.apply_chat_template(
        normalized_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if text != expected_text:
        raise ValueError("text does not match the selected tokenizer chat template")
    return TrainingExample(record_id=record_id, conversations=normalized_messages, text=text)


def iter_training_examples(path: Path, *, tokenizer: Any) -> Iterator[TrainingExample]:
    """Stream valid examples from JSONL for callers that do not need shuffle or holdout."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                yield validate_training_record(payload, tokenizer=tokenizer)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path.name}:{line_number}: {exc}") from exc


def load_training_examples(
    path: Path,
    *,
    tokenizer: Any,
    report: DatasetLoadReport,
) -> list[TrainingExample]:
    """Load one JSONL artifact and collect every malformed row before failing closed."""

    examples: list[TrainingExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                examples.append(validate_training_record(payload, tokenizer=tokenizer))
            except (json.JSONDecodeError, ValueError):
                report.malformed_records.append(f"{path.name}:{line_number}")
    if report.malformed_records:
        raise DatasetSchemaError(report)
    if not examples:
        raise ValueError(f"No valid calibration records found in {path}")
    return examples


def deterministic_holdout(
    examples: list[TrainingExample],
    *,
    ratio: float,
    seed: int,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Create an in-memory validation set without mutating the locked source corpus."""

    if not 0.0 < ratio < 1.0:
        raise ValueError("calibration_validation_holdout_ratio must be between zero and one")
    if len(examples) < 2:
        raise ValueError("At least two records are required for an in-memory validation holdout")
    ordered = sorted(examples, key=lambda example: example.record_id)
    random.Random(seed).shuffle(ordered)
    validation_count = max(1, min(len(ordered) - 1, round(len(ordered) * ratio)))
    return ordered[validation_count:], ordered[:validation_count]


def tokenize_examples(
    examples: list[TrainingExample],
    *,
    tokenizer: Any,
    max_seq_length: int,
) -> TokenizedTrainingDataset:
    """Tokenize pre-rendered chat samples for causal language-model training."""

    rows: list[dict[str, list[int]]] = []
    for example in examples:
        encoded = tokenizer(
            example.text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        input_ids = list(encoded["input_ids"])
        if not input_ids:
            raise ValueError(f"Tokenizer produced an empty input for {example.record_id}")
        row = {"input_ids": input_ids, "labels": list(input_ids)}
        if "attention_mask" in encoded:
            row["attention_mask"] = list(encoded["attention_mask"])
        rows.append(row)
    return TokenizedTrainingDataset(rows)


def load_calibration_datasets(
    *,
    tokenizer: Any,
    config: RedAesthConfig = default_config,
    train_path: Path | None = None,
    validation_path: Path | None = None,
) -> CalibrationDatasets:
    """Load explicit JSONL splits or a deterministic in-memory holdout from production data."""

    resolved_train_path = config.resolve_path(train_path or config.calibration_train_path)
    resolved_validation_path = validation_path or config.calibration_validation_path
    if resolved_validation_path is not None:
        resolved_validation_path = config.resolve_path(resolved_validation_path)
    report = DatasetLoadReport(
        train_path=resolved_train_path,
        validation_path=resolved_validation_path,
    )
    train_examples = load_training_examples(resolved_train_path, tokenizer=tokenizer, report=report)
    if resolved_validation_path is None:
        train_examples, validation_examples = deterministic_holdout(
            train_examples,
            ratio=config.calibration_validation_holdout_ratio,
            seed=config.seed,
        )
        report.validation_source = "deterministic_in_memory_holdout"
    else:
        validation_examples = load_training_examples(
            resolved_validation_path,
            tokenizer=tokenizer,
            report=report,
        )
    report.train_count = len(train_examples)
    report.validation_count = len(validation_examples)
    return CalibrationDatasets(
        train_dataset=tokenize_examples(
            train_examples,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
        ),
        validation_dataset=tokenize_examples(
            validation_examples,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
        ),
        report=report,
    )
