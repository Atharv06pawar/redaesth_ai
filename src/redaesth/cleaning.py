"""Dataset cleaning and normalization helpers for the Phase 2 data pipeline."""

from __future__ import annotations

import hashlib
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import RedAesthConfig, config


CLEANING_REPORT_VERSION = 1
SUPPORTED_DATA_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet"}
CONVERSATION_KEY_CANDIDATES = ("conversations", "messages", "dialogue", "dialog", "chat")
ROLE_KEY_CANDIDATES = ("role", "speaker", "from", "author")
CONTENT_KEY_CANDIDATES = ("content", "text", "value", "message")
ROLE_ALIASES = {
    "assistant": "assistant",
    "bot": "assistant",
    "coach": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "user": "user",
    "human": "user",
    "client": "user",
    "system": "system",
    "developer": "system",
}
MIN_REQUIRED_MESSAGES = 2
MAX_MESSAGE_CHARACTERS = 4_000
MAX_CONVERSATION_CHARACTERS = 12_000
NON_ASCII_LANGUAGE_THRESHOLD = 0.5
INSTRUCT_TEXT_PATTERN = re.compile(
    r"(?:<s>\s*)?\[INST\](.*?)\[/INST\](.*?)(?:</s>)?$",
    flags=re.DOTALL | re.IGNORECASE,
)


class ConversationMessage(BaseModel):
    """One normalized conversational turn."""

    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Require non-empty message content after normalization."""

        if not value:
            raise ValueError("message content must not be empty")
        return value


class CleanedConversationRecord(BaseModel):
    """Normalized conversation ready for downstream deduplication and scoring."""

    conversation_id: str
    dataset_id: str
    source_file: str
    source_record_index: int = Field(ge=0)
    source_license: str | None = None
    source_dataset_url: str | None = None
    conversations: list[ConversationMessage] = Field(min_length=MIN_REQUIRED_MESSAGES)
    turn_count: int = Field(ge=MIN_REQUIRED_MESSAGES)
    user_turn_count: int = Field(ge=1)
    assistant_turn_count: int = Field(ge=1)
    contains_system_message: bool = False
    language_hint: str
    majority_non_ascii: bool
    conversation_characters: int = Field(ge=1)
    quality_flags: list[str] = Field(default_factory=list)
    normalized_sha256: str


@dataclass(slots=True)
class DatasetCleaningSummary:
    """Aggregate counts for one dataset while it is being cleaned."""

    dataset_id: str
    source_license: str | None
    source_dataset_url: str | None
    processed_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    input_records: int = 0
    kept_records: int = 0
    rejected_records: int = 0
    kept_turns: int = 0
    kept_characters: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    language_hints: Counter[str] = field(default_factory=Counter)
    quality_flags: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataset summary for the cleaning report."""

        average_turns = (self.kept_turns / self.kept_records) if self.kept_records else 0.0
        average_characters = (
            self.kept_characters / self.kept_records if self.kept_records else 0.0
        )
        return {
            "dataset_id": self.dataset_id,
            "source_license": self.source_license,
            "source_dataset_url": self.source_dataset_url,
            "processed_files": self.processed_files,
            "skipped_files": self.skipped_files,
            "input_records": self.input_records,
            "kept_records": self.kept_records,
            "rejected_records": self.rejected_records,
            "average_turns_kept": average_turns,
            "average_characters_kept": average_characters,
            "rejection_reasons": dict(self.rejection_reasons),
            "language_hints": dict(self.language_hints),
            "quality_flags": dict(self.quality_flags),
        }


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def read_json_file(path: Path) -> Any:
    """Read a UTF-8 JSON file from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, payload: Any) -> Path:
    """Write a UTF-8 JSON file with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def normalize_text(value: Any) -> str:
    """Normalize message content without destroying meaningful newlines."""

    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    normalized_lines = [line.strip() for line in text.split("\n")]
    return "\n".join(normalized_lines).strip()


def normalize_role(value: Any) -> str | None:
    """Map heterogeneous role labels into the canonical chat roles."""

    normalized = normalize_text(value).lower()
    if not normalized:
        return None
    return ROLE_ALIASES.get(normalized)


def lower_key_lookup(record: dict[str, Any]) -> dict[str, Any]:
    """Build a case-insensitive lookup view over a record."""

    return {str(key).lower(): value for key, value in record.items()}


def first_string_value(record: dict[str, Any], keys: Iterable[str]) -> str | None:
    """Return the first non-empty string-like value for the given keys."""

    lowered = lower_key_lookup(record)
    for key in keys:
        value = lowered.get(key.lower())
        if value is None:
            continue
        normalized = normalize_text(value)
        if normalized:
            return normalized
    return None


def build_user_text(record: dict[str, Any]) -> str | None:
    """Extract or synthesize the user side of a paired example."""

    primary = first_string_value(
        record,
        ("instruction", "prompt", "question", "query", "user", "human", "context"),
    )
    if primary is None:
        return None

    extra_input = first_string_value(record, ("input",))
    if extra_input and extra_input != primary:
        return f"{primary}\n\nContext:\n{extra_input}"
    return primary


def extract_messages_from_list(raw_messages: list[Any]) -> list[ConversationMessage] | None:
    """Normalize a list-style chat payload into canonical message objects."""

    normalized_messages: list[ConversationMessage] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            return None

        role_value = first_string_value(raw_message, ROLE_KEY_CANDIDATES)
        content_value = first_string_value(raw_message, CONTENT_KEY_CANDIDATES)
        role = normalize_role(role_value) if role_value is not None else None
        if role is None or content_value is None:
            return None

        try:
            normalized_messages.append(
                ConversationMessage(role=role, content=content_value)
            )
        except ValidationError:
            return None

    return normalized_messages


def extract_messages_from_record(record: dict[str, Any]) -> list[ConversationMessage] | None:
    """Extract a normalized conversation from a heterogeneous source record."""

    lowered = lower_key_lookup(record)
    for key in CONVERSATION_KEY_CANDIDATES:
        value = lowered.get(key)
        if isinstance(value, list):
            return extract_messages_from_list(value)

    user_text = build_user_text(record)
    assistant_text = first_string_value(record, ("output", "response", "answer", "assistant", "gpt"))
    if user_text is None or assistant_text is None:
        text_blob = first_string_value(record, ("text",))
        if text_blob is None:
            return None
        match = INSTRUCT_TEXT_PATTERN.match(text_blob)
        if match is None:
            return None
        user_text = normalize_text(match.group(1))
        assistant_text = normalize_text(match.group(2))
        if not user_text or not assistant_text:
            return None

    try:
        return [
            ConversationMessage(role="user", content=user_text),
            ConversationMessage(role="assistant", content=assistant_text),
        ]
    except ValidationError:
        return None


def validate_turn_sequence(messages: list[ConversationMessage]) -> str | None:
    """Return a rejection code when the conversation sequence is invalid."""

    if len(messages) < MIN_REQUIRED_MESSAGES:
        return "too_few_messages"

    system_count = sum(message.role == "system" for message in messages)
    if system_count > 1:
        return "multiple_system_messages"
    if system_count == 1 and messages[0].role != "system":
        return "system_message_not_leading"

    non_system_messages = [message for message in messages if message.role != "system"]
    if not non_system_messages:
        return "no_user_assistant_turns"
    if non_system_messages[0].role != "user":
        return "conversation_must_start_with_user"
    if non_system_messages[-1].role != "assistant":
        return "conversation_must_end_with_assistant"

    previous_role: str | None = None
    for message in non_system_messages:
        if previous_role == message.role:
            return "non_alternating_roles"
        previous_role = message.role

    return None


def detect_language_hint(messages: list[ConversationMessage]) -> tuple[str, bool]:
    """Infer a coarse language hint from Unicode distribution."""

    content = "".join(message.content for message in messages)
    significant_characters = [character for character in content if not character.isspace()]
    if not significant_characters:
        return "unknown", False

    non_ascii_ratio = sum(ord(character) > 127 for character in significant_characters) / len(
        significant_characters
    )
    majority_non_ascii = non_ascii_ratio > NON_ASCII_LANGUAGE_THRESHOLD
    if majority_non_ascii:
        return "majority_non_ascii", True
    return "mostly_ascii", False


def conversation_character_count(messages: list[ConversationMessage]) -> int:
    """Count the total characters across all conversation messages."""

    return sum(len(message.content) for message in messages)


def normalized_conversation_signature(messages: list[ConversationMessage]) -> str:
    """Render a stable text signature for hashing and exact deduplication."""

    return "\n".join(f"{message.role}::{message.content}" for message in messages)


def conversation_sha256(messages: list[ConversationMessage]) -> str:
    """Compute a deterministic SHA-256 signature for a normalized conversation."""

    signature = normalized_conversation_signature(messages)
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def clean_record(
    record: Any,
    *,
    dataset_id: str,
    source_file: str,
    source_record_index: int,
    source_license: str | None,
    source_dataset_url: str | None,
) -> tuple[CleanedConversationRecord | None, str | None]:
    """Normalize and validate one raw record."""

    if not isinstance(record, dict):
        return None, "record_not_object"

    messages = extract_messages_from_record(record)
    if messages is None:
        return None, "unable_to_extract_conversation"

    sequence_error = validate_turn_sequence(messages)
    if sequence_error is not None:
        return None, sequence_error

    if any(len(message.content) > MAX_MESSAGE_CHARACTERS for message in messages):
        return None, "message_too_long"

    character_count = conversation_character_count(messages)
    if character_count > MAX_CONVERSATION_CHARACTERS:
        return None, "conversation_too_long"

    language_hint, majority_non_ascii = detect_language_hint(messages)
    quality_flags: list[str] = []
    if majority_non_ascii:
        quality_flags.append("majority_non_ascii")
    if any(message.role == "system" for message in messages):
        quality_flags.append("contains_system_message")

    user_turn_count = sum(message.role == "user" for message in messages)
    assistant_turn_count = sum(message.role == "assistant" for message in messages)
    conversation_id = f"{dataset_id.replace('/', '__')}::{source_record_index:06d}"

    try:
        cleaned = CleanedConversationRecord(
            conversation_id=conversation_id,
            dataset_id=dataset_id,
            source_file=source_file,
            source_record_index=source_record_index,
            source_license=source_license,
            source_dataset_url=source_dataset_url,
            conversations=messages,
            turn_count=len(messages),
            user_turn_count=user_turn_count,
            assistant_turn_count=assistant_turn_count,
            contains_system_message=any(message.role == "system" for message in messages),
            language_hint=language_hint,
            majority_non_ascii=majority_non_ascii,
            conversation_characters=character_count,
            quality_flags=quality_flags,
            normalized_sha256=conversation_sha256(messages),
        )
    except ValidationError:
        return None, "schema_validation_failed"

    return cleaned, None


def iter_jsonl_records(path: Path) -> Iterator[Any]:
    """Yield records from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def iter_json_records(path: Path) -> Iterator[Any]:
    """Yield records from a JSON file that stores a list or a single object."""

    payload = read_json_file(path)
    if isinstance(payload, list):
        yield from payload
        return
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        yield from payload["data"]
        return
    yield payload


def iter_parquet_records(path: Path) -> Iterator[Any]:
    """Yield records from a Parquet file."""

    import pandas as pd

    frame = pd.read_parquet(path)
    frame = frame.where(frame.notna(), None)
    for record in frame.to_dict(orient="records"):
        yield record


def iter_csv_records(path: Path) -> Iterator[Any]:
    """Yield records from a CSV file."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            yield record


def iter_records_from_path(path: Path) -> Iterator[Any]:
    """Dispatch record loading based on file suffix."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from iter_csv_records(path)
        return
    if suffix == ".jsonl":
        yield from iter_jsonl_records(path)
        return
    if suffix == ".json":
        yield from iter_json_records(path)
        return
    if suffix == ".parquet":
        yield from iter_parquet_records(path)
        return
    raise ValueError(f"Unsupported dataset file suffix: {path}")


def write_jsonl_record(handle: Any, payload: dict[str, Any]) -> None:
    """Write one JSONL record using stable formatting."""

    handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def clean_raw_datasets(
    *,
    config: RedAesthConfig = config,
    raw_manifest_path: Path | None = None,
    dataset_ids: list[str] | None = None,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> tuple[Path, Path]:
    """Clean all supported raw datasets referenced by the raw manifest."""

    manifest_path = config.resolve_path(raw_manifest_path or config.raw_data_manifest_path)
    cleaned_output_path = config.resolve_path(output_path or config.cleaned_dataset_path)
    cleaning_report_path = config.resolve_path(report_path or config.cleaning_report_path)

    manifest = read_json_file(manifest_path)
    manifest_datasets = manifest.get("datasets")
    if not isinstance(manifest_datasets, list):
        raise ValueError(f"Raw manifest is malformed: {manifest_path}")

    selected_ids = {dataset_id.strip() for dataset_id in dataset_ids or [] if dataset_id.strip()}
    selected_datasets = [
        dataset
        for dataset in manifest_datasets
        if not selected_ids or str(dataset.get("id")) in selected_ids
    ]
    if not selected_datasets:
        raise ValueError(f"No raw datasets matched the requested dataset IDs in {manifest_path}")

    cleaned_output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries: list[DatasetCleaningSummary] = []
    total_input_records = 0
    total_kept_records = 0
    total_rejected_records = 0
    total_kept_turns = 0
    total_kept_characters = 0
    total_rejection_reasons: Counter[str] = Counter()
    total_language_hints: Counter[str] = Counter()
    total_quality_flags: Counter[str] = Counter()

    with cleaned_output_path.open("w", encoding="utf-8") as output_handle:
        for dataset in selected_datasets:
            dataset_id = str(dataset["id"])
            summary = DatasetCleaningSummary(
                dataset_id=dataset_id,
                source_license=dataset.get("license"),
                source_dataset_url=dataset.get("dataset_url"),
            )
            local_dir = Path(str(dataset["local_dir"]))
            file_records = dataset.get("files", [])
            supported_files: list[Path] = []
            for file_record in file_records:
                relative_path = Path(str(file_record["path"]))
                candidate_path = local_dir / relative_path
                if candidate_path.suffix.lower() in SUPPORTED_DATA_SUFFIXES:
                    supported_files.append(candidate_path)
                    summary.processed_files.append(str(relative_path).replace("\\", "/"))
                else:
                    summary.skipped_files.append(str(relative_path).replace("\\", "/"))

            if not supported_files:
                raise RuntimeError(
                    f"Dataset {dataset_id} has no supported data files in {manifest_path}"
                )

            for dataset_file in supported_files:
                for record_index, raw_record in enumerate(iter_records_from_path(dataset_file)):
                    summary.input_records += 1
                    cleaned_record, rejection_reason = clean_record(
                        raw_record,
                        dataset_id=dataset_id,
                        source_file=str(dataset_file.relative_to(local_dir)).replace("\\", "/"),
                        source_record_index=record_index,
                        source_license=summary.source_license,
                        source_dataset_url=summary.source_dataset_url,
                    )
                    if cleaned_record is None:
                        assert rejection_reason is not None
                        summary.rejected_records += 1
                        summary.rejection_reasons[rejection_reason] += 1
                        continue

                    payload = cleaned_record.model_dump(exclude_none=True)
                    write_jsonl_record(output_handle, payload)
                    summary.kept_records += 1
                    summary.kept_turns += cleaned_record.turn_count
                    summary.kept_characters += cleaned_record.conversation_characters
                    summary.language_hints[cleaned_record.language_hint] += 1
                    for flag in cleaned_record.quality_flags:
                        summary.quality_flags[flag] += 1

            summaries.append(summary)
            total_input_records += summary.input_records
            total_kept_records += summary.kept_records
            total_rejected_records += summary.rejected_records
            total_kept_turns += summary.kept_turns
            total_kept_characters += summary.kept_characters
            total_rejection_reasons.update(summary.rejection_reasons)
            total_language_hints.update(summary.language_hints)
            total_quality_flags.update(summary.quality_flags)

    if total_kept_records == 0:
        raise RuntimeError(f"Cleaning produced zero usable conversations from {manifest_path}")

    report_payload = {
        "report_version": CLEANING_REPORT_VERSION,
        "generated_at": utc_timestamp(),
        "source_manifest": str(manifest_path),
        "source_manifest_generated_at": manifest.get("generated_at"),
        "output_dataset": str(cleaned_output_path),
        "datasets": [summary.to_dict() for summary in summaries],
        "totals": {
            "input_records": total_input_records,
            "kept_records": total_kept_records,
            "rejected_records": total_rejected_records,
            "average_turns_kept": total_kept_turns / total_kept_records,
            "average_characters_kept": total_kept_characters / total_kept_records,
            "rejection_reasons": dict(total_rejection_reasons),
            "language_hints": dict(total_language_hints),
            "quality_flags": dict(total_quality_flags),
        },
    }
    write_json_file(cleaning_report_path, report_payload)
    return cleaned_output_path, cleaning_report_path
