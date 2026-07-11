"""Locked final dataset assembly for the first calibration training run."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import RedAesthConfig, config
from .dataset_pipeline import count_jsonl_records
from .final_audit import audit_final_dataset
from .final_split import (
    FinalSplitResult,
    build_final_manifest,
    split_final_dataset,
    spot_check_training_split,
)


FITNESS_TOPIC_TAGS = {
    "bodybuilding_prep",
    "gym_etiquette",
    "injury_pain",
    "nutrition",
    "recovery_sleep",
    "running_cardio",
    "strength_training",
}
FITNESS_SOURCE_IDS = {"ulysses531/fitness-conversation-dataset"}
MENTAL_HEALTH_SOURCE_IDS = {"hizardev/MentalHealth-Counseling"}
TokenizerLoader = Callable[[str], Any]


@dataclass(slots=True, frozen=True)
class FinalDatasetBuildResult:
    """Primary output paths emitted by final dataset assembly."""

    final_dataset_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    manifest_path: Path
    composition_audit_path: Path
    spot_check_report_path: Path
    dataset_card_path: Path
    training_readiness_report_path: Path


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


def write_text(path: Path, content: str) -> Path:
    """Write UTF-8 text to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def load_tokenizer(model_id: str) -> Any:
    """Load the selected base-model tokenizer lazily."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def deterministic_shuffle(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Return a deterministically shuffled copy of the records."""

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def load_scored_real_records(scored_dataset_path: Path) -> list[dict[str, Any]]:
    """Load the scored real-world records that passed the training filter."""

    records = read_jsonl(scored_dataset_path)
    return [record for record in records if record.get("passes_training_filter")]


def load_optional_jsonl_directory(directory: Path, *, source_type: str) -> list[dict[str, Any]]:
    """Load optional JSONL datasets from a directory tree without re-labeling them."""

    if not directory.exists():
        return []

    records: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*.jsonl")):
        for index, record in enumerate(read_jsonl(path)):
            if "conversations" not in record:
                raise ValueError(f"Optional final-dataset input is missing `conversations`: {path}")
            normalized = dict(record)
            normalized.setdefault("record_id", record.get("conversation_id") or f"{source_type}::{path.stem}::{index:06d}")
            normalized.setdefault("source_type", source_type)
            normalized.setdefault("dataset_id", record.get("dataset_id") or f"{source_type}/{path.stem}")
            records.append(normalized)
    return records


def split_real_record_buckets(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into fitness-primary, emotional-support-only, and general buckets."""

    fitness_primary: list[dict[str, Any]] = []
    emotional_support_only: list[dict[str, Any]] = []
    general_records: list[dict[str, Any]] = []

    for record in records:
        tags = set(record.get("topic_tags", []))
        if tags & FITNESS_TOPIC_TAGS or record.get("dataset_id") in FITNESS_SOURCE_IDS:
            fitness_primary.append(record)
            continue
        if "emotional_support" in tags:
            emotional_support_only.append(record)
            continue
        general_records.append(record)

    return fitness_primary, emotional_support_only, general_records


def cap_emotional_support_records(
    *,
    fitness_primary: list[dict[str, Any]],
    emotional_support_only: list[dict[str, Any]],
    max_share: float,
) -> list[dict[str, Any]]:
    """Cap emotional-support-only records so they do not dominate the final dataset."""

    if not emotional_support_only:
        return []
    if not fitness_primary:
        return emotional_support_only

    max_allowed = int(len(fitness_primary) * (max_share / max(1e-9, 1.0 - max_share)))
    max_allowed = max(0, min(max_allowed, len(emotional_support_only)))
    return emotional_support_only[:max_allowed]


def resolve_source_id(record: dict[str, Any], *, source_type: str) -> str:
    """Resolve the original source dataset identifier for audit and stratification."""

    dataset_id = record.get("dataset_id")
    if isinstance(dataset_id, str) and dataset_id.strip():
        return dataset_id
    existing_source_id = record.get("source_id")
    if isinstance(existing_source_id, str) and existing_source_id.strip():
        return existing_source_id
    return source_type


def resolve_domain(record: dict[str, Any], *, source_id: str) -> str:
    """Project existing labels into the coarse training-readiness domain buckets."""

    topic_tags = set(record.get("topic_tags", []))
    exclusion_reasons = set(record.get("exclusion_reasons", []))
    if "programming_offdomain" in topic_tags or any(
        reason.startswith("off_domain") for reason in exclusion_reasons
    ):
        return "off-domain"
    if source_id in MENTAL_HEALTH_SOURCE_IDS:
        return "mental-health-adjacent"
    if "emotional_support" in topic_tags and not (topic_tags & FITNESS_TOPIC_TAGS):
        return "mental-health-adjacent"
    return "fitness-coaching-adjacent"


def build_training_record(
    record: dict[str, Any],
    *,
    source_type: str,
    tokenizer: Any,
) -> dict[str, Any]:
    """Project a balanced record into the locked final training schema."""

    conversations = record["conversations"]
    text = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=False,
    )
    source_id = resolve_source_id(record, source_type=source_type)
    return {
        "record_id": record.get("conversation_id") or record.get("record_id"),
        "source_type": source_type,
        "source_id": source_id,
        "dataset_id": record.get("dataset_id"),
        "language": record.get("language") or record.get("language_hint") or "unknown",
        "domain": resolve_domain(record, source_id=source_id),
        "topic_tags": record.get("topic_tags", []),
        "overall_quality_score": record.get("overall_quality_score"),
        "normalized_sha256": record.get("normalized_sha256"),
        "quality_flags": record.get("quality_flags", []),
        "source_license": record.get("source_license"),
        "conversations": conversations,
        "text": text,
    }


def compose_balanced_records(
    *,
    config: RedAesthConfig,
    scored_dataset_path: Path,
) -> list[dict[str, Any]]:
    """Reuse the current balancing behavior to assemble the pre-split corpus."""

    real_records = load_scored_real_records(scored_dataset_path)
    if not real_records:
        raise RuntimeError("No scored real-world records passed the training filter.")

    fitness_primary, emotional_support_only, general_records = split_real_record_buckets(real_records)
    shuffled_fitness = deterministic_shuffle(fitness_primary, config.seed)
    shuffled_emotional = deterministic_shuffle(emotional_support_only, config.seed + 1)
    shuffled_general = deterministic_shuffle(general_records, config.seed + 2)
    selected_emotional = cap_emotional_support_records(
        fitness_primary=shuffled_fitness,
        emotional_support_only=shuffled_emotional,
        max_share=config.max_emotional_support_share,
    )

    balanced_real_records = shuffled_fitness + selected_emotional + shuffled_general
    for record in balanced_real_records:
        record["source_type"] = "real"

    synthetic_records = load_optional_jsonl_directory(
        config.synthetic_data_dir / "validated",
        source_type="synthetic",
    )
    augmented_records = load_optional_jsonl_directory(
        config.cleaned_data_dir / "augmented",
        source_type="augmented",
    )

    return deterministic_shuffle(
        balanced_real_records + synthetic_records + augmented_records,
        config.seed,
    )


def build_dataset_card(
    *,
    final_records: list[dict[str, Any]],
    split_result: FinalSplitResult,
) -> str:
    """Render a concise markdown card for the locked dataset artifact."""

    total_records = len(final_records)
    language_counts = Counter(record["language"] for record in final_records)
    domain_counts = Counter(record["domain"] for record in final_records)
    source_counts = Counter(record["source_id"] for record in final_records)

    lines = [
        "# RedAesth Final Dataset",
        "",
        f"- Total records: {total_records}",
        f"- Train: {split_result.train_count}",
        f"- Val: {split_result.validation_count}",
        f"- Test: {split_result.test_count}",
        "",
        "## Language distribution",
        "",
    ]
    for label, count in sorted(language_counts.items()):
        percentage = (count / total_records) * 100 if total_records else 0.0
        lines.append(f"- {label}: {count} ({percentage:.2f}%)")
    lines.extend(["", "## Domain distribution", ""])
    for label, count in sorted(domain_counts.items()):
        percentage = (count / total_records) * 100 if total_records else 0.0
        lines.append(f"- {label}: {count} ({percentage:.2f}%)")
    lines.extend(["", "## Source contribution", ""])
    for label, count in sorted(source_counts.items()):
        percentage = (count / total_records) * 100 if total_records else 0.0
        lines.append(f"- {label}: {count} ({percentage:.2f}%)")
    return "\n".join(lines) + "\n"


def format_distribution_lines(distribution: dict[str, dict[str, float | int]]) -> list[str]:
    """Render report lines for a distribution block."""

    lines: list[str] = []
    for label, payload in distribution.items():
        lines.append(
            f"- {label}: {int(payload['count'])} samples ({float(payload['percentage']):.2f}%)"
        )
    return lines


def readiness_blockers(
    *,
    audit_report: dict[str, Any],
    spot_check_report: dict[str, Any],
) -> list[str]:
    """Build a human-readable blocker list for the readiness report."""

    blockers: list[str] = []
    for metric_name, payload in audit_report["checks"].items():
        if payload["status"] != "FAIL":
            continue
        blockers.append(
            f"{metric_name}: {payload['actual_percentage']:.2f}% versus "
            f"{payload['comparison']} threshold {payload['threshold_percentage']:.2f}%"
        )
    if spot_check_report["status"] != "PASS":
        blockers.append(
            f"spot_check_malformed_records: {spot_check_report['malformed_record_count']} malformed "
            "sample(s) in the 50-row tokenizer validation"
        )
    return blockers


def recommended_base_model_rationale(model_id: str) -> str:
    """Return the locked-selection rationale for the readiness report."""

    if model_id != "HuggingFaceTB/SmolLM2-1.7B-Instruct":
        return (
            f"{model_id} is the currently selected base model in repository configuration and "
            "should remain the tokenizer and prompt-formatting anchor until a new formal model "
            "selection decision is recorded."
        )
    return (
        "The repository already contains a formal base-model decision selecting "
        "`HuggingFaceTB/SmolLM2-1.7B-Instruct`, so this readiness pass respects that lock. Among "
        "the sub-2B candidates already considered in the project, SmolLM2 offered the strongest "
        "instruction-following tradeoff at the chosen size, uses a straightforward chat template "
        "that is well suited to English conversational fine-tuning, and remains comfortably small "
        "enough for a 4-bit rank-16 LoRA calibration run on Kaggle T4 GPUs."
    )


def target_modules_for_model(model_id: str) -> list[str]:
    """Return the LoRA target modules for the selected architecture."""

    if model_id == "HuggingFaceTB/SmolLM2-1.7B-Instruct":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def estimate_kaggle_runtime_hours(
    *,
    train_sample_count: int,
    average_tokens: float,
) -> tuple[float, float]:
    """Estimate one-epoch wall-clock runtime on Kaggle T4 x2 for the calibration run."""

    total_tokens = train_sample_count * max(1.0, average_tokens)
    low_throughput_tokens_per_second = 600.0
    high_throughput_tokens_per_second = 900.0
    lower_bound_hours = total_tokens / high_throughput_tokens_per_second / 3600.0
    upper_bound_hours = total_tokens / low_throughput_tokens_per_second / 3600.0
    return round(lower_bound_hours, 1), round(upper_bound_hours, 1)


def build_next_command(config: RedAesthConfig) -> str:
    """Return the exact Kaggle notebook entrypoint for the calibration run."""

    target_modules = ", ".join(repr(module) for module in target_modules_for_model(config.base_model_id))
    return "\n".join(
        [
            "python -m pip install -q peft bitsandbytes trl",
            "python - <<'PY'",
            "from datasets import load_dataset",
            "from peft import LoraConfig",
            "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments",
            "from trl import SFTTrainer",
            "",
            f"model_id = {config.base_model_id!r}",
            f"train_path = {str(config.final_train_path)!r}",
            f"val_path = {str(config.final_validation_path)!r}",
            f"output_dir = {str(config.checkpoint_dir / 'calibration_lora_run')!r}",
            "",
            "tokenizer = AutoTokenizer.from_pretrained(model_id)",
            "if tokenizer.pad_token is None:",
            "    tokenizer.pad_token = tokenizer.eos_token",
            "bnb_config = BitsAndBytesConfig(",
            "    load_in_4bit=True,",
            "    bnb_4bit_quant_type='nf4',",
            "    bnb_4bit_use_double_quant=True,",
            "    bnb_4bit_compute_dtype='float16',",
            ")",
            "model = AutoModelForCausalLM.from_pretrained(",
            "    model_id,",
            "    quantization_config=bnb_config,",
            "    device_map='auto',",
            ")",
            "lora_config = LoraConfig(",
            "    r=16,",
            "    lora_alpha=32,",
            "    lora_dropout=0.05,",
            f"    target_modules=[{target_modules}],",
            "    task_type='CAUSAL_LM',",
            ")",
            "train_dataset = load_dataset('json', data_files=train_path, split='train')",
            "eval_dataset = load_dataset('json', data_files=val_path, split='train')",
            "training_args = TrainingArguments(",
            "    output_dir=output_dir,",
            "    per_device_train_batch_size=2,",
            "    per_device_eval_batch_size=2,",
            "    gradient_accumulation_steps=4,",
            "    learning_rate=2e-4,",
            "    warmup_ratio=0.05,",
            "    num_train_epochs=1,",
            "    logging_steps=25,",
            "    eval_strategy='steps',",
            "    eval_steps=100,",
            "    save_steps=100,",
            "    save_total_limit=2,",
            "    fp16=True,",
            "    report_to='none',",
            "    seed=42,",
            "    dataloader_num_workers=2,",
            "    gradient_checkpointing=True,",
            "    remove_unused_columns=False,",
            ")",
            "trainer = SFTTrainer(",
            "    model=model,",
            "    args=training_args,",
            "    train_dataset=train_dataset,",
            "    eval_dataset=eval_dataset,",
            "    peft_config=lora_config,",
            "    processing_class=tokenizer,",
            "    dataset_text_field='text',",
            f"    max_seq_length={config.max_seq_length},",
            ")",
            "trainer.train()",
            "trainer.evaluate()",
            "trainer.save_model()",
            "PY",
        ]
    )


def build_training_readiness_report(
    *,
    config: RedAesthConfig,
    audit_report: dict[str, Any],
    split_result: FinalSplitResult,
    manifest_path: Path,
    spot_check_report: dict[str, Any],
) -> Path:
    """Write the final readiness report required for Technical Director review."""

    blockers = readiness_blockers(audit_report=audit_report, spot_check_report=spot_check_report)
    go_decision = audit_report["overall_status"] == "PASS" and spot_check_report["status"] == "PASS"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_low, runtime_high = estimate_kaggle_runtime_hours(
        train_sample_count=split_result.train_count,
        average_tokens=float(spot_check_report["average_observed_tokens"]),
    )

    report_lines = [
        "## GO / NO GO Decision",
        "",
        "GO" if go_decision else "NO GO",
        "",
        "## Remaining Blockers",
        "",
    ]
    if blockers:
        report_lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        report_lines.append("None.")

    report_lines.extend(
        [
            "",
            "## Recommended Base Model",
            "",
            f"Selected model: `{config.base_model_id}`",
            "",
            recommended_base_model_rationale(config.base_model_id),
            "",
            "## LoRA Hyperparameters",
            "",
            "```yaml",
            f"lora_rank: {config.lora_r}",
            f"lora_alpha: {config.lora_alpha}",
            f"lora_dropout: {config.lora_dropout}",
            f"target_modules: {target_modules_for_model(config.base_model_id)}",
            f"learning_rate: {config.learning_rate}",
            f"warmup_ratio: {config.warmup_ratio}",
            "num_train_epochs: 1",
            f"per_device_train_batch_size: {config.per_device_train_batch_size}",
            f"gradient_accumulation_steps: {config.gradient_accumulation_steps}",
            "fp16: true",
            "bf16: false",
            f"max_seq_length: {config.max_seq_length}",
            "```",
            "",
            "## Estimated VRAM",
            "",
            "Estimated requirement: approximately 12 GB VRAM per GPU for a 4-bit QLoRA run with "
            "rank 16, alpha 32, sequence length 2048, and per-device batch size 2. Recommended "
            "accelerator: Kaggle T4 x2.",
            "",
            "## Estimated Kaggle Runtime",
            "",
            f"Estimated one-epoch wall-clock time on Kaggle T4 x2: approximately {runtime_low} to "
            f"{runtime_high} hours, based on {split_result.train_count} train samples and an average "
            f"observed tokenizer length of {spot_check_report['average_observed_tokens']} tokens in "
            "the 50-sample spot check.",
            "",
            "## Dataset Statistics",
            "",
            f"- Total samples assembled: {audit_report['total_samples']}",
            f"- Samples in train / val / test: {split_result.train_count} / "
            f"{split_result.validation_count} / {split_result.test_count}",
            "- Language labels use the existing pipeline outputs: `mostly_ascii` and "
            "`majority_non_ascii`.",
            "",
            "Language distribution:",
        ]
    )
    report_lines.extend(format_distribution_lines(audit_report["distributions"]["language"]))
    report_lines.extend(["", "Domain distribution:"])
    report_lines.extend(format_distribution_lines(audit_report["distributions"]["domain"]))
    report_lines.extend(["", "Source contribution:"])
    report_lines.extend(format_distribution_lines(audit_report["distributions"]["source"]))
    report_lines.extend(
        [
            f"- Truncation rate at max_seq_length: {spot_check_report['truncation_percentage']:.2f}%",
            "",
            "## Train / Validation / Test Split",
            "",
            f"- Train: `{split_result.train_path}` | SHA256 "
            f"`{manifest['splits']['train']['sha256']}` | {split_result.train_count} samples",
            f"- Val: `{split_result.validation_path}` | SHA256 "
            f"`{manifest['splits']['val']['sha256']}` | {split_result.validation_count} samples",
            f"- Test: `{split_result.test_path}` | SHA256 "
            f"`{manifest['splits']['test']['sha256']}` | {split_result.test_count} samples",
            "",
            "## Recommended Next Command",
            "",
            "```bash",
            build_next_command(config),
            "```",
            "",
            "## Expected Output Artifacts",
            "",
            f"- `{config.checkpoint_dir / 'calibration_lora_run' / 'adapter_model.safetensors'}`",
            f"- `{config.checkpoint_dir / 'calibration_lora_run' / 'adapter_config.json'}`",
            f"- `{config.checkpoint_dir / 'calibration_lora_run' / 'trainer_state.json'}`",
            f"- `{config.checkpoint_dir / 'calibration_lora_run' / 'training_args.bin'}`",
            f"- `{config.checkpoint_dir / 'calibration_lora_run' / 'train_results.json'}`",
            f"- `{config.checkpoint_dir / 'calibration_lora_run' / 'eval_results.json'}`",
            f"- `{config.checkpoint_dir / 'calibration_lora_run' / 'all_results.json'}`",
        ]
    )

    return write_text(config.training_readiness_report_path, "\n".join(report_lines) + "\n")


def build_final_dataset(
    *,
    config: RedAesthConfig = config,
    scored_dataset_path: Path | None = None,
    tokenizer: Any | None = None,
    tokenizer_loader: TokenizerLoader = load_tokenizer,
) -> FinalDatasetBuildResult:
    """Assemble, audit, split, hash, and report on the locked final dataset artifact."""

    source_path = config.resolve_path(scored_dataset_path or config.scored_dataset_path)
    tokenizer = tokenizer or tokenizer_loader(config.base_model_id)

    balanced_records = compose_balanced_records(config=config, scored_dataset_path=source_path)
    final_records = [
        build_training_record(
            record,
            source_type=str(record.get("source_type", "real")),
            tokenizer=tokenizer,
        )
        for record in balanced_records
    ]
    if count_jsonl_records(source_path) <= 0 or not final_records:
        raise RuntimeError("Final dataset assembly produced zero records.")

    final_dataset_path = write_jsonl(config.final_dataset_path, final_records)
    composition_audit_path, audit_report = audit_final_dataset(
        config=config,
        final_dataset_path=final_dataset_path,
    )
    split_result = split_final_dataset(config=config, final_dataset_path=final_dataset_path)
    spot_check_report_path, spot_check_report = spot_check_training_split(
        config=config,
        train_path=split_result.train_path,
        tokenizer=tokenizer,
    )
    manifest_path = build_final_manifest(
        config=config,
        final_dataset_path=final_dataset_path,
        split_result=split_result,
        composition_audit_path=composition_audit_path,
        spot_check_report_path=spot_check_report_path,
    )
    dataset_card_path = write_text(
        config.final_dataset_card_path,
        build_dataset_card(final_records=final_records, split_result=split_result),
    )
    training_readiness_report_path = build_training_readiness_report(
        config=config,
        audit_report=audit_report,
        split_result=split_result,
        manifest_path=manifest_path,
        spot_check_report=spot_check_report,
    )
    return FinalDatasetBuildResult(
        final_dataset_path=final_dataset_path,
        train_path=split_result.train_path,
        validation_path=split_result.validation_path,
        test_path=split_result.test_path,
        manifest_path=manifest_path,
        composition_audit_path=composition_audit_path,
        spot_check_report_path=spot_check_report_path,
        dataset_card_path=dataset_card_path,
        training_readiness_report_path=training_readiness_report_path,
    )
