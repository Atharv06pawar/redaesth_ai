from __future__ import annotations

import re
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_MODEL_FILE = PROJECT_ROOT / "research" / "model_comparison" / "selected_model.txt"
DEFAULT_DECISION_LOG = PROJECT_ROOT / "DECISION_LOG.md"
DEFAULT_SELECTED_EMBEDDING_MODEL_FILE = (
    PROJECT_ROOT / "research" / "model_comparison" / "selected_embedding_model.txt"
)


def resolve_base_model_id(
    selected_model_path: Path | None = None,
    decision_log_path: Path | None = None,
) -> str:
    """Resolve the active base model from the selected-model file or decision log."""

    selected_model_path = selected_model_path or DEFAULT_SELECTED_MODEL_FILE
    decision_log_path = decision_log_path or DEFAULT_DECISION_LOG

    if selected_model_path.exists():
        value = selected_model_path.read_text(encoding="utf-8").strip()
        if value:
            return value

    if decision_log_path.exists():
        content = decision_log_path.read_text(encoding="utf-8")
        matches = re.findall(r"Select `([^`]+)` as the initial BASE_MODEL", content)
        if matches:
            return matches[-1]

    return "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def resolve_embedding_model_id(selected_embedding_model_path: Path | None = None) -> str:
    """Resolve the selected embedding model if it exists."""

    selected_embedding_model_path = (
        selected_embedding_model_path or DEFAULT_SELECTED_EMBEDDING_MODEL_FILE
    )
    if selected_embedding_model_path.exists():
        value = selected_embedding_model_path.read_text(encoding="utf-8").strip()
        if value:
            return value
    return "sentence-transformers/all-MiniLM-L6-v2"


class RedAesthConfig(BaseSettings):
    """Typed runtime configuration for Phase 2 systems."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="REDAESTH_",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT

    base_model_id: str = "read_from_file"
    selected_model_file: Path = Path("research/model_comparison/selected_model.txt")
    selected_embedding_model_file: Path = Path("research/model_comparison/selected_embedding_model.txt")
    embedding_model_id: str = "read_from_file"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    seed: int = 42

    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    raw_data_manifest_path: Path = Path("data/raw/raw_data_manifest.json")
    cleaned_data_dir: Path = Path("data/cleaned")
    cleaned_dataset_path: Path = Path("data/cleaned/cleaned_dataset.jsonl")
    cleaning_report_path: Path = Path("data/cleaned/cleaning_report.json")
    scored_data_dir: Path = Path("data/scored")
    scored_dataset_path: Path = Path("data/scored/scored_dataset.jsonl")
    scoring_report_path: Path = Path("data/scored/scoring_report.json")
    synthetic_data_dir: Path = Path("data/synthetic")
    final_data_dir: Path = Path("data/final")
    final_dataset_path: Path = Path("data/final/final_dataset.jsonl")
    final_train_path: Path = Path("data/final/train.jsonl")
    final_validation_path: Path = Path("data/final/val.jsonl")
    final_test_path: Path = Path("data/final/test.jsonl")
    final_dataset_card_path: Path = Path("data/final/dataset_card.md")
    final_manifest_path: Path = Path("data/final/final_dataset_manifest.json")
    final_composition_audit_path: Path = Path("data/final/composition_audit.json")
    final_spot_check_report_path: Path = Path("data/final/spot_check_report.json")
    retrieval_corpus_dir: Path = Path("data/retrieval_corpus")
    checkpoint_dir: Path = Path("training/checkpoints")
    evaluation_output_dir: Path = Path("evaluation/reports")
    approved_datasets_report: Path = Path("research/dataset_discovery/reports/approved_datasets.json")
    training_readiness_report_path: Path = Path("TRAINING_READINESS_REPORT.md")
    synthetic_dataset_specification_path: Path = Path("SYNTHETIC_DATASET_SPECIFICATION.md")
    synthetic_pilot_dataset_path: Path = Path(
        "data/synthetic/validated/synthetic_coaching_pilot.jsonl"
    )
    synthetic_generation_report_path: Path = Path("SYNTHETIC_GENERATION_REPORT.md")
    synthetic_production_dir: Path = Path("data/synthetic/production")
    synthetic_production_train_path: Path = Path(
        "data/synthetic/production/synthetic_train.jsonl"
    )
    synthetic_production_manifest_path: Path = Path(
        "data/synthetic/production/dataset_manifest.json"
    )
    synthetic_production_card_path: Path = Path("data/synthetic/production/dataset_card.md")
    synthetic_production_statistics_path: Path = Path(
        "data/synthetic/production/generation_statistics.json"
    )
    synthetic_factory_state_path: Path = Path("data/synthetic/production/factory_state.json")
    synthetic_factory_accepted_staging_path: Path = Path(
        "data/synthetic/production/accepted_staging.jsonl"
    )
    synthetic_factory_rejection_log_path: Path = Path(
        "data/synthetic/production/rejections.jsonl"
    )
    synthetic_production_report_path: Path = Path("PRODUCTION_CORPUS_REPORT.md")
    calibration_train_path: Path = Path("data/synthetic/production/synthetic_train.jsonl")
    calibration_validation_path: Path | None = None
    calibration_output_dir: Path = Path("training/outputs/calibration_lora_run")
    calibration_adapter_dir: Path = Path("training/outputs/calibration_lora_run/adapter")
    calibration_metrics_path: Path = Path(
        "training/outputs/calibration_lora_run/calibration_metrics.json"
    )
    calibration_checkpoint_metadata_path: Path = Path(
        "training/outputs/calibration_lora_run/checkpoint_metadata.json"
    )
    calibration_report_path: Path = Path("CALIBRATION_REPORT.md")

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    hf_token: str | None = None
    huggingface_hub_token: str | None = None

    memory_db_path: Path = Field(
        default=Path("data/memory.db"),
        validation_alias=AliasChoices("MEMORY_DB_PATH", "DB_PATH"),
    )
    memory_token_budget: int = 800

    retrieval_token_budget: int = 600
    retrieval_top_k: int = 3
    vector_db_path: Path = Field(
        default=Path("data/vector_store"),
        validation_alias=AliasChoices("VECTOR_DB_PATH", "VECTOR_STORE_PATH"),
    )

    synthetic_target_count: int = 8000
    synthetic_pilot_target_count: int = 100
    synthetic_generation_seed: int = 20260710
    synthetic_production_target_count: int = 250
    synthetic_factory_batch_size: int = 50
    synthetic_factory_attempt_multiplier: int = 20
    synthetic_min_validator_pass_rate: float = 1.0
    synthetic_max_duplicate_rate: float = 0.0
    synthetic_max_distribution_deviation: float = 0.01
    synthetic_max_memory_category_share: float = 0.40
    synthetic_max_conversation_length_share: float = 0.50
    synthetic_near_duplicate_similarity: float = 0.92
    synthetic_max_repeated_opening_share: float = 0.08
    synthetic_quality_threshold: float = 0.75
    generation_batch_size: int = 20
    calibration_validation_holdout_ratio: float = 0.10
    calibration_logging_steps: int = 10
    calibration_eval_steps: int = 10
    calibration_save_steps: int = 10
    calibration_save_total_limit: int = 2
    calibration_early_stopping_patience: int = 3
    calibration_dataloader_num_workers: int = 2
    calibration_fp16: bool = True
    calibration_bf16: bool = False
    calibration_gradient_checkpointing: bool = True
    calibration_device_map: str = "auto"
    calibration_optimizer: str = "paged_adamw_8bit"
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    synthetic_min_empathy_score: float = 0.70
    synthetic_min_coaching_quality_score: float = 0.70
    synthetic_min_personalization_score: float = 0.70
    synthetic_min_behavioral_adaptation_score: float = 0.65
    synthetic_min_scientific_consistency_score: float = 0.75
    synthetic_min_memory_usage_score: float = 0.70
    synthetic_min_follow_up_questioning_score: float = 0.60
    synthetic_min_hallucination_safety_score: float = 0.85
    synthetic_min_repetition_score: float = 0.75
    synthetic_min_scenario_consistency_score: float = 0.75
    minimum_training_quality_score: float = 0.55
    minimum_domain_relevance_score: float = 0.50
    final_train_ratio: float = 0.90
    final_validation_ratio: float = 0.05
    final_test_ratio: float = 0.05
    final_min_train_examples: int = 10_000
    final_min_validation_examples: int = 500
    final_min_test_examples: int = 500
    max_emotional_support_share: float = 0.35
    minimum_mostly_ascii_share: float = 0.60
    maximum_majority_non_ascii_share: float = 0.40
    maximum_mental_health_adjacent_share: float = 0.35
    maximum_off_domain_share: float = 0.01
    maximum_single_source_share: float = 0.70
    maximum_exact_duplicate_rate: float = 0.0
    final_spot_check_sample_size: int = 50
    final_spot_check_truncation_warning_threshold: float = 0.05

    def model_post_init(self, __context: object) -> None:
        if self.base_model_id == "read_from_file":
            self.base_model_id = resolve_base_model_id(
                self.resolve_path(self.selected_model_file),
                self.project_root / "DECISION_LOG.md",
            )

        if self.embedding_model_id == "read_from_file":
            self.embedding_model_id = resolve_embedding_model_id(
                self.resolve_path(self.selected_embedding_model_file)
            )

        self.selected_model_file = self.resolve_path(self.selected_model_file)
        self.selected_embedding_model_file = self.resolve_path(self.selected_embedding_model_file)
        self.data_dir = self.resolve_path(self.data_dir)
        self.raw_data_dir = self.resolve_path(self.raw_data_dir)
        self.raw_data_manifest_path = self.resolve_path(self.raw_data_manifest_path)
        self.cleaned_data_dir = self.resolve_path(self.cleaned_data_dir)
        self.cleaned_dataset_path = self.resolve_path(self.cleaned_dataset_path)
        self.cleaning_report_path = self.resolve_path(self.cleaning_report_path)
        self.scored_data_dir = self.resolve_path(self.scored_data_dir)
        self.scored_dataset_path = self.resolve_path(self.scored_dataset_path)
        self.scoring_report_path = self.resolve_path(self.scoring_report_path)
        self.synthetic_data_dir = self.resolve_path(self.synthetic_data_dir)
        self.final_data_dir = self.resolve_path(self.final_data_dir)
        self.final_dataset_path = self.resolve_path(self.final_dataset_path)
        self.final_train_path = self.resolve_path(self.final_train_path)
        self.final_validation_path = self.resolve_path(self.final_validation_path)
        self.final_test_path = self.resolve_path(self.final_test_path)
        self.final_dataset_card_path = self.resolve_path(self.final_dataset_card_path)
        self.final_manifest_path = self.resolve_path(self.final_manifest_path)
        self.final_composition_audit_path = self.resolve_path(self.final_composition_audit_path)
        self.final_spot_check_report_path = self.resolve_path(self.final_spot_check_report_path)
        self.retrieval_corpus_dir = self.resolve_path(self.retrieval_corpus_dir)
        self.checkpoint_dir = self.resolve_path(self.checkpoint_dir)
        self.memory_db_path = self.resolve_path(self.memory_db_path)
        self.vector_db_path = self.resolve_path(self.vector_db_path)
        self.evaluation_output_dir = self.resolve_path(self.evaluation_output_dir)
        self.approved_datasets_report = self.resolve_path(self.approved_datasets_report)
        self.training_readiness_report_path = self.resolve_path(self.training_readiness_report_path)
        self.synthetic_dataset_specification_path = self.resolve_path(self.synthetic_dataset_specification_path)
        self.synthetic_pilot_dataset_path = self.resolve_path(self.synthetic_pilot_dataset_path)
        self.synthetic_generation_report_path = self.resolve_path(self.synthetic_generation_report_path)
        self.synthetic_production_dir = self.resolve_path(self.synthetic_production_dir)
        self.synthetic_production_train_path = self.resolve_path(self.synthetic_production_train_path)
        self.synthetic_production_manifest_path = self.resolve_path(
            self.synthetic_production_manifest_path
        )
        self.synthetic_production_card_path = self.resolve_path(self.synthetic_production_card_path)
        self.synthetic_production_statistics_path = self.resolve_path(
            self.synthetic_production_statistics_path
        )
        self.synthetic_factory_state_path = self.resolve_path(self.synthetic_factory_state_path)
        self.synthetic_factory_accepted_staging_path = self.resolve_path(
            self.synthetic_factory_accepted_staging_path
        )
        self.synthetic_factory_rejection_log_path = self.resolve_path(
            self.synthetic_factory_rejection_log_path
        )
        self.synthetic_production_report_path = self.resolve_path(
            self.synthetic_production_report_path
        )
        self.calibration_train_path = self.resolve_path(self.calibration_train_path)
        if self.calibration_validation_path is not None:
            self.calibration_validation_path = self.resolve_path(self.calibration_validation_path)
        self.calibration_output_dir = self.resolve_path(self.calibration_output_dir)
        self.calibration_adapter_dir = self.resolve_path(self.calibration_adapter_dir)
        self.calibration_metrics_path = self.resolve_path(self.calibration_metrics_path)
        self.calibration_checkpoint_metadata_path = self.resolve_path(
            self.calibration_checkpoint_metadata_path
        )
        self.calibration_report_path = self.resolve_path(self.calibration_report_path)

        total_ratio = self.final_train_ratio + self.final_validation_ratio + self.final_test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                "Final dataset split ratios must sum to 1.0, "
                f"received train={self.final_train_ratio}, "
                f"validation={self.final_validation_ratio}, test={self.final_test_ratio}"
            )
        if not 0.0 < self.calibration_validation_holdout_ratio < 1.0:
            raise ValueError("Calibration validation holdout ratio must be between zero and one")
        if self.calibration_fp16 and self.calibration_bf16:
            raise ValueError("Calibration cannot enable both fp16 and bf16")
        if self.calibration_logging_steps <= 0:
            raise ValueError("Calibration logging steps must be positive")
        if self.calibration_eval_steps <= 0 or self.calibration_save_steps <= 0:
            raise ValueError("Calibration evaluation and save steps must be positive")
        if self.calibration_eval_steps != self.calibration_save_steps:
            raise ValueError(
                "Calibration evaluation and save steps must match when loading the best model"
            )
        if self.calibration_save_total_limit < 1:
            raise ValueError("Calibration save total limit must be at least one")
        if self.calibration_early_stopping_patience < 1:
            raise ValueError("Calibration early stopping patience must be at least one")

    def resolve_path(self, path: Path) -> Path:
        """Resolve a possibly relative path against the project root."""

        return path if path.is_absolute() else self.project_root / path


config = RedAesthConfig()
