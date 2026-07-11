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
    num_train_epochs: int = 3
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
    synthetic_quality_threshold: float = 0.75
    generation_batch_size: int = 20
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

        total_ratio = self.final_train_ratio + self.final_validation_ratio + self.final_test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                "Final dataset split ratios must sum to 1.0, "
                f"received train={self.final_train_ratio}, "
                f"validation={self.final_validation_ratio}, test={self.final_test_ratio}"
            )

    def resolve_path(self, path: Path) -> Path:
        """Resolve a possibly relative path against the project root."""

        return path if path.is_absolute() else self.project_root / path


config = RedAesthConfig()
