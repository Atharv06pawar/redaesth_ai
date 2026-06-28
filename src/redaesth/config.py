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
    scored_data_dir: Path = Path("data/scored")
    synthetic_data_dir: Path = Path("data/synthetic")
    final_data_dir: Path = Path("data/final")
    retrieval_corpus_dir: Path = Path("data/retrieval_corpus")
    checkpoint_dir: Path = Path("training/checkpoints")
    evaluation_output_dir: Path = Path("evaluation/reports")
    approved_datasets_report: Path = Path("research/dataset_discovery/reports/approved_datasets.json")

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
    synthetic_quality_threshold: float = 0.75
    generation_batch_size: int = 20

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
        self.scored_data_dir = self.resolve_path(self.scored_data_dir)
        self.synthetic_data_dir = self.resolve_path(self.synthetic_data_dir)
        self.final_data_dir = self.resolve_path(self.final_data_dir)
        self.retrieval_corpus_dir = self.resolve_path(self.retrieval_corpus_dir)
        self.checkpoint_dir = self.resolve_path(self.checkpoint_dir)
        self.memory_db_path = self.resolve_path(self.memory_db_path)
        self.vector_db_path = self.resolve_path(self.vector_db_path)
        self.evaluation_output_dir = self.resolve_path(self.evaluation_output_dir)
        self.approved_datasets_report = self.resolve_path(self.approved_datasets_report)

    def resolve_path(self, path: Path) -> Path:
        """Resolve a possibly relative path against the project root."""

        return path if path.is_absolute() else self.project_root / path


config = RedAesthConfig()
