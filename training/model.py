"""Selected-model, QLoRA, and PEFT construction helpers for calibration training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from redaesth.config import RedAesthConfig, config as default_config


@dataclass(slots=True, frozen=True)
class LoraSettings:
    """Framework-neutral LoRA settings derived exclusively from typed configuration."""

    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    bias: str
    task_type: str
    gradient_checkpointing: bool


def load_selected_tokenizer(config: RedAesthConfig = default_config) -> Any:
    """Load the tokenizer selected by repository configuration and normalize padding."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _compute_dtype(config: RedAesthConfig) -> Any:
    """Resolve a hardware-supported 4-bit compute dtype for Kaggle accelerators."""

    import torch

    if config.calibration_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_quantized_model(config: RedAesthConfig = default_config) -> Any:
    """Load the configured causal model in 4-bit mode and prepare it for LoRA training."""

    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_compute_dtype(config),
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        quantization_config=quantization_config,
        device_map=config.calibration_device_map,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.calibration_gradient_checkpointing,
    )
    if config.calibration_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def infer_lora_target_modules(model: Any) -> tuple[str, ...]:
    """Infer compatible projection names from model modules instead of branching on model IDs."""

    supported = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    available = {
        module_name.rsplit(".", maxsplit=1)[-1]
        for module_name, _ in model.named_modules()
    }
    target_modules = tuple(name for name in supported if name in available)
    required_attention = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if not required_attention.issubset(target_modules):
        missing = ", ".join(sorted(required_attention.difference(target_modules)))
        raise ValueError(f"Selected model does not expose required attention projections: {missing}")
    return target_modules


def build_lora_settings(
    model: Any,
    *,
    config: RedAesthConfig = default_config,
) -> LoraSettings:
    """Create reusable LoRA settings for SmolLM, Qwen, or Gemma-style projection modules."""

    return LoraSettings(
        rank=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=infer_lora_target_modules(model),
        bias=config.lora_bias,
        task_type=config.lora_task_type,
        gradient_checkpointing=config.calibration_gradient_checkpointing,
    )


def create_peft_lora_config(settings: LoraSettings) -> Any:
    """Convert framework-neutral settings into a PEFT LoraConfig at training time."""

    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias=settings.bias,
        task_type=getattr(TaskType, settings.task_type),
    )


def apply_lora_adapter(model: Any, peft_lora_config: Any) -> Any:
    """Attach the configured PEFT adapter to a prepared quantized model."""

    from peft import get_peft_model

    return get_peft_model(model, peft_lora_config)
