from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from redaesth.config import RedAesthConfig
from training.callbacks import TrainingTelemetry
from training.dataset import (
    DatasetLoadReport,
    DatasetSchemaError,
    TrainingExample,
    deterministic_holdout,
    load_calibration_datasets,
)
from training.evaluation import (
    build_calibration_metrics,
    calculate_perplexity,
    write_calibration_metrics,
    write_calibration_report,
)
from training.export import export_adapter_bundle
from training.model import build_lora_settings, infer_lora_target_modules
from training.train import build_parser, resolve_cli_config
from training.trainer import (
    build_trainer_settings,
    build_training_arguments,
    resolve_resume_checkpoint,
)
from training.utils import load_calibration_config, with_calibration_output_dir


class FakeTokenizer:
    """Offline tokenizer fake that exposes the production loader contract."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        del tokenize, add_generation_prompt
        return "".join(
            f"<|{message['role']}|>{message['content']}<|end|>" for message in messages
        )

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        del truncation, add_special_tokens
        length = min(max_length, max(1, len(text.split())))
        return {"input_ids": list(range(length)), "attention_mask": [1] * length}

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")


class FakeModel:
    def __init__(self, module_names: tuple[str, ...]) -> None:
        self.module_names = module_names
        self.generation_config = FakeGenerationConfig()

    def named_modules(self):
        yield "", self
        for name in self.module_names:
            yield f"layers.0.{name}", object()

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_model.safetensors").write_text("adapter\n", encoding="utf-8")


class FakeGenerationConfig:
    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "generation_config.json").write_text("{}\n", encoding="utf-8")


def make_config(root: Path, **overrides: object) -> RedAesthConfig:
    values: dict[str, object] = {
        "project_root": root,
        "base_model_id": "test/SmolLM2-1.7B-Instruct",
        "embedding_model_id": "test/embedding",
        "calibration_train_path": Path("train.jsonl"),
        "calibration_output_dir": Path("training-output"),
        "calibration_adapter_dir": Path("training-output/adapter"),
        "calibration_metrics_path": Path("training-output/calibration_metrics.json"),
        "calibration_checkpoint_metadata_path": Path("training-output/checkpoint_metadata.json"),
        "calibration_report_path": Path("CALIBRATION_REPORT.md"),
    }
    values.update(overrides)
    return RedAesthConfig(**values)


def build_record(record_id: str, tokenizer: FakeTokenizer, *, valid_text: bool = True) -> dict[str, object]:
    messages = [
        {"role": "user", "content": f"I need help with session {record_id}."},
        {"role": "assistant", "content": "We will use a manageable next step."},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {
        "record_id": record_id,
        "source_id": "redaesth/synthetic-coaching-production",
        "language": "mostly_ascii",
        "domain": "fitness-coaching-adjacent",
        "conversations": messages,
        "text": text if valid_text else "template drift",
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class TrainingDatasetTests(unittest.TestCase):
    def test_loader_validates_schema_tokenizes_and_uses_deterministic_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer = FakeTokenizer()
            records = [build_record(f"sample-{index}", tokenizer) for index in range(6)]
            write_jsonl(root / "train.jsonl", records)
            config = make_config(root, calibration_validation_holdout_ratio=0.25, seed=19)

            first = load_calibration_datasets(tokenizer=tokenizer, config=config)
            second = load_calibration_datasets(tokenizer=tokenizer, config=config)

            self.assertEqual(first.report.train_count, 4)
            self.assertEqual(first.report.validation_count, 2)
            self.assertEqual(first.report.validation_source, "deterministic_in_memory_holdout")
            self.assertEqual(first.train_dataset.rows, second.train_dataset.rows)
            self.assertEqual(first.validation_dataset.rows, second.validation_dataset.rows)
            self.assertEqual(first.train_dataset[0]["labels"], first.train_dataset[0]["input_ids"])

    def test_loader_fails_closed_for_template_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer = FakeTokenizer()
            write_jsonl(root / "train.jsonl", [build_record("bad", tokenizer, valid_text=False)])
            config = make_config(root)

            with self.assertRaises(DatasetSchemaError) as context:
                load_calibration_datasets(tokenizer=tokenizer, config=config)

            self.assertEqual(context.exception.report.malformed_records, ["train.jsonl:1"])

    def test_explicit_validation_file_is_loaded_without_repartitioning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer = FakeTokenizer()
            write_jsonl(root / "train.jsonl", [build_record("train", tokenizer)])
            write_jsonl(root / "val.jsonl", [build_record("validation", tokenizer)])
            config = make_config(root, calibration_validation_path=Path("val.jsonl"))

            datasets = load_calibration_datasets(tokenizer=tokenizer, config=config)

            self.assertEqual(datasets.report.validation_source, "provided")
            self.assertEqual(datasets.report.train_count, 1)
            self.assertEqual(datasets.report.validation_count, 1)

    def test_holdout_is_stable_and_preserves_all_examples(self) -> None:
        examples = [
            TrainingExample(record_id=f"id-{index}", conversations=[], text=str(index))
            for index in range(10)
        ]

        train_one, validation_one = deterministic_holdout(examples, ratio=0.2, seed=3)
        train_two, validation_two = deterministic_holdout(examples, ratio=0.2, seed=3)

        self.assertEqual([example.record_id for example in train_one], [example.record_id for example in train_two])
        self.assertEqual(
            [example.record_id for example in validation_one],
            [example.record_id for example in validation_two],
        )
        self.assertEqual(len(train_one) + len(validation_one), len(examples))


class TrainingConfigurationTests(unittest.TestCase):
    def test_lora_settings_support_architecture_projection_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(Path(temp_dir))
            model = FakeModel(
                ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
            )

            settings = build_lora_settings(model, config=config)

            self.assertEqual(settings.rank, config.lora_r)
            self.assertEqual(settings.alpha, config.lora_alpha)
            self.assertEqual(settings.target_modules[:4], ("q_proj", "k_proj", "v_proj", "o_proj"))

    def test_lora_settings_reject_missing_attention_projections(self) -> None:
        model = FakeModel(("q_proj", "k_proj", "v_proj"))

        with self.assertRaisesRegex(ValueError, "o_proj"):
            infer_lora_target_modules(model)

    def test_trainer_settings_and_resume_resolution_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = make_config(root)
            settings = build_trainer_settings(config)
            (settings.output_dir / "checkpoint-2").mkdir(parents=True)
            (settings.output_dir / "checkpoint-10").mkdir()
            (settings.output_dir / "checkpoint-not-a-step").mkdir()

            self.assertEqual(settings.num_train_epochs, 1)
            self.assertEqual(resolve_resume_checkpoint("latest", settings.output_dir).name, "checkpoint-10")
            self.assertEqual(
                resolve_resume_checkpoint(settings.output_dir / "checkpoint-2", settings.output_dir).name,
                "checkpoint-2",
            )

    def test_training_arguments_use_typed_qlora_configuration(self) -> None:
        class FakeTrainingArguments:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(Path(temp_dir))
            transformers_module = ModuleType("transformers")
            transformers_module.TrainingArguments = FakeTrainingArguments
            original_module = sys.modules.get("transformers")
            sys.modules["transformers"] = transformers_module
            try:
                arguments = build_training_arguments(config)
            finally:
                if original_module is None:
                    del sys.modules["transformers"]
                else:
                    sys.modules["transformers"] = original_module

        self.assertEqual(arguments.kwargs["optim"], "paged_adamw_8bit")
        self.assertEqual(arguments.kwargs["gradient_accumulation_steps"], 4)
        self.assertTrue(arguments.kwargs["gradient_checkpointing"])

    def test_config_rejects_invalid_mixed_precision_and_checkpoint_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "both fp16 and bf16"):
                make_config(root, calibration_bf16=True)
            with self.assertRaisesRegex(ValueError, "must match"):
                make_config(root, calibration_save_steps=20)


class TrainingCliAndArtifactsTests(unittest.TestCase):
    def test_cli_overrides_remain_within_typed_configuration(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--epochs",
                "2",
                "--batch-size",
                "3",
                "--learning-rate",
                "0.0001",
                "--output-dir",
                "custom-output",
                "--resume",
            ]
        )

        config = resolve_cli_config(args)

        self.assertEqual(config.num_train_epochs, 2)
        self.assertEqual(config.per_device_train_batch_size, 3)
        self.assertEqual(config.learning_rate, 0.0001)
        self.assertEqual(args.resume, "latest")
        self.assertEqual(config.calibration_output_dir.name, "custom-output")
        self.assertEqual(config.calibration_adapter_dir, config.calibration_output_dir / "adapter")

    def test_metrics_reports_and_adapter_export_write_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = make_config(root)
            report = DatasetLoadReport(
                train_path=config.calibration_train_path,
                validation_path=None,
                train_count=8,
                validation_count=2,
                validation_source="deterministic_in_memory_holdout",
            )
            telemetry = TrainingTelemetry(losses=[{"step": 1.0, "loss": 0.4}], completed=True)
            metrics = build_calibration_metrics(
                train_metrics={"train_runtime": 12.5, "train_samples_per_second": 0.64},
                evaluation_metrics={"eval_loss": 0.5},
                data_report=report,
                telemetry=telemetry,
                output_dir=config.calibration_output_dir,
                config=config,
            )

            metrics_path = write_calibration_metrics(metrics, config=config)
            report_path = write_calibration_report(metrics, config=config)
            exports = export_adapter_bundle(
                model=FakeModel(("q_proj", "k_proj", "v_proj", "o_proj")),
                tokenizer=FakeTokenizer(),
                metrics=metrics,
                config=config,
            )

            self.assertEqual(calculate_perplexity(0.5), metrics["perplexity"])
            self.assertTrue(metrics_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue((exports["adapter_dir"] / "adapter_model.safetensors").exists())
            metadata = json.loads(exports["checkpoint_metadata"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["gguf_conversion"]["status"], "not_performed")

    def test_config_file_and_output_override_preserve_config_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_config = make_config(root)
            override_path = root / "override.json"
            override_path.write_text('{"learning_rate": 0.0003}\n', encoding="utf-8")

            loaded = load_calibration_config(config_path=override_path, base_config=base_config)
            output_config = with_calibration_output_dir(loaded, root / "alternate-output")

            self.assertEqual(output_config.learning_rate, 0.0003)
            self.assertEqual(output_config.calibration_metrics_path.parent, root / "alternate-output")


if __name__ == "__main__":
    unittest.main()
