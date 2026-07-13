"""Trainer callbacks for calibration loss, evaluation, checkpoint, and summary telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from redaesth.config import RedAesthConfig, config as default_config


@dataclass(slots=True)
class TrainingTelemetry:
    """Serializable runtime events captured during one calibration run."""

    losses: list[dict[str, float]] = field(default_factory=list)
    evaluations: list[dict[str, float]] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe telemetry payload for evaluation reporting."""

        return {
            "losses": self.losses,
            "evaluations": self.evaluations,
            "checkpoints": self.checkpoints,
            "completed": self.completed,
        }


def build_training_callbacks(
    telemetry: TrainingTelemetry,
    *,
    config: RedAesthConfig = default_config,
) -> list[Any]:
    """Create Hugging Face callbacks lazily so offline tests need no training dependencies."""

    from transformers import EarlyStoppingCallback, TrainerCallback

    class CalibrationTelemetryCallback(TrainerCallback):
        """Record trainer events in the calibration telemetry object."""

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> Any:
            del args
            if logs and "loss" in logs:
                telemetry.losses.append(
                    {"step": float(state.global_step), "loss": float(logs["loss"])}
                )
            return control

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, Any] | None = None,
            **_: Any,
        ) -> Any:
            del args
            if metrics:
                telemetry.evaluations.append(
                    {
                        "step": float(state.global_step),
                        "eval_loss": float(metrics.get("eval_loss", 0.0)),
                    }
                )
            return control

        def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            telemetry.checkpoints.append(str(args.output_dir) + f"/checkpoint-{state.global_step}")
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args, state
            telemetry.completed = True
            return control

    return [
        CalibrationTelemetryCallback(),
        EarlyStoppingCallback(early_stopping_patience=config.calibration_early_stopping_patience),
    ]
