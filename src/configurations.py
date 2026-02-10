from pydantic import BaseModel, Field
from enum import StrEnum
from typing import Callable, List, Literal, Dict, Optional, Any
import torch
from .utils import supervised_step
import matplotlib.pyplot as plt
import seaborn as sns
from .early_stopping import StoppingCriteria

sns.set_theme()


class Task(StrEnum):
    classification = "classification"
    regression = "regression"


class TrainingHistoryType(StrEnum):
    training_history = "training_history"
    fine_tuning_history = "fine_tuning_history"


class EvaluationMetric(BaseModel):
    name: str
    function: Callable


class TrainingPhaseType(StrEnum):
    training = "training"
    fine_tuning = "fine_tuning"
    pre_training = "pre_training"


class TrainingParams(BaseModel):
    epochs: int | None = 10
    lr: float = 0.001
    batch_size: Literal["full"] | int = 64
    val_size: float = 0.25
    print_every: int = 1
    metrics: List[EvaluationMetric] = []
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        torch.nn.CrossEntropyLoss
    )
    optimizer: type[torch.optim.Optimizer] = torch.optim.Adam
    optimizer_params: Optional[Dict[str, Any]] = None
    phase: TrainingPhaseType = TrainingPhaseType.training
    output_layer: str | None = None
    training_step: Callable = supervised_step
    stopping_criteria: List[StoppingCriteria] | None = None


class TerminationReason(StrEnum):
    MAX_EPOCHS = "max_epochs"
    EARLY_STOPPING = "early_stopping"
    HARD_SAFETY_LIMIT = "hard_safety_limit"
    MANUAL_INTERRUPTION = "manual_interruption"


class TrainingTermination(BaseModel):
    epoch: int
    reason: TerminationReason
    details: Optional[str] = None
    final_val_metrics: Dict[str, float]
    best_model_recovered: bool


class TrainingHistory(BaseModel):
    params: TrainingParams
    phase: TrainingPhaseType

    train: Dict[str, List[float]] = Field(default_factory=dict)
    val: Dict[str, List[float]] = Field(default_factory=dict)

    epoch_times: List[float] = Field(default_factory=list)
    total_time: float = 0.0
    termination: TrainingTermination | None = None

    def initialize(self):
        """Initialize standard metric containers."""
        self.train = {"loss": [], "accuracy": []}
        self.val = {"loss": [], "accuracy": []}

        for metric in self.params.metrics:
            self.train[metric.name] = []
            self.val[metric.name] = []

    def log_train(self, values: Dict[str, float]):
        for k, v in values.items():
            self.train.setdefault(k, []).append(v)

    def log_val(self, values: Dict[str, float]):
        for k, v in values.items():
            self.val.setdefault(k, []).append(v)

    def log_epoch_time(self, duration: float):
        self.epoch_times.append(duration)

    def set_total_time(self, duration: float):
        self.total_time = duration

    @property
    def average_epoch_time(self) -> float:
        if not self.epoch_times:
            return None
        return sum(self.epoch_times) / len(self.epoch_times)

    @property
    def epochs(self) -> int:
        return len(self.train.get("loss", []))

    def visualize(
        self,
        title: str | None = None,
        show_or_export: Literal["show", "export", "both"] = "show",
        export_path: str | None = None,
    ):
        plot_metrics = ["loss"] + [m.name for m in self.params.metrics]
        num_plots = len(plot_metrics)

        fig, axes = plt.subplots(
            1, num_plots, figsize=(7.5 * num_plots, 5), squeeze=False
        )
        axes = axes.flatten()

        display_title = title if title else f"Training History - {self.phase.value}"

        fig.suptitle(display_title, fontsize=16)

        for ax, metric_name in zip(axes, plot_metrics):
            # Fix: Check for the exact name first, THEN try lowercase
            if metric_name in self.train:
                key = metric_name
            elif metric_name.lower() in self.train:
                key = metric_name.lower()
            else:
                ax.set_visible(False)
                continue

            if key in self.train:
                ax.plot(self.train[key], label="Train")
                ax.plot(self.val[key], label="Val")
                ax.set_title(metric_name.capitalize())
                ax.set_xlabel("Epochs")
                ax.legend()
                ax.grid(True)
            else:
                ax.set_visible(False)

        plt.tight_layout()

        if show_or_export in ["export", "both"]:
            if export_path:
                final_filename = export_path
            else:
                safe_title = "".join(
                    [c if c.isalnum() or c in (" ", "_") else "" for c in display_title]
                )
                final_filename = safe_title.replace(" ", "_")

            plt.savefig(final_filename)

        # --- SHOW LOGIC ---
        if show_or_export in ["show", "both"]:
            plt.show()
        else:
            plt.close(fig)
