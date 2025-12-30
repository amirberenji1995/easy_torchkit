from pydantic import BaseModel, Field
from enum import StrEnum
from typing import Callable, List, Literal, Dict, Optional, Any
import torch
from .utils import supervised_step
import matplotlib.pyplot as plt
import seaborn as sns

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
    epochs: int = 10
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


class TrainingHistory(BaseModel):
    params: TrainingParams
    phase: TrainingPhaseType

    train: Dict[str, List[float]] = Field(default_factory=dict)
    val: Dict[str, List[float]] = Field(default_factory=dict)

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

    @property
    def epochs(self) -> int:
        return len(self.train.get("loss", []))

    def visualize(self, title: str | None = None):
        plot_metrics = ["loss"] + [m.name for m in self.params.metrics]
        num_plots = len(plot_metrics)

        fig, axes = plt.subplots(
            1, num_plots, figsize=(7.5 * num_plots, 5), squeeze=False
        )
        axes = axes.flatten()

        if title:
            fig.suptitle(title, fontsize=16)

        for ax, title in zip(axes, plot_metrics):
            key = title.lower() if title.lower() in self.train else title

            if key in self.train:
                ax.plot(self.train[key], label="Train")
                ax.plot(self.val[key], label="Val")
                ax.set_title(title.capitalize())
                ax.set_xlabel("Epochs")
                ax.legend()
                ax.grid(True)
            else:
                ax.set_visible(False)

        plt.tight_layout()
        plt.show()
