from pydantic import BaseModel, Field
from enum import StrEnum
from typing import Callable, List, Literal, Dict

class Task(StrEnum):
    classification = 'classification'
    regression = 'regression'

class TrainingHistoryType(StrEnum):
    training_history = "training_history"
    fine_tuning_history = "fine_tuning_history"

class EvaluationMetric(BaseModel):
    name: str
    function: Callable

class TrainingPhaseType(StrEnum):
    training="training"
    fine_tuning="fine_tuning"

class TrainingParams(BaseModel):
    epochs: int = 10
    lr: float = 0.001
    batch_size: Literal["full"] | int = 64
    val_size: float = 0.25
    print_every: int = 1
    metrics: List[EvaluationMetric] = []
    phase: TrainingPhaseType = TrainingPhaseType.training


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