from pydantic import BaseModel
from enum import StrEnum
from typing import Callable, List, Literal

class Task(StrEnum):
    classification = 'classification'
    regression = 'regression'

class TrainingHistoryType(StrEnum):
    training_history = "training_history"
    fine_tuning_history = "fine_tuning_history"

class EvaluationMetric(BaseModel):
    name: str
    function: Callable
class TrainingParams(BaseModel):
    epochs: int = 10
    lr: float = 0.001
    batch_size: Literal["full"] | int = 64
    val_size: float = 0.25
    print_every: int = 1
    metrics: List[EvaluationMetric] = []


