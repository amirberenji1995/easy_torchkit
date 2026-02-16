from pydantic import BaseModel
from enum import StrEnum
from typing import Callable, Dict, Optional
import torch


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
