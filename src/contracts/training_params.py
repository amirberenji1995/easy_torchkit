from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Callable, Any
from .configurations import EvaluationMetric, TrainingPhaseType
from ..early_stopping import StoppingCriteria
from ..training_steps.training_step_protocol import TrainingStep
from ..training_steps.supervised_training_step import SupervisedTrainingStep
import torch


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
    training_step: Callable | TrainingStep = SupervisedTrainingStep()
    stopping_criteria: List[StoppingCriteria] | None = None

    class Config:
        arbitrary_types_allowed = True
