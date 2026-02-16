from abc import ABC, abstractmethod
from typing import Callable, Tuple
import torch
from dataclasses import dataclass


@dataclass(slots=True)
class ModelOutput:
    loss_input: tuple
    preds: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    embeddings: torch.Tensor | None = None


class TrainingStep(ABC):
    @abstractmethod
    def train_batch(
        self,
        *,
        model: torch.nn.Module,
        xb: torch.Tensor,
        yb: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        output_layer: str | None = None,
    ) -> Tuple[torch.Tensor, ModelOutput]:
        pass

    @abstractmethod
    def eval_batch(
        self,
        *,
        model: torch.nn.Module,
        xb: torch.Tensor,
        yb: torch.Tensor | None = None,
        output_layer: str | None = None,
    ) -> ModelOutput:
        pass
