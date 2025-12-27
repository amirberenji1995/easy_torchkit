import torch
from abc import ABC, abstractmethod
from typing import Callable, List
from .configurations import TrainingParams, Task, TrainingHistoryType


class BaseModel(torch.nn.Module, ABC):
    def __init__(
        self,
        task: Task = Task.classification,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.task = task
        self.device = device

        # ---- ARCHITECTURE CONTAINER (ABSTRACTED) ----
        self.network: torch.nn.Sequential | None = None

        # ---- TRAINING STATE ----
        self.loss_fn: Callable | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.training_history: dict | None = None
        self.metrics: List | None = None

        # ---- BEST MODEL TRACKING ----
        self.track_best_model = True
        self.best_state_dict = None
        self.best_epoch = None
        self.best_metrics = None
        self.best_val_loss = float("inf")

        # ---- EARLY STOPPING ----
        self.early_stopping = True
        self.early_stopping_patience = 50

    # ------------------------------------------------------------------
    # GENERIC FORWARD (LAYER-AWARE)
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        output_layer: str | None = None,
    ) -> torch.Tensor:
        """
        Generic forward pass through self.network.

        Args:
            x: input tensor
            output_layer: name of layer to stop at (None → last layer)

        Returns:
            torch.Tensor
        """
        if self.network is None:
            raise RuntimeError(
                "self.network is not defined. "
                "Subclasses must assign a torch.nn.Sequential to self.network."
            )

        for name, layer in self.network.named_children():
            x = layer(x)
            if output_layer is not None and name == output_layer:
                return x

        if output_layer is not None:
            raise ValueError(
                f"Layer '{output_layer}' not found. "
                f"Available layers: {list(self.network._modules.keys())}"
            )

        return x

    # ------------------------------------------------------------------
    # ABSTRACT API (TASK-SPECIFIC)
    # ------------------------------------------------------------------

    @abstractmethod
    def summary(self, input_size, **kwargs):
        pass

    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor, training_params: TrainingParams):
        pass

    @abstractmethod
    def recover_best_model(self) -> None:
        pass

    @abstractmethod
    def visualize_training_history(
        self,
        history: TrainingHistoryType = TrainingHistoryType.training_history,
    ):
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def evaluate(self, x: torch.Tensor, y: torch.Tensor):
        pass

    @abstractmethod
    def export(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def import_(cls, path: str):
        pass
