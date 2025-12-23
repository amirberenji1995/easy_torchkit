import torch
from abc import ABC, abstractmethod
from typing import Callable, List
from src.configurations import TrainingParams, Task, TrainingHistoryType

class BaseModel(torch.nn.Module, ABC):
    def __init__(self, task: Task = Task.classification, device: torch.device = torch.device("cpu")):
        super(BaseModel, self).__init__()

        self.task = task
        self.device = device

        self.loss_fn: Callable | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.training_history: dict | None = None
        self.metrics: List | None = None

        self.track_best_model = True
        self.best_state_dict = None
        self.best_epoch = None
        self.best_metrics = None
        self.best_val_loss = float("inf")

        self.early_stopping = True
        self.early_stopping_patience = 50

    @abstractmethod
    def summary(self, input_size, **kwargs):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    # @abstractmethod
    # def register_loss(self, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
    #     pass

    # @abstractmethod
    # def register_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
    #     pass
    
    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor, training_params: TrainingParams):
        pass

    @abstractmethod
    def recover_best_model(self) -> None:
        pass

    @abstractmethod
    def visualize_training_history(self, history: TrainingHistoryType = TrainingHistoryType.training_history):
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        pass

    @abstractmethod
    def export(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def import_(cls, path: str):
        pass