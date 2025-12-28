import torch
from abc import ABC, abstractmethod
from typing import Callable, List, TypeVar, Literal, Type
from .configurations import TrainingParams, Task, TrainingHistory, TrainingHistoryType
import copy
from pathlib import Path

T = TypeVar("T", bound="BaseModel")

class BaseModel(torch.nn.Module, ABC):
    def __init__(
        self,
        task: Task = Task.classification,
        device: torch.device = torch.device("cpu"),
        track_best_model: bool = True,
        early_stopping: bool = True,
        early_stopping_patience: int = 50
    ):
        super().__init__()

        self.task = task
        self.device = device

        # ---- ARCHITECTURE CONTAINER (ABSTRACTED) ----
        self.network: torch.nn.Sequential | None = None

        # ---- TRAINING STATE ----
        self.history: List[TrainingHistory] =[]
        self.metrics: List | None = None

        # ---- BEST MODEL TRACKING ----
        self.track_best_model = track_best_model
        self.best_state_dict = None
        self.best_epoch = None
        self.best_metrics = None
        self.best_val_loss = float("inf")

        # ---- EARLY STOPPING ----
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience

        self.init_params: dict = {}

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

    
    @abstractmethod
    def summary(self, input_size, **kwargs):
        pass

    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor, training_params: TrainingParams):
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
    
    def freeze_layer(self, layer_name: str) -> None:
        if self.network is None:
            raise RuntimeError("Model has no network defined.")

        found = False
        for name, param in self.named_parameters():
            if name.startswith(f"network.{layer_name}."):
                param.requires_grad = False
                found = True

        if not found:
            raise ValueError(
                f"Layer '{layer_name}' not found. "
                f"Available layers: {list(self.network._modules.keys())}"
            )


    def freeze_layers(self, layer_names: List[str] | Literal["all"] = "all") -> None:
        if self.network is None:
            raise RuntimeError("Model has no network defined.")

        if layer_names == "all":
            for param in self.network.parameters():
                param.requires_grad = False
            return

        for layer_name in layer_names:
            self.freeze_layer(layer_name)


    def unfreeze_layer(self, layer_name: str) -> None:
        if self.network is None:
            raise RuntimeError("Model has no network defined.")

        found = False
        for name, param in self.named_parameters():
            if name.startswith(f"network.{layer_name}."):
                param.requires_grad = True
                found = True

        if not found:
            raise ValueError(
                f"Layer '{layer_name}' not found. "
                f"Available layers: {list(self.network._modules.keys())}"
            )


    def unfreeze_layers(self, layer_names: List[str] | Literal["all"] = "all") -> None:
        if self.network is None:
            raise RuntimeError("Model has no network defined.")

        if layer_names == "all":
            for param in self.network.parameters():
                param.requires_grad = True
            return

        for layer_name in layer_names:
            self.unfreeze_layer(layer_name)

    def freeze_up_to(self, layer_name: str) -> None:
        if self.network is None:
            raise RuntimeError("Model has no network defined.")

        layer_names = list(self.network._modules.keys())

        if layer_name not in layer_names:
            raise ValueError(
                f"Layer '{layer_name}' not found. "
                f"Available layers: {layer_names}"
            )

        # Freeze layers from start up to (and including) layer_name
        index = layer_names.index(layer_name)
        layers_to_freeze = layer_names[: index + 1]

        self.freeze_layers(layers_to_freeze)

    def copy(self, *, reset_history: bool = True, reset_optimizer: bool = True):

        model_copy = copy.deepcopy(self)

        # ---- Optimizer should NOT be copied ----
        if reset_optimizer:
            model_copy.optimizer = None

        # ---- History should usually be reset ----
        if reset_history:
            model_copy.history = []

        # ---- Best-model tracking reset ----
        model_copy.best_state_dict = None
        model_copy.best_epoch = None
        model_copy.best_metrics = None
        model_copy.best_val_loss = float("inf")

        return model_copy
    
    def export(self, path: str | Path) -> None:
        """
        Save the model and metadata to disk (PyTorch 2.6+ safe).
        """
        path = Path(path)

        checkpoint = {
            "class_name": self.__class__.__name__,
            "state_dict": self.state_dict(),
            "init_params": self.init_params,
            "task": self.task.name if isinstance(self.task, Task) else str(self.task),
            "best_state_dict": self.best_state_dict,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        torch.save(checkpoint, path)

    @classmethod
    def import_(
        cls: Type[T],
        path: str | Path,
        device: torch.device | str = "cpu",
    ) -> T:
        """
        Load a model from disk and restore its metadata.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)

        # ---- Rebuild model ----
        init_params = checkpoint.get("init_params", {})
        model = cls(**init_params)

        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        # ---- Restore metadata ----
        task_name = checkpoint.get("task")
        if task_name:
            model.task = Task[task_name] if isinstance(task_name, str) else task_name

        model.best_state_dict = checkpoint.get("best_state_dict")
        model.best_epoch = checkpoint.get("best_epoch")
        model.best_metrics = checkpoint.get("best_metrics")
        model.best_val_loss = checkpoint.get("best_val_loss")
        model.history = checkpoint.get("history", [])

        return model

    def recover_best_model(self) -> None:
        """
        Restore the best-performing model (by validation loss).
        """
        if self.best_state_dict is None:
            print("No best model stored.")
            return

        self.load_state_dict(self.best_state_dict)

        print("\n✔ Best model recovered")
        print(f"Epoch: {self.best_epoch}")
        if self.best_metrics:
            for k, v in self.best_metrics.items():
                print(f"{k}: {v:.4f}")

    def _optimizer_creator(self, training_params: TrainingParams) -> torch.optim.Optimizer:
        optimizer_kwargs = training_params.optimizer_params or {}

        return training_params.optimizer(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=training_params.lr,
            **optimizer_kwargs
        )