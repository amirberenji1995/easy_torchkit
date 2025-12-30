import torch
from torchinfo import summary as torchinfo_summary
import copy
from abc import ABC, abstractmethod
from typing import List, TypeVar, Literal, Type, Dict, Callable
from pathlib import Path
from .configurations import TrainingParams, Task, TrainingHistory
from .early_stopping import EarlyStoppingHandler, StoppingCriteria

T = TypeVar("T", bound="BaseTaskModel")


class BaseTaskModel(torch.nn.Module, ABC):
    def __init__(
        self,
        task: Task,
        device: torch.device = torch.device("cpu"),
        track_best_model: bool = True,
        stopping_criteria: List[StoppingCriteria] | None = None,
    ):
        super().__init__()
        self.task = task
        self.device = device
        self.network: torch.nn.Sequential | None = None

        self.history: List[TrainingHistory] = []
        self.track_best_model = track_best_model
        self.best_state_dict = None
        self.best_epoch = None
        self.best_metrics = None
        self.best_val_loss = float("inf")

        self.stopping_criteria = stopping_criteria
        self.init_params: dict = {}

    def forward(
        self, x: torch.Tensor, *, output_layer: str | None = None
    ) -> torch.Tensor:
        if self.network is None:
            raise RuntimeError("self.network is not defined.")

        for name, layer in self.network.named_children():
            x = layer(x)
            if output_layer is not None and name == output_layer:
                return x

        if output_layer is not None:
            raise ValueError(f"Layer '{output_layer}' not found.")
        return x

    @abstractmethod
    def _compute_metrics(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        metrics: List = None,
    ) -> Dict[str, float]:
        pass

    def summary(self, input_size, **kwargs):
        return torchinfo_summary(self, input_size, **kwargs)

    def _run_evaluation_pass(
        self, x: torch.Tensor, output_layer: str | None = None
    ) -> torch.Tensor:
        """
        Default evaluation pass hook.
        Subclasses override this for Siamese/Contrastive logic.
        """
        return self(x, output_layer=output_layer)

    def _run_one_epoch(
        self, x, y, *, optimizer, loss_fn, batch_size, training_step, output_layer
    ) -> tuple[float, torch.Tensor]:
        self.train()
        epoch_loss = 0.0
        if batch_size == "full":
            batch_size = x.size(0)
        num_batches = (x.size(0) + batch_size - 1) // batch_size

        all_outputs = []
        for i in range(num_batches):
            xb, yb = (
                x[i * batch_size : (i + 1) * batch_size],
                y[i * batch_size : (i + 1) * batch_size],
            )
            loss, logits = training_step(
                model=self,
                xb=xb,
                yb=yb,
                optimizer=optimizer,
                loss_fn=loss_fn,
                output_layer=output_layer,
            )
            epoch_loss += loss * xb.size(0)
            all_outputs.append(logits)

        return epoch_loss / x.size(0), torch.cat(all_outputs, dim=0)

    def _run_training_loop(self, x, y, *, optimizer, loss_fn, params: TrainingParams):
        from sklearn.model_selection import train_test_split

        # --- Data Preparation ---
        stratify = (
            y.cpu() if (self.task == Task.classification and y.ndim == 1) else None
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x.cpu(), y.cpu(), test_size=params.val_size, stratify=stratify
        )
        x_train, y_train, x_val, y_val = (
            x_train.to(self.device),
            y_train.to(self.device),
            x_val.to(self.device),
            y_val.to(self.device),
        )

        # --- Initialization ---
        history = TrainingHistory(params=params, phase=params.phase)
        history.initialize()
        self.history.append(history)

        # Initialize the generalized handler
        criteria_to_use = (
            self.stopping_criteria if self.stopping_criteria is not None else []
        )
        es_handler = EarlyStoppingHandler(criteria_to_use)

        for epoch in range(1, params.epochs + 1):
            # 1. Training Phase
            _, train_logits = self._run_one_epoch(
                x_train,
                y_train,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batch_size=params.batch_size,
                training_step=params.training_step,
                output_layer=params.output_layer,
            )
            train_metrics = self._compute_metrics(
                train_logits, y_train, loss_fn, params.metrics
            )

            # 2. Validation Phase
            self.eval()
            with torch.no_grad():
                val_outputs = self._run_evaluation_pass(
                    x_val, output_layer=params.output_layer
                )
                val_metrics = self._compute_metrics(
                    val_outputs, y_val, loss_fn, params.metrics
                )

            # 3. Logging
            history.log_train(train_metrics)
            history.log_val(val_metrics)

            # 4. Checkpoint Best Model (Always based on validation loss)
            if self.track_best_model and val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss, self.best_epoch, self.best_metrics = (
                    val_metrics["loss"],
                    epoch,
                    val_metrics,
                )
                self.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                }

            # 5. Generalized Early Stopping Check
            # Using the instance 'es_handler' initialized before the loop
            if es_handler.check(epoch, train_metrics, val_metrics):
                break

            if epoch % params.print_every == 0:
                self._print_epoch_log(epoch, params, train_metrics, val_metrics)

    def _print_epoch_log(self, epoch, params, train_m, val_m):
        m_str = (
            " | ".join(
                [
                    f"T-{m.name}: {train_m[m.name]:.4f} | V-{m.name}: {val_m[m.name]:.4f}"
                    for m in params.metrics
                ]
            )
            if params.metrics
            else ""
        )
        print(
            f"[{params.phase.upper()} | {epoch}/{params.epochs}] Loss: T-{train_m['loss']:.4f} / V-{val_m['loss']:.4f} "
            + m_str
        )

    def fit(self, x: torch.Tensor, y: torch.Tensor, training_params: TrainingParams):
        self.to(self.device)
        optimizer = self._optimizer_creator(training_params)
        self._run_training_loop(
            x=x.to(self.device),
            y=y.to(self.device),
            optimizer=optimizer,
            loss_fn=training_params.loss_fn,
            params=training_params,
        )

    def set_layers_grad(
        self, layer_names: List[str] | Literal["all"], requires_grad: bool
    ):
        if layer_names == "all":
            for p in self.network.parameters():
                p.requires_grad = requires_grad
        else:
            for name in layer_names:
                found = False
                for p_name, param in self.named_parameters():
                    if p_name.startswith(f"network.{name}."):
                        param.requires_grad = requires_grad
                        found = True
                if not found:
                    raise ValueError(f"Layer {name} not found.")

    def freeze_layers(self, names="all"):
        self.set_layers_grad(names, False)

    def unfreeze_layers(self, names="all"):
        self.set_layers_grad(names, True)

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

    def _optimizer_creator(self, params: TrainingParams):
        return params.optimizer(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=params.lr,
            **(params.optimizer_params or {}),
        )

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

    def copy(self, *, reset_history: bool = True, reset_optimizer: bool = True):
        model_copy = copy.deepcopy(self)

        if reset_optimizer:
            model_copy.optimizer = None

        if reset_history:
            model_copy.history = []

        model_copy.best_state_dict = None
        model_copy.best_epoch = None
        model_copy.best_metrics = None
        model_copy.best_val_loss = float("inf")

        return model_copy
