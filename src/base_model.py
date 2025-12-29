import torch
import copy
from abc import ABC, abstractmethod
from typing import List, TypeVar, Literal, Type, Dict, Any, Callable
from pathlib import Path
from .configurations import TrainingParams, Task, TrainingHistory, TrainingPhaseType

T = TypeVar("T", bound="BaseModel")


class BaseModel(torch.nn.Module, ABC):
    def __init__(
        self,
        task: Task,
        device: torch.device = torch.device("cpu"),
        track_best_model: bool = True,
        early_stopping: bool = True,
        early_stopping_patience: int = 50,
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

        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
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
        """Task-specific metric logic (Classification vs Regression)."""
        pass

    @abstractmethod
    def summary(self, input_size, **kwargs):
        pass

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

        # Data prep
        x_train, x_val, y_train, y_val = train_test_split(
            x.cpu(),
            y.cpu(),
            test_size=params.val_size,
            stratify=y.cpu()
            if (self.task == Task.classification and y.ndim == 1)
            else None,
        )
        x_train, y_train, x_val, y_val = (
            x_train.to(self.device),
            y_train.to(self.device),
            x_val.to(self.device),
            y_val.to(self.device),
        )

        history = TrainingHistory(params=params, phase=params.phase)
        history.initialize()
        self.history.append(history)
        patience_counter = 0

        for epoch in range(1, params.epochs + 1):
            # Train
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

            # Val
            self.eval()
            with torch.no_grad():
                val_logits = self(x_val, output_layer=params.output_layer)
                val_metrics = self._compute_metrics(
                    val_logits, y_val, loss_fn, params.metrics
                )

            history.log_train(train_metrics)
            history.log_val(val_metrics)

            # Checkpointing
            if self.track_best_model and val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss, self.best_epoch, self.best_metrics = (
                    val_metrics["loss"],
                    epoch,
                    val_metrics,
                )
                self.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if self.early_stopping and patience_counter >= self.early_stopping_patience:
                print(f"⏹ Early stopping at epoch {epoch}")
                break

            if epoch % params.print_every == 0:
                self._print_epoch_log(epoch, params, train_metrics, val_metrics)

    def _print_epoch_log(self, epoch, params, train_m, val_m):
        m_str = (
            " | ".join(
                [
                    f"Train {m.name}: {train_m[m.name]:.4f} | Val {m.name}: {val_m[m.name]:.4f}"
                    for m in params.metrics
                ]
            )
            if params.metrics
            else ""
        )
        print(
            f"[{params.phase.upper()} | {epoch}/{params.epochs}] Loss: T {train_m['loss']:.4f} / V {val_m['loss']:.4f} "
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

    # --- Layer Management (Simplified) ---
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

    def copy(self, reset_history=True, reset_optimizer=True):
        cp = copy.deepcopy(self)
        if reset_optimizer:
            cp.optimizer = None
        if reset_history:
            cp.history = []
        cp.best_state_dict, cp.best_epoch, cp.best_metrics, cp.best_val_loss = (
            None,
            None,
            None,
            float("inf"),
        )
        return cp

    def export(self, path: str | Path):
        checkpoint = {
            "state_dict": self.state_dict(),
            "init_params": self.init_params,
            "task": self.task,
            "best_state_dict": self.best_state_dict,
            "history": self.history,
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def import_(cls: Type[T], path: str | Path, device="cpu") -> T:
        ckpt = torch.load(Path(path), map_location=device)
        model = cls(**ckpt.get("init_params", {}))
        model.load_state_dict(ckpt["state_dict"])
        model.history = ckpt.get("history", [])
        return model.to(device)

    def recover_best_model(self):
        if self.best_state_dict:
            self.load_state_dict(self.best_state_dict)

    def _optimizer_creator(self, params: TrainingParams):
        return params.optimizer(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=params.lr,
            **(params.optimizer_params or {}),
        )
