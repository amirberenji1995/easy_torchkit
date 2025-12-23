import torch
from typing import Callable, List, Iterable
from .configurations import TrainingParams, Task, TrainingHistory
from .base_model import BaseModel
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from torchinfo import summary as torchinfo_summary
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T", bound = "ClassificationModel")


class ClassificationModel(BaseModel):
    """
    A concrete base class for classification tasks.
    Subclasses must still implement:
        - forward()
    """

    def __init__(self, device: torch.device = torch.device("cpu"), track_best_model=True, early_stopping=True, early_stopping_patience=500):
        super(ClassificationModel, self).__init__(task=Task.classification, device=device)
        self.loss_fn: Callable | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.history: List[TrainingHistory] =[]
        self.metrics: List | None = None
        
        self.track_best_model = track_best_model
        self.best_state_dict = None
        self.best_epoch = None
        self.best_metrics = None
        self.best_val_loss = float("inf")

        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience

    # -------------------------------------------
    # REGISTER LOSS
    # -------------------------------------------

    def summary(self, input_size, **kwargs):
        return torchinfo_summary(self, input_size, **kwargs)

    def _train_one_epoch(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        *,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        batch_size: int,
    ):
        self.train()
        epoch_loss = 0.0

        if batch_size == "full":
            batch_size = x_train.size(0)

        num_batches = (x_train.size(0) + batch_size - 1) // batch_size

        for i in range(num_batches):
            xb = x_train[i * batch_size:(i + 1) * batch_size]
            yb = y_train[i * batch_size:(i + 1) * batch_size]

            optimizer.zero_grad()
            logits = self(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        return epoch_loss / x_train.size(0)

    def _run_training_loop(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        params: TrainingParams,
    ):
        from sklearn.model_selection import train_test_split

        self.metrics = params.metrics

        # Train / Val split
        x_train, x_val, y_train, y_val = train_test_split(
            x.cpu(), y.cpu(),
            test_size=params.val_size,
            stratify=y.cpu()
        )
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        x_val, y_val = x_val.to(self.device), y_val.to(self.device)

        # History
        history = TrainingHistory(params=params, phase=params.phase)
        history.initialize()
        self.history.append(history)
        current_history = history

        patience_counter = 0

        for epoch in range(1, params.epochs + 1):
            train_loss = self._train_one_epoch(
                x_train=x_train,
                y_train=y_train,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batch_size=params.batch_size,
            )

            train_metrics = self.evaluate(x_train, y_train)
            val_metrics = self.evaluate(x_val, y_val)

            current_history.log_train({
                "loss": train_loss,
                "accuracy": train_metrics["accuracy"],
                **{k: v for k, v in train_metrics.items() if k not in ["loss", "accuracy"]},
            })

            current_history.log_val({
                "loss": val_metrics["loss"],
                "accuracy": val_metrics["accuracy"],
                **{k: v for k, v in val_metrics.items() if k not in ["loss", "accuracy"]},
            })

            # Best model tracking
            if self.track_best_model and val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                }
                self.best_epoch = epoch
                self.best_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1

            if self.early_stopping and patience_counter >= self.early_stopping_patience:
                print(f"\n⏹ Early stopping at epoch {epoch}")
                break

            if epoch % params.print_every == 0:
                print(
                    f"[{params.phase.upper()} | Epoch {epoch}/{params.epochs}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )


    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        training_params: TrainingParams,
    ):
        self.to(self.device)
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)

        loss_fn = training_params.loss_fn()
        optimizer = training_params.optimizer(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=training_params.lr,
        )

        self._run_training_loop(
            x=x_train,
            y=y_train,
            optimizer=optimizer,
            loss_fn=loss_fn,
            params=training_params,
        )

    def visualize_training_history(self, index = -1):
        if self.history is None:
            print("Training history is not available. Call fit() first.")
            return

        history = self.history[index]
        train, val = history.train, history.val

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ---- Loss ----
        axes[0].plot(train["loss"], label="Train Loss")
        axes[0].plot(val["loss"], label="Val Loss")
        axes[0].set_title("Loss vs Epochs")
        axes[0].legend()
        axes[0].grid(True)

        # ---- Accuracy ----
        axes[1].plot(train["accuracy"], label="Train Accuracy")
        axes[1].plot(val["accuracy"], label="Val Accuracy")
        axes[1].set_title("Accuracy vs Epochs")
        axes[1].legend()
        axes[1].grid(True)

        fig.suptitle(
            f"{history.phase.upper()} "
            f"(epochs={history.params.epochs}, "
            f"lr={history.params.lr})"
        )

        plt.tight_layout()
        plt.show()

    # -------------------------------------------
    # PREDICT (classification logic)
    # -------------------------------------------
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for input x.
        """
        self.eval()

        x = x.to(self.device)

        with torch.no_grad():
            logits = self(x)
            preds = torch.argmax(logits, dim=1)

        return preds.cpu()

    # -------------------------------------------
    # EVALUATE (simple accuracy return)
    # -------------------------------------------
    from sklearn.metrics import accuracy_score
    import torch

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Evaluate the model on given data.

        Returns a dictionary with:
            - "loss": loss value using self.loss_fn
            - "accuracy": sklearn accuracy
            - other metrics from self.metrics by their name
        """
        if self.loss_fn is None:
            raise ValueError("Loss function is not registered. Call register_loss().")

        self.to(self.device)
        x, y = x.to(self.device), y.to(self.device)

        self.eval()
        with torch.no_grad():
            logits = self(x)
            # Compute loss
            loss_val = self.loss_fn(logits, y).item()

            # Compute accuracy
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true = y.cpu().numpy()
            acc = accuracy_score(y_true, preds)

            # Compute additional metrics
            metrics_dict = {}
            if hasattr(self, "metrics") and self.metrics:
                for metric in self.metrics:
                    metrics_dict[metric.name] = metric.function(y_true, preds)

        # Combine all results in a single dictionary
        results = {"loss": loss_val, "accuracy": acc}
        results.update(metrics_dict)

        return results
    
    def export(self, path: str | Path) -> None:
        """
        Save the model and metadata to disk in a safe way (PyTorch 2.6+ compatible).
        """
        path = Path(path)

        init_params: Dict[str, Any] = getattr(self, "init_params", {})

        checkpoint = {
            "class_name": self.__class__.__name__,
            "state_dict": self.state_dict(),
            "init_params": init_params,
            "task": self.task.name if isinstance(self.task, Task) else str(self.task),
            "best_state_dict": self.best_state_dict,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "best_val_loss": self.best_val_loss,
            "history": getattr(self, "history", None),
        }

        torch.save(checkpoint, path)

    # -------------------------------------------
    # IMPORT (LOAD) — PyTorch 2.6+ safe
    # -------------------------------------------
    @classmethod
    def import_(cls: Type[T], path: str | Path, device: torch.device | str = "cpu") -> T:
        """
        Load a model from disk, reconstructing the instance and metadata.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)

        # Reconstruct the model instance
        init_params: Dict[str, Any] = checkpoint.get("init_params", {})
        model = cls(**init_params)

        # Load weights
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        # Restore metadata
        task_name = checkpoint.get("task", "classification")
        model.task = Task[task_name] if hasattr(Task, task_name) else task_name

        model.best_state_dict = checkpoint.get("best_state_dict")
        model.best_epoch = checkpoint.get("best_epoch")
        model.best_metrics = checkpoint.get("best_metrics")
        model.best_val_loss = checkpoint.get("best_val_loss")
        model.history = checkpoint.get("history")

        return model
    
    def _set_trainable_layers(self, layers: Iterable[str]):
        for name, param in self.named_parameters():
            param.requires_grad = False

            if layers is None:
                param.requires_grad = True
            else:
                for layer_name in layers:
                    if name.startswith(layer_name):
                        param.requires_grad = True
                        break

    def recover_best_model(self) -> None:
        """
        Recover the best model parameters based on validation loss.
        """
        if self.best_state_dict is None:
            print("No best model stored. Did you enable track_best_model=True?")
            return
    
        self.load_state_dict(self.best_state_dict)
    
        print("\nBest model recovered.")
        print(f"Epoch: {self.best_epoch}")
        for k, v in self.best_metrics.items():
            print(f"{k}: {v:.4f}")