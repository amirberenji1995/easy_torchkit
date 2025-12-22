import torch
from typing import Callable, List
from configurations import TrainingParams, Task
from base_model import BaseModel
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
        self.history: dict | None = None
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
    def register_loss(self, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.CrossEntropyLoss) -> None:
        """
        Register the loss function for classification training.
        Default value is CrossEntropyLoss.
        """
        self.loss_fn = loss

    def register_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    # -------------------------------------------
    # FIT (simple, straightforward)
    # -------------------------------------------
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        training_params: TrainingParams,
    ):
        """
        Train the classification model with optional best-model tracking
        and early stopping based on validation loss.
        """

        if self.loss_fn is None:
            raise ValueError("Loss function is not registered. Call register_loss().")

        if not hasattr(self, "optimizer") or self.optimizer is None:
            raise ValueError("Optimizer is not registered. Call register_optimizer().")

        optimizer = self.optimizer

        self.metrics = training_params.metrics
        self.to(self.device)
        x, y = x.to(self.device), y.to(self.device)

        # -------------------------------
        # Random validation split (fixed)
        # -------------------------------
        N = x.shape[0]
        val_size = int(training_params.val_size * N)

        if val_size > 0:
            indices = torch.randperm(N)
            val_idx = indices[:val_size]
            train_idx = indices[val_size:]

            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]
        else:
            x_train, y_train = x, y
            x_val = y_val = None

        if training_params.batch_size == "full":
            batch_size = x_train.size(0)
            num_batches = 1
        else:
            batch_size = training_params.batch_size
            num_batches = (x_train.size(0) + batch_size - 1) // batch_size

        # -------------------------------
        # Initialize history
        # -------------------------------
        self.history = {
            "train": {"loss": [], "accuracy": []},

            "val": {"loss": [], "accuracy": []},
        }
        for metric in training_params.metrics:
            self.history["train"][metric.name] = []
            self.history["val"][metric.name] = []

        # -------------------------------
        # Early stopping state
        # -------------------------------
        epochs_without_improvement = 0

        # -------------------------------
        # Training loop
        # -------------------------------
        for epoch in range(1, training_params.epochs + 1):
            self.train()
            epoch_loss = 0.0

            for i in range(num_batches):
                xb = x_train[i * batch_size:(i + 1) * batch_size]
                yb = y_train[i * batch_size:(i + 1) * batch_size]

                optimizer.zero_grad()
                logits = self(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            # -------------------------------
            # Training metrics
            # -------------------------------
            train_loss = epoch_loss / x_train.size(0)
            self.history["train"]["loss"].append(train_loss)

            self.eval()
            with torch.no_grad():
                train_logits = self(x_train)
                train_preds = torch.argmax(train_logits, dim=1).cpu().numpy()
                train_true = y_train.cpu().numpy()

                train_acc = accuracy_score(train_true, train_preds)
                self.history["train"]["accuracy"].append(train_acc)

                for metric in training_params.metrics:
                    m_value = metric.function(train_true, train_preds)
                    self.history["train"][metric.name].append(m_value)

            # -------------------------------
            # Validation metrics
            # -------------------------------
            if x_val is not None:
                with torch.no_grad():
                    val_logits = self(x_val)
                    val_loss = self.loss_fn(val_logits, y_val).item()
                    self.history["val"]["loss"].append(val_loss)

                    val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
                    val_true = y_val.cpu().numpy()

                    val_acc = accuracy_score(val_true, val_preds)
                    self.history["val"]["accuracy"].append(val_acc)

                    for metric in training_params.metrics:
                        m_value = metric.function(val_true, val_preds)
                        self.history["val"][metric.name].append(m_value)

            # -------------------------------
            # Best model tracking & early stopping
            # -------------------------------
            if self.track_best_model and x_val is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    epochs_without_improvement = 0

                    self.best_metrics = {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                    }

                    self.best_state_dict = {
                        k: v.detach().clone() for k, v in self.state_dict().items()
                    }
                else:
                    epochs_without_improvement += 1

                    if self.early_stopping and epochs_without_improvement >= self.early_stopping_patience:
                        print(
                            f"\nEarly stopping triggered at epoch {epoch}. "
                            f"Best epoch was {self.best_epoch} "
                            f"(val_loss={self.best_val_loss:.4f})."
                        )
                        break

            # -------------------------------
            # Logging
            # -------------------------------
            if epoch % training_params.print_every == 0:
                train_metrics_str = " | ".join(
                    [f"{m.name}: {self.history['train'][m.name][-1]:.4f}" for m in training_params.metrics]
                )
                val_metrics_str = " | ".join(
                    [f"{m.name}: {self.history['val'][m.name][-1]:.4f}" for m in training_params.metrics]
                )

                log_msg = (
                    f"[Epoch {epoch}/{training_params.epochs}] "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | {train_metrics_str}"
                )

                if x_val is not None:
                    log_msg += (
                        f" || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | {val_metrics_str}"
                    )

                print(log_msg)



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



    def visualize_training_history(self):
        if not hasattr(self, "history") or not self.history:
            print("Training history is not available. Call fit() first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax = axes[0]
        ax.plot(self.history["train"]["loss"], label="Training Loss")
        ax.plot(self.history["val"]["loss"], label="Validation Loss")
        ax.set_title("Loss Vs. Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

        ax = axes[1]
        ax.plot(self.history["train"]["accuracy"], label="Training Accuracy")
        ax.plot(self.history["val"]["accuracy"], label="Validation Accuracy")
        ax.set_title("Accuracy Vs. Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True)

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

        device = torch.device(self.device.value)
        x = x.to(device)

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