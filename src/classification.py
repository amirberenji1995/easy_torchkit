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

    def __init__(self, device: torch.device = torch.device("cpu"), track_best_model=True, early_stopping=True, early_stopping_patience=500):
        super(ClassificationModel, self).__init__(task=Task.classification, device=device, track_best_model=track_best_model, early_stopping=early_stopping, early_stopping_patience=early_stopping_patience)

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
        output_layer: str | None = None
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
            logits = self(xb, output_layer=output_layer)
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
                output_layer=params.output_layer
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

        if self.loss_fn is None:
            raise ValueError("Loss function must be set before calling fit().")

        optimizer_kwargs = training_params.optimizer_params or {}
        optimizer = training_params.optimizer(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=training_params.lr,
            **optimizer_kwargs
        )

        self._run_training_loop(
            x=x_train,
            y=y_train,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
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
    def predict(self, x: torch.Tensor, output_layer: str | None = None) -> torch.Tensor:
        """
        Predict class labels for input x.
        """
        self.eval()

        x = x.to(self.device)

        with torch.no_grad():
            logits = self(x, output_layer=output_layer)
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
    
