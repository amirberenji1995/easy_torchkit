import torch
from typing import Callable

from .utils import ContrastiveLoss
from .configurations import TrainingParams, Task, TrainingHistory, TrainingPhaseType
from .base_model import BaseModel
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary as torchinfo_summary

sns.set_theme()


class ClassificationModel(BaseModel):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        track_best_model=True,
        early_stopping=True,
        early_stopping_patience=500,
    ):
        super(ClassificationModel, self).__init__(
            task=Task.classification,
            device=device,
            track_best_model=track_best_model,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
        )

    def summary(self, input_size, **kwargs):
        return torchinfo_summary(self, input_size, **kwargs)

    def _run_one_epoch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        batch_size: int,
        training_step: Callable,
        output_layer: str | None,
    ) -> float:
        self.train()
        epoch_loss = 0.0

        if batch_size == "full":
            batch_size = x.size(0)

        num_batches = (x.size(0) + batch_size - 1) // batch_size

        all_logits = []
        for i in range(num_batches):
            xb = x[i * batch_size : (i + 1) * batch_size]
            yb = y[i * batch_size : (i + 1) * batch_size]

            loss, logits = training_step(
                model=self,
                xb=xb,
                yb=yb,
                optimizer=optimizer,
                loss_fn=loss_fn,
                output_layer=output_layer,
            )

            epoch_loss += loss * xb.size(0)
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)
        return epoch_loss / x.size(0), all_logits

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        params: TrainingParams,
    ):
        """Compute loss and metrics based on logits and y."""

        if isinstance(loss_fn, ContrastiveLoss):
            # embeddings are concatenated for contrastive loss
            z1, z2 = logits.chunk(2, dim=0)
            loss = loss_fn(z1, z2, y)
            preds = None
        else:
            # Flatten extra dimensions if needed
            if logits.ndim > 2:
                logits = logits.view(logits.size(0), -1)  # ensure [batch, num_classes]
            preds = logits.argmax(dim=1)
            loss = loss_fn(logits, y)

        metrics_dict = {"loss": loss.item()}
        if params.metrics and preds is not None:
            for metric in params.metrics:
                metrics_dict[metric.name] = metric.function(
                    y.cpu().numpy(), preds.cpu().numpy()
                )

        return metrics_dict

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
            x.cpu(),
            y.cpu(),
            test_size=params.val_size,
            stratify=y.cpu() if y.ndim == 1 else None,
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
            # Run one epoch using the provided training_step
            epoch_loss, epoch_logits = self._run_one_epoch(
                x_train,
                y_train,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batch_size=params.batch_size,
                output_layer=params.output_layer,
                training_step=params.training_step,
            )

            # Compute loss and metrics via a helper function
            train_metrics = self._compute_metrics(
                epoch_logits, y_train, loss_fn, params
            )

            val_logits = self(x_val, output_layer=params.output_layer)
            val_metrics = self._compute_metrics(val_logits, y_val, loss_fn, params)

            current_history.log_train(train_metrics)
            current_history.log_val(val_metrics)

            # --- Best model tracking ---
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

            # --- Print logging ---
            if epoch % params.print_every == 0:
                metrics_str = " | ".join(
                    [
                        f"Train {m.name}: {train_metrics[m.name]:.4f} | Val {m.name}: {val_metrics[m.name]:.4f}"
                        for m in params.metrics
                    ]
                    if params.metrics
                    else []
                )
                print(
                    f"[{params.phase.upper()} | Epoch {epoch}/{params.epochs}] "
                    f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}"
                    + (f" | {metrics_str}" if metrics_str else "")
                )

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        training_params: TrainingParams,
    ):
        self.to(self.device)
        x, y = x.to(self.device), y.to(self.device)

        optimizer = self._optimizer_creator(training_params)

        self._run_training_loop(
            x=x,
            y=y,
            optimizer=optimizer,
            loss_fn=training_params.loss_fn,
            params=training_params,
        )

    def visualize_training_history(self, index=-1):
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

    def fine_tune(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        training_params: TrainingParams,
        *,
        reset_best: bool = True,
    ):
        training_params = training_params.model_copy(
            update={"phase": TrainingPhaseType.fine_tuning}
        )

        if reset_best:
            self.best_state_dict = None
            self.best_epoch = None
            self.best_metrics = None
            self.best_val_loss = float("inf")

        self.fit(x, y, training_params)
