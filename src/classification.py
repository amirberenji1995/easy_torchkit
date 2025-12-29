import torch
from typing import Callable, List, Dict, Any
from .utils import ContrastiveLoss
from .configurations import (
    TrainingParams,
    Task,
    TrainingHistory,
    TrainingPhaseType,
    EvaluationMetric,
)
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

    def _get_predictions_and_logits(
        self, x: torch.Tensor, output_layer: str | None = None
    ):
        """Internal helper to get raw logits and class predictions."""
        logits = self(x, output_layer=output_layer)

        # Ensure [batch, num_classes] for standard classification
        if logits.ndim > 2:
            logits = logits.view(logits.size(0), -1)

        preds = torch.argmax(logits, dim=1)
        return logits, preds

    def predict(self, x: torch.Tensor, output_layer: str | None = None) -> torch.Tensor:
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            _, preds = self._get_predictions_and_logits(x, output_layer)
        return preds.cpu()

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        metrics: List[Any] = None,
    ) -> Dict[str, float]:
        """Centralized metric computation logic used by training and evaluation."""

        # Handle Contrastive Loss vs Standard Classification
        if isinstance(loss_fn, ContrastiveLoss):
            z1, z2 = logits.chunk(2, dim=0)
            loss_val = loss_fn(z1, z2, y)
            preds = None
        else:
            # Flatten if coming from a raw forward pass not yet processed
            if logits.ndim > 2:
                logits = logits.view(logits.size(0), -1)
            loss_val = loss_fn(logits, y)
            preds = torch.argmax(logits, dim=1)

        metrics_dict = {"loss": loss_val.item()}

        if metrics:
            for m in metrics:
                # Support both TrainingParams metrics and EvaluationMetric objects
                name = m.name
                fn = m.function

                if isinstance(fn, torch.nn.Module):
                    # Torch loss functions take (logits, targets)
                    metrics_dict[name] = fn(logits, y).item()
                elif preds is not None:
                    # Scikit-learn style metrics take (y_true, y_pred)
                    metrics_dict[name] = fn(y.cpu().numpy(), preds.cpu().numpy())

        return metrics_dict

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
    ) -> tuple[float, torch.Tensor]:
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

        return epoch_loss / x.size(0), torch.cat(all_logits, dim=0)

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

        # Split and move to device
        x_train, x_val, y_train, y_val = train_test_split(
            x.cpu(),
            y.cpu(),
            test_size=params.val_size,
            stratify=y.cpu() if y.ndim == 1 else None,
        )
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        x_val, y_val = x_val.to(self.device), y_val.to(self.device)

        history = TrainingHistory(params=params, phase=params.phase)
        history.initialize()
        self.history.append(history)

        patience_counter = 0

        for epoch in range(1, params.epochs + 1):
            # Training phase
            _, train_logits = self._run_one_epoch(
                x_train,
                y_train,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batch_size=params.batch_size,
                output_layer=params.output_layer,
                training_step=params.training_step,
            )
            train_metrics = self._compute_metrics(
                train_logits, y_train, loss_fn, params.metrics
            )

            # Validation phase
            self.eval()
            with torch.no_grad():
                val_logits = self(x_val, output_layer=params.output_layer)
                val_metrics = self._compute_metrics(
                    val_logits, y_val, loss_fn, params.metrics
                )

            history.log_train(train_metrics)
            history.log_val(val_metrics)

            # Best model tracking & Early stopping
            if self.track_best_model and val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                }
                self.best_epoch, self.best_metrics = epoch, val_metrics
                patience_counter = 0
            else:
                patience_counter += 1

            if self.early_stopping and patience_counter >= self.early_stopping_patience:
                print(f"\n⏹ Early stopping at epoch {epoch}")
                break

            if epoch % params.print_every == 0:
                self._print_logs(epoch, params, train_metrics, val_metrics)

    def _print_logs(self, epoch, params, train_metrics, val_metrics):
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

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        metrics: list[EvaluationMetric] = None,
        output_layer: str | None = None,
    ) -> dict:
        if metrics is None:
            metrics = [
                EvaluationMetric(
                    name="CrossEntropyLoss", function=torch.nn.CrossEntropyLoss()
                ),
                EvaluationMetric(name="Accuracy", function=accuracy_score),
            ]

        self.eval()
        self.to(self.device)
        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            logits = self(x, output_layer=output_layer)
            # Use the shared computation logic
            # Note: We use the first metric's function as the primary 'loss_fn' for _compute_metrics
            primary_loss_fn = (
                metrics[0].function
                if isinstance(metrics[0].function, torch.nn.Module)
                else torch.nn.CrossEntropyLoss()
            )
            return self._compute_metrics(logits, y, primary_loss_fn, metrics)

    def fine_tune(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        training_params: TrainingParams,
        reset_best: bool = True,
    ):
        training_params = training_params.model_copy(
            update={"phase": TrainingPhaseType.fine_tuning}
        )
        if reset_best:
            self.best_state_dict, self.best_epoch, self.best_metrics = None, None, None
            self.best_val_loss = float("inf")
        self.fit(x, y, training_params)

    def visualize_training_history(self, index=-1):
        if not self.history:
            print("No history found.")
            return
        h = self.history[index]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, m, title in zip(
            axes, ["loss", "accuracy"], ["Loss vs Epochs", "Accuracy vs Epochs"]
        ):
            ax.plot(h.train[m], label="Train")
            ax.plot(h.val[m], label="Val")
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.show()
