import torch
from typing import List, Callable, Dict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from .base_model import BaseTaskModel
from .utils import ContrastiveLoss
from .configurations import Task, TrainingParams, EvaluationMetric, TrainingPhaseType

sns.set_theme()


class ClassificationModel(BaseTaskModel):
    def __init__(
        self,
        device=torch.device("cpu"),
        track_best_model=True,
        early_stopping=True,
        early_stopping_patience=500,
    ):
        super().__init__(
            Task.classification,
            device,
            track_best_model,
            early_stopping,
            early_stopping_patience,
        )

    def _run_evaluation_pass(
        self, x: torch.Tensor, output_layer: str | None = None
    ) -> torch.Tensor:
        """
        Overrides the base hook to handle Siamese split-forward pass
        if the data contains pairs (Batch, 2, ...).
        """
        # Logic: If x is 4D (CNN) or 3D (Tabular/Linear) and the second dim is 2
        # we treat it as a pair.
        if x.ndim >= 2 and x.size(1) == 2:
            x1, x2 = x[:, 0], x[:, 1]

            z1 = self(x1, output_layer=output_layer)
            z2 = self(x2, output_layer=output_layer)

            # Flatten to ensure we return embeddings for distance calc
            return torch.cat([z1.view(z1.size(0), -1), z2.view(z2.size(0), -1)], dim=0)

        return super()._run_evaluation_pass(x, output_layer=output_layer)

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        metrics: List = None,
    ) -> Dict[str, float]:
        if isinstance(loss_fn, ContrastiveLoss):
            z1, z2 = logits.chunk(2, dim=0)
            loss_val = loss_fn(z1, z2, y.view(-1).float())
            preds = None
        else:
            if logits.ndim > 2:
                logits = logits.view(logits.size(0), -1)
            loss_val = loss_fn(logits, y)
            preds = torch.argmax(logits, dim=1)

        res = {"loss": loss_val.item()}
        if metrics:
            for m in metrics:
                if isinstance(m.function, torch.nn.Module):
                    res[m.name] = m.function(logits, y).item()
                elif preds is not None:
                    res[m.name] = m.function(y.cpu().numpy(), preds.cpu().numpy())
        return res

    def predict(self, x: torch.Tensor, output_layer: str = None) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            # Uses the standard forward pass (not pairs)
            logits = self(x.to(self.device), output_layer=output_layer)
            if logits.ndim > 2:
                logits = logits.view(logits.size(0), -1)
            return torch.argmax(logits, dim=1).cpu()

    def evaluate(self, x, y, metrics=None, output_layer=None):
        if metrics is None:
            metrics = [EvaluationMetric(name="Accuracy", function=accuracy_score)]

        self.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            # Use the hook to handle potential pairs in x
            outputs = self._run_evaluation_pass(x, output_layer=output_layer)

            # Default fallback for metric compute logic
            loss_fn = torch.nn.CrossEntropyLoss()
            for m in metrics:
                if isinstance(m.function, torch.nn.Module):
                    loss_fn = m.function
                    break

            return self._compute_metrics(outputs, y, loss_fn, metrics)

    def fine_tune(self, x, y, params: TrainingParams, reset_best=True):
        params = params.model_copy(update={"phase": TrainingPhaseType.fine_tuning})
        if reset_best:
            self.best_state_dict, self.best_val_loss = None, float("inf")
        self.fit(x, y, params)

    def visualize_training_history(self, index=-1):
        if not self.history:
            return
        h = self.history[index]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, m, title in zip(axes, ["loss", "accuracy"], ["Loss", "Accuracy"]):
            if m in h.train:
                ax.plot(h.train[m], label="Train")
                ax.plot(h.val[m], label="Val")
                ax.set_title(title)
                ax.legend()
                ax.grid(True)
        plt.tight_layout()
        plt.show()
