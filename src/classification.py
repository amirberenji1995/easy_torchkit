import torch
from typing import List, Callable, Dict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary as torchinfo_summary
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
            task=Task.classification,
            device=device,
            track_best_model=track_best_model,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
        )

    def summary(self, input_size, **kwargs):
        return torchinfo_summary(self, input_size, **kwargs)

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        metrics: List = None,
    ) -> Dict[str, float]:
        if isinstance(loss_fn, ContrastiveLoss):
            z1, z2 = logits.chunk(2, dim=0)
            loss_val = loss_fn(z1, z2, y)
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
            logits = self(x.to(self.device), output_layer=output_layer)
            return torch.argmax(logits.view(logits.size(0), -1), dim=1).cpu()

    def evaluate(self, x, y, metrics=None, output_layer=None):
        if metrics is None:
            metrics = [
                EvaluationMetric(name="Loss", function=torch.nn.CrossEntropyLoss()),
                EvaluationMetric(name="Accuracy", function=accuracy_score),
            ]
        self.eval()
        with torch.no_grad():
            logits = self(x.to(self.device), output_layer=output_layer)
            # Use the first metric as primary loss if it's a module
            loss_fn = (
                metrics[0].function
                if isinstance(metrics[0].function, torch.nn.Module)
                else torch.nn.CrossEntropyLoss()
            )
            return self._compute_metrics(logits, y.to(self.device), loss_fn, metrics)

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
            ax.plot(h.train[m], label="Train")
            ax.plot(h.val[m], label="Val")
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
        plt.show()
