import torch
from typing import Dict, Literal
from sklearn.metrics import accuracy_score
import seaborn as sns
from .base_model import BaseTaskModel
from .utils import ContrastiveLoss
from .configurations import Task, TrainingParams, EvaluationMetric, TrainingPhaseType

sns.set_theme()


class ClassificationModel(BaseTaskModel):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        track_best_model: bool = True,
        random_state: int | None = None,
    ):
        super().__init__(
            Task.classification,
            device=device,
            track_best_model=track_best_model,
            random_state=random_state,
        )

    def _run_evaluation_pass(
        self, x: torch.Tensor, output_layer: str | None = None
    ) -> torch.Tensor:
        if isinstance(x, (list, tuple)) or (x.ndim >= 3 and x.size(1) == 2):
            x1, x2 = x if isinstance(x, (list, tuple)) else (x[:, 0], x[:, 1])
            z1, z2 = (
                self(x1, output_layer=output_layer),
                self(x2, output_layer=output_layer),
            )

            diff = z1 - z2
            dist = torch.norm(diff.reshape(diff.size(0), -1), p=2, dim=1)
            return torch.stack([dist, 1.0 - dist], dim=1)

        return super()._run_evaluation_pass(x, output_layer=output_layer)

    def _compute_metrics(self, logits, y, loss_fn, metrics=None) -> Dict[str, float]:
        if logits.ndim > 2:
            logits = logits.view(logits.size(0), -1)

        if isinstance(loss_fn, ContrastiveLoss):
            loss_val = loss_fn.forward(
                logits[:, 0], torch.zeros_like(logits[:, 0]), y.float()
            )
        else:
            loss_val = loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        res = {"loss": loss_val.item()}

        if metrics:
            for m in metrics:
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

    def visualize_training_history(
        self,
        index=-1,
        title: str | None = None,
        show_or_export: Literal["show", "export", "both"] = "show",
        export_path: str | None = None,
    ):
        if not self.history:
            return
        h = self.history[index]

        h.visualize(title, show_or_export, export_path)
