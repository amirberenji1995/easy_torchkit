import torch
from typing import Literal
from sklearn.metrics import accuracy_score
import seaborn as sns
from .base_model import BaseTaskModel
from .contracts.configurations import (
    Task,
    EvaluationMetric,
    TrainingPhaseType,
)
from .contracts.training_params import TrainingParams
from .training_steps.training_step_protocol import TrainingStep
from .training_steps.supervised_training_step import SupervisedTrainingStep

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

    def predict(self, x: torch.Tensor, output_layer: str = None) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            # Uses the standard forward pass (not pairs)
            logits = self(x.to(self.device), output_layer=output_layer)
            if logits.ndim > 2:
                logits = logits.view(logits.size(0), -1)
            return torch.argmax(logits, dim=1).cpu()

    def evaluate(
        self,
        x,
        y,
        training_step: TrainingStep = SupervisedTrainingStep(),
        metrics=None,
        output_layer=None,
    ):
        if metrics is None:
            metrics = [EvaluationMetric(name="accuracy", function=accuracy_score)]

        self.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)

            outputs = self._run_evaluation_pass(
                x=x,
                y=y,
                training_step=training_step,
                output_layer=output_layer,
            )

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
