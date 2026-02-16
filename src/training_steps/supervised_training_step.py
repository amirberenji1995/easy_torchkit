import torch
from typing import Callable, Tuple
from .training_step_protocol import TrainingStep, ModelOutput


class SupervisedTrainingStep(TrainingStep):
    def train_batch(
        self,
        *,
        model: torch.nn.Module,
        xb: torch.Tensor,
        yb: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        output_layer: str | None = None,
    ) -> Tuple[torch.Tensor, ModelOutput]:

        optimizer.zero_grad(set_to_none=True)

        logits = model(xb, output_layer=output_layer)
        loss = loss_fn(logits, yb)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        return (
            loss.detach().item(),
            ModelOutput(
                loss_input=(logits.detach(), yb),
                preds=preds.detach(),
                logits=logits.detach(),
            ),
        )

    def eval_batch(
        self,
        *,
        model: torch.nn.Module,
        xb: torch.Tensor,
        yb: torch.Tensor | None = None,
        output_layer: str | None = None,
    ) -> ModelOutput:

        logits = model(xb, output_layer=output_layer)
        preds = torch.argmax(logits, dim=1)

        return ModelOutput(
            loss_input=(logits, yb) if yb is not None else None,
            preds=preds,
            logits=logits,
        )
