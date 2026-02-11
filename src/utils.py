import torch
from typing import Callable
import torch.nn.functional as F


def training_step(
    *,
    model: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    output_layer: str | None = None,
) -> torch.Tensor: ...


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor):
        # Calculate Euclidean distance agnostic of input dimensions (flattening all but batch)
        # y = 1 for similar, 0 for dissimilar
        diff = z1 - z2
        dist = torch.norm(diff.reshape(diff.size(0), -1), p=2, dim=1)

        loss_similar = y * dist.pow(2)
        loss_dissimilar = (1 - y) * torch.clamp(self.margin - dist, min=0).pow(2)

        return (loss_similar + loss_dissimilar).mean()


def supervised_step(*, model, xb, yb, optimizer, loss_fn, output_layer=None):
    model.train()
    optimizer.zero_grad()
    logits = model(xb, output_layer=output_layer)
    loss = loss_fn(logits, yb)
    loss.backward()
    optimizer.step()
    return loss.detach(), logits.detach()


def contrastive_step(
    *,
    model: torch.nn.Module,
    xb: torch.Tensor | tuple,  # Unpacks if passed as (x1, x2)
    yb: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    output_layer: str | None = None,
):
    model.train()
    optimizer.zero_grad()

    # Rule 3: Handle unpacking here. Assumes xb is (x1, x2) or similar
    x1, x2 = xb if isinstance(xb, (list, tuple)) else (xb[:, 0], xb[:, 1])

    z1 = model(x1, output_layer=output_layer)
    z2 = model(x2, output_layer=output_layer)

    loss = loss_fn(z1, z2, yb.view(-1).float())
    loss.backward()
    optimizer.step()

    # Rule 5: Facilitate Accuracy calculation
    # We create pseudo-logits where:
    # Column 0: Dissimilarity (represented by distance)
    # Column 1: Similarity (represented by proximity to 0)
    with torch.no_grad():
        diff = z1 - z2
        dist = torch.norm(diff.reshape(diff.size(0), -1), p=2, dim=1)
        # Margin logic: if distance < 0.5 * margin, it's 'Similar'
        margin = getattr(loss_fn, "margin", 1.0)
        logits = torch.stack([dist, margin - dist], dim=1)

    return loss.detach(), logits


def dynamic_bootstrapping_step(
    *,
    model,
    xb,
    yb,
    optimizer,
    loss_fn,
    output_layer=None,
    alpha=0.8,
):
    model.train()
    optimizer.zero_grad()

    logits = model(xb, output_layer=output_layer)
    probs = F.softmax(logits, dim=1)

    num_classes = logits.size(1)
    y_one_hot = F.one_hot(yb, num_classes=num_classes).float()

    with torch.no_grad():
        max_probs, _ = torch.max(probs, dim=1, keepdim=True)
        beta = 1.0 - (max_probs * (1.0 - alpha))

    refurbished_targets = (beta * y_one_hot) + ((1 - beta) * probs.detach())

    loss = torch.sum(-refurbished_targets * F.log_softmax(logits, dim=1), dim=1).mean()

    loss.backward()
    optimizer.step()

    return loss.detach(), logits.detach()
