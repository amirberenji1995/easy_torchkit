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

    def forward(self, z1, z2, y):
        # y = 1 → similar, 0 → dissimilar
        dist = torch.nn.functional.pairwise_distance(z1, z2)
        loss = y * dist.pow(2) + (1 - y) * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss.mean()


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
    xb: torch.Tensor,  # [B, 2, C, L]
    yb: torch.Tensor,  # [B, 1] or [B]
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    output_layer: str | None = None,
):
    model.train()
    optimizer.zero_grad()

    # Split pairs
    x1 = xb[:, 0]  # [B, C, L]
    x2 = xb[:, 1]  # [B, C, L]

    # Forward independently
    z1 = model(x1, output_layer=output_layer)
    z2 = model(x2, output_layer=output_layer)

    # Flatten embeddings if needed
    z1 = z1.view(z1.size(0), -1)
    z2 = z2.view(z2.size(0), -1)

    yb = yb.view(-1).float()

    loss = loss_fn(z1, z2, yb)
    loss.backward()
    optimizer.step()

    # Return concatenated embeddings for logging compatibility
    logits = torch.cat([z1.detach(), z2.detach()], dim=0)

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
