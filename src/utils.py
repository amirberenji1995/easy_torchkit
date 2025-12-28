import torch
from typing import Callable


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


def contrastive_step(*, model, xb, yb, optimizer, loss_fn, output_layer=None):
    model.train()
    optimizer.zero_grad()
    
    x1, x2 = xb[:, 0, :], xb[:, 1, :]
    
    z1 = model(x1, output_layer=output_layer)
    z2 = model(x2, output_layer=output_layer)
    
    yb = yb.float().view(-1)
    
    loss = loss_fn(z1, z2, yb)
    loss.backward()
    optimizer.step()
    
    embeddings = torch.cat([z1, z2], dim=0)
    return loss.detach(), embeddings.detach()
