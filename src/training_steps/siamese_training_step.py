from .training_step_protocol import TrainingStep, ModelOutput
import torch


class ContrastiveLoss(torch.nn.Module):
    """
    Basic contrastive loss for siamese networks.
    margin: distance margin for dissimilar pairs
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, labels):
        # labels: 1 if similar, 0 if dissimilar
        distances = torch.nn.functional.pairwise_distance(emb1, emb2)
        loss_similar = labels * distances.pow(2)
        loss_dissimilar = (1 - labels) * torch.clamp(
            self.margin - distances, min=0
        ).pow(2)
        return (loss_similar + loss_dissimilar).mean()


class SiameseTrainingStep(TrainingStep):
    def train_batch(self, *, model, xb, yb, optimizer, loss_fn, output_layer=None):
        model.train()
        optimizer.zero_grad()

        x1, x2 = xb[:, 0, :], xb[:, 1, :]
        z1 = model(x1, output_layer=output_layer)
        z2 = model(x2, output_layer=output_layer)

        loss = loss_fn(z1, z2, yb)
        loss.backward()
        optimizer.step()

        return loss.item(), ModelOutput(
            logits=z1,  # embedding tensor
            loss_input=(z1, z2, yb),
            preds=None,
        )

    def eval_batch(self, *, model, xb, yb=None, output_layer=None):
        model.eval()
        with torch.no_grad():
            x1, x2 = xb[:, 0, :], xb[:, 1, :]
            z1 = model(x1, output_layer=output_layer)
            z2 = model(x2, output_layer=output_layer)

            return ModelOutput(
                logits=z1,  # embedding tensor
                loss_input=(z1, z2, yb),
                preds=None,
            )
