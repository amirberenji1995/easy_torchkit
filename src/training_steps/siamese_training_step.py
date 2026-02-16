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
    def _prepare_pair_batch(self, xb, yb):
        """
        Create positive and negative pairs from a batch.
        """
        device = xb.device

        pairs_1 = []
        pairs_2 = []
        labels = []

        for i in range(len(xb)):
            # positive pair
            same_class = torch.where(yb == yb[i])[0]
            j = same_class[torch.randint(len(same_class), (1,))]
            pairs_1.append(xb[i])
            pairs_2.append(xb[j])
            labels.append(1)

            # negative pair
            diff_class = torch.where(yb != yb[i])[0]
            j = diff_class[torch.randint(len(diff_class), (1,))]
            pairs_1.append(xb[i])
            pairs_2.append(xb[j])
            labels.append(0)

        return (
            torch.stack(pairs_1).to(device),
            torch.stack(pairs_2).to(device),
            torch.tensor(labels, dtype=torch.float32, device=device),
        )

    def train_batch(self, *, model, xb, yb, optimizer, loss_fn, output_layer=None):
        model.train()
        optimizer.zero_grad()

        x1, x2, pair_labels = self._prepare_pair_batch(xb, yb)

        z1 = model(x1, output_layer=output_layer)
        z2 = model(x2, output_layer=output_layer)

        loss = loss_fn(z1, z2, pair_labels)
        loss.backward()
        optimizer.step()

        return loss.item(), ModelOutput(
            logits=z1,  # <-- IMPORTANT: tensor, so _compute_metrics won't crash
            loss_input=(z1, z2, pair_labels),
            preds=None,
        )

    def eval_batch(self, *, model, xb, yb=None, output_layer=None):
        model.eval()
        with torch.no_grad():
            if yb is None:
                return ModelOutput(logits=model(xb), loss_input=None, preds=None)

            x1, x2, pair_labels = self._prepare_pair_batch(xb, yb)

            z1 = model(x1, output_layer=output_layer)
            z2 = model(x2, output_layer=output_layer)

            return ModelOutput(
                logits=z1,
                loss_input=(z1, z2, pair_labels),
                preds=None,
            )
