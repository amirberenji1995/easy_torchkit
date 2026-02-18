import torch
import torch.nn.functional as F
from .training_step_protocol import TrainingStep, ModelOutput


class BetaMixtureModel:
    """
    Two-component Beta Mixture Model.
    Component 0: clean samples
    Component 1: noisy samples
    """

    def __init__(self, max_iters: int = 10, eps: float = 1e-6):
        self.max_iters = max_iters
        self.eps = eps

        # mixture weights
        self.pi = torch.tensor([0.5, 0.5])

        # beta params: alpha, beta for each component
        self.alpha = torch.tensor([2.0, 2.0])
        self.beta = torch.tensor([2.0, 2.0])

        self.fitted = False

    def _beta_pdf(self, x, alpha, beta):
        """
        Beta probability density function.
        x must be in (0,1)
        """
        log_pdf = (
            (alpha - 1) * torch.log(x + self.eps)
            + (beta - 1) * torch.log(1 - x + self.eps)
            - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))
        )
        return torch.exp(log_pdf)

    def fit(self, losses: torch.Tensor):
        """
        losses: Tensor in (0,1), shape [N]
        """
        losses = losses.detach().clamp(self.eps, 1 - self.eps)

        for _ in range(self.max_iters):
            # ---------- E-step ----------
            p0 = self.pi[0] * self._beta_pdf(losses, self.alpha[0], self.beta[0])
            p1 = self.pi[1] * self._beta_pdf(losses, self.alpha[1], self.beta[1])

            norm = p0 + p1 + self.eps
            gamma0 = p0 / norm
            gamma1 = p1 / norm

            # ---------- M-step ----------
            self.pi[0] = gamma0.mean()
            self.pi[1] = gamma1.mean()

            self.alpha[0], self.beta[0] = self._fit_beta(losses, gamma0)
            self.alpha[1], self.beta[1] = self._fit_beta(losses, gamma1)

        self.fitted = True

    def _fit_beta(self, x, weights):
        """
        Method-of-moments weighted Beta fit
        """
        w = weights / (weights.sum() + self.eps)

        mean = (w * x).sum()
        var = (w * (x - mean) ** 2).sum()

        var = torch.clamp(var, min=self.eps)

        common = mean * (1 - mean) / var - 1
        alpha = torch.clamp(mean * common, min=1e-2)
        beta = torch.clamp((1 - mean) * common, min=1e-2)

        return alpha, beta

    def predict_clean_probability(self, losses: torch.Tensor):
        """
        Returns P(sample is clean)
        """
        if not self.fitted:
            raise RuntimeError("BMM must be fitted before prediction")

        losses = losses.detach().clamp(self.eps, 1 - self.eps)

        p_clean = self.pi[0] * self._beta_pdf(losses, self.alpha[0], self.beta[0])
        p_noisy = self.pi[1] * self._beta_pdf(losses, self.alpha[1], self.beta[1])

        return p_clean / (p_clean + p_noisy + self.eps)


class DynamicBootstrappingTrainingStep(TrainingStep):
    """
    Implements loss-based label bootstrapping with a Beta Mixture Model.
    """

    def __init__(
        self,
        warmup_epochs: int = 5,
        bmm_iters: int = 10,
    ):
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        self.bmm = BetaMixtureModel(max_iters=bmm_iters)
        self.loss_buffer = []

    def on_epoch_start(self, epoch: int):
        self.current_epoch = epoch
        self.loss_buffer.clear()

    def train_batch(self, *, model, xb, yb, optimizer, loss_fn, output_layer=None):
        model.train()
        optimizer.zero_grad()

        logits = model(xb, output_layer=output_layer)
        probs = F.softmax(logits, dim=1)

        # Per-sample CE loss
        per_sample_loss = F.cross_entropy(logits, yb, reduction="none")

        # Store losses for BMM
        self.loss_buffer.append(per_sample_loss.detach())

        # Warmup: standard CE
        if self.current_epoch < self.warmup_epochs:
            loss = per_sample_loss.mean()
        else:
            # Normalize loss to (0,1)
            losses = per_sample_loss.detach()
            losses = losses / (losses.max() + 1e-6)

            clean_prob = self.bmm.predict_clean_probability(losses).to(xb.device)

            # One-hot labels
            y_onehot = F.one_hot(yb, num_classes=logits.size(1)).float()

            # Bootstrapped labels
            y_tilde = (
                clean_prob.unsqueeze(1) * y_onehot
                + (1 - clean_prob.unsqueeze(1)) * probs.detach()
            )

            loss = (-y_tilde * torch.log(probs + 1e-8)).sum(dim=1).mean()

        loss.backward()
        optimizer.step()

        return loss.item(), ModelOutput(
            logits=logits.detach(),
            loss_input=None,
            preds=probs.argmax(dim=1),
        )

    def eval_batch(self, *, model, xb, yb=None, output_layer=None):
        model.eval()
        with torch.no_grad():
            logits = model(xb, output_layer=output_layer)
            preds = logits.argmax(dim=1)

            return ModelOutput(
                logits=logits,
                loss_input=None,
                preds=preds,
            )

    def on_epoch_end(self):
        """
        Fit BMM after collecting losses for the epoch
        """
        if self.current_epoch < self.warmup_epochs:
            return

        losses = torch.cat(self.loss_buffer)
        losses = losses / (losses.max() + 1e-6)

        self.bmm.fit(losses.cpu())
