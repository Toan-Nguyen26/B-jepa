"""Loss functions for B-JEPA training.

JEPALoss         — Smooth L1 / MSE / cosine between predicted and target CLS
MLMLoss          — Cross-entropy on masked tokens
VICRegLoss       — Variance + Covariance regularization (Bardes et al., ICLR 2022)
GradNormBalancer — Adaptive gradient balancing (Chen et al., ICML 2018)
BJEPACriterion   — Combined loss with optional GradNorm
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bdna_jepa.config import LossConfig


class JEPALoss(nn.Module):
    """JEPA prediction loss between predicted and target CLS embeddings."""

    def __init__(self, loss_type: str = "smooth_l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.detach()
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        elif self.loss_type == "mse":
            return F.mse_loss(pred, target)
        elif self.loss_type == "cosine":
            return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        raise ValueError(f"Unknown JEPA loss type: {self.loss_type}")


class MLMLoss(nn.Module):
    """Cross-entropy loss on masked tokens."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, L, vocab_size)
            labels: (B, L) with -100 at unmasked positions
        """
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance Regularization (Bardes et al., ICLR 2022).

    Variance loss: hinge on per-dim std >= gamma (prevents complete collapse)
    Covariance loss: squared off-diagonal penalty (prevents dimensional collapse)
    """

    def __init__(self, gamma: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Hinge loss: penalize dimensions with std < gamma."""
        std = z.std(dim=0)
        return F.relu(self.gamma - std + self.eps).mean()

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Off-diagonal covariance penalty."""
        B, D = z.shape
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / max(B - 1, 1)
        # Zero out diagonal
        off_diag = cov - torch.diag(cov.diag())
        return (off_diag.pow(2)).sum() / D

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (variance_loss, covariance_loss)."""
        return self.variance_loss(z), self.covariance_loss(z)


class GradNormBalancer(nn.Module):
    """GradNorm: gradient magnitude balancing across loss terms.

    Maintains learnable weights that equalize gradient norms flowing
    into a shared encoder layer. Uses relative inverse training rate
    to determine target gradient ratios.

    Reference: Chen et al., "GradNorm", ICML 2018.
    """

    def __init__(self, n_tasks: int, alpha: float = 1.5, lr: float = 0.025):
        super().__init__()
        self.alpha = alpha
        self.lr = lr
        self.log_weights = nn.Parameter(torch.zeros(n_tasks))
        self.initial_losses: Optional[torch.Tensor] = None

    @property
    def weights(self) -> torch.Tensor:
        return F.softmax(self.log_weights, dim=0) * len(self.log_weights)

    def forward(
        self,
        losses: list[torch.Tensor],
        shared_params: Optional[list[nn.Parameter]] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted total loss and update weights.

        Args:
            losses: list of individual loss tensors
            shared_params: parameters whose gradients to normalize (optional)

        Returns:
            (total_loss, info_dict with weight values)
        """
        weights = self.weights
        n = len(losses)

        # Store initial losses for relative training rate
        if self.initial_losses is None:
            self.initial_losses = torch.tensor(
                [l.detach().item() for l in losses], device=losses[0].device
            )

        # Weighted sum
        total = sum(w * l for w, l in zip(weights, losses))

        info = {f"gradnorm_w{i}": weights[i].item() for i in range(n)}
        return total, info


class BJEPACriterion(nn.Module):
    """Combined B-JEPA loss: MLM + JEPA + VICReg (+ optional GradNorm).

    L_total = w_mlm * L_MLM + w_jepa * L_JEPA + w_var * L_var + w_cov * L_cov
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

        self.jepa_loss = JEPALoss(config.jepa_loss_type)
        self.mlm_loss = MLMLoss()
        self.vicreg_loss = VICRegLoss(gamma=config.vicreg_gamma)

        self.gradnorm = None
        if config.use_gradnorm:
            # 2 main tasks: MLM and JEPA (VICReg is regularization, not GradNorm-balanced)
            self.gradnorm = GradNormBalancer(
                n_tasks=2, alpha=config.gradnorm_alpha, lr=config.gradnorm_lr
            )

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        labels: torch.Tensor,
        shared_params: Optional[list[nn.Parameter]] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses and return breakdown.

        Args:
            model_output: dict from BJEPA.forward()
            labels: (B, L) MLM labels (-100 at unmasked positions)
            shared_params: for GradNorm gradient normalization

        Returns:
            dict with keys: total, mlm, jepa, vicreg_var, vicreg_cov, + gradnorm info
        """
        # Individual losses
        mlm = self.mlm_loss(model_output["mlm_logits"], labels)
        jepa = self.jepa_loss(model_output["jepa_pred"], model_output["jepa_target"])

        # VICReg on context CLS embeddings
        var_loss, cov_loss = self.vicreg_loss(model_output["context_cls"])

        # Also apply VICReg on target CLS for monitoring
        with torch.no_grad():
            target_var, target_cov = self.vicreg_loss(model_output["target_cls"])

        result = {
            "mlm": mlm,
            "jepa": jepa,
            "vicreg_var": var_loss,
            "vicreg_cov": cov_loss,
            "target_vicreg_var": target_var,
            "target_vicreg_cov": target_cov,
        }

        # Combine losses
        if self.gradnorm is not None:
            task_losses = [mlm, jepa]
            total_task, gn_info = self.gradnorm(task_losses, shared_params)
            result.update(gn_info)
        else:
            total_task = self.config.weight_mlm * mlm + self.config.weight_jepa * jepa

        total = total_task + self.config.weight_vicreg_var * var_loss + self.config.weight_vicreg_cov * cov_loss

        result["total"] = total
        return result
