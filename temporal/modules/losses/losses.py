import torch
import torch.nn as nn
from typing import Optional, Union, List
from temporal.losses.loss_functions import MQLoss, QuantileLoss


class BaseLoss(nn.Module):
    """Base class for time series loss functions."""
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, predictions, labels, loss_masks=None):
        raise NotImplementedError("Each loss class must implement its own forward method.")


class TimeSeriesLoss(BaseLoss):
    def __init__(
        self,
        config,
        loss_type: str = "mse",
        quantile: Union[float, List[float]] = None
    ):
        super().__init__(config)

        self.loss_type = loss_type.lower()
        self.quantiles = quantile or getattr(config, "quantiles", 0.5)

        if self.loss_type == "quantile":
            if isinstance(self.quantiles, list):
                if len(self.quantiles) > 1:
                    raise ValueError("Use 'mq' for multi-quantile loss, or pass a single float for quantile loss.")
                self.loss_fn = QuantileLoss(quantile=self.quantiles[0])
            else:
                self.loss_fn = QuantileLoss(quantile=self.quantiles)

        elif self.loss_type == "mq":
            if not isinstance(self.quantiles, list):
                raise ValueError("MQ loss requires a list of quantiles.")
            self.loss_fn = MQLoss(quantiles=self.quantiles)

        elif self.loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")

        elif self.loss_type == "mae":
            self.loss_fn = nn.L1Loss(reduction="none")

        elif self.loss_type == "rmse":
            self.loss_fn = lambda preds, labels: torch.sqrt(
                nn.MSELoss(reduction="none")(preds, labels)
            )

        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss_masks: Optional[torch.Tensor] = None,
        output_token_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Computes loss based on the selected loss function.

        Args:
            predictions (Tensor): [B, T, Q] or [B, T]
            labels (Tensor): [B, T, Q] or [B, T]
            loss_masks (Tensor, optional): [B, T] or broadcastable
            output_token_len (int, optional): If using shifting/sliding logic

        Returns:
            Tensor: Loss scalar
        """
        # If needed, align shapes here (for sliding window targets etc)
        if output_token_len is None:
            output_token_len = getattr(self.config, "output_token_len", 1)

        # Compute raw loss
        losses = self.loss_fn(predictions, labels)  # shape [B, T] or [B, T, Q]

        # Apply optional loss mask
        if loss_masks is not None:
            losses = losses * loss_masks.unsqueeze(-1) if losses.ndim == 3 else losses * loss_masks
            loss = losses.sum() / loss_masks.sum()
        else:
            loss = losses.mean()

        return loss



import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLLLoss(nn.Module):
    """
    Computes the negative log-likelihood of a Gaussian distribution
    where the model predicts mean and log-variance (logσ²) or log-std.

    Inputs:
        preds: Tensor of shape [B, T, 2] = [μ, logσ]
        targets: Tensor of shape [B, T] or [B, T, 1]
    Returns:
        scalar loss (mean across batch and time)
    """

    def __init__(self, reduction="mean", log_sigma=True):
        super().__init__()
        self.reduction = reduction
        self.log_sigma = log_sigma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mu, log_sigma = preds.unbind(dim=-1)  # [B, T]
        sigma = F.softplus(log_sigma) + 1e-3  # ensure positive std

        # Gaussian NLL: 0.5 * ((y - μ)² / σ² + log σ²)
        nll = 0.5 * ((targets - mu) ** 2 / (sigma ** 2)) + torch.log(sigma)

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll  # no reduction



import torch
import torch.nn as nn
import torch.nn.functional as F

class TDistributionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        # preds: [B, T, 3] => μ, log(σ), log(ν)
        mu, log_sigma, log_nu = preds.unbind(dim=-1)
        sigma = F.softplus(log_sigma) + 1e-3  # Ensure positivity
        nu = F.softplus(log_nu) + 2.0         # ν > 2 for finite variance

        # Compute the negative log-likelihood
        term1 = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
        term2 = -0.5 * torch.log(nu * torch.pi * sigma ** 2)
        term3 = -((nu + 1) / 2) * torch.log(1 + ((targets - mu) ** 2) / (nu * sigma ** 2))
        nll = -(term1 + term2 + term3)
        return nll.mean()
