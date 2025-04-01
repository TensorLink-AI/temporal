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
