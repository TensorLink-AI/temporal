import torch
import torch.nn as nn
from temporal.losses.loss_functions import MQLoss, QuantileLoss
from typing import Dict, Callable

class BaseLoss(nn.Module):
    """Base class for time series loss functions."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, predictions, labels, loss_masks=None):
        """Override this method in child classes for custom loss functions."""
        raise NotImplementedError("Each loss class must implement its own forward method.")

class TimeSeriesLoss(BaseLoss):
    def __init__(self, config, loss_type="mse", quantile=0.5):
        super().__init__(config)
        self.loss_type = loss_type.lower()
        self.quantiles = quantile  # Could be float or list

        # Build correct loss function
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
            self.loss_fn = lambda preds, labels: torch.sqrt(nn.MSELoss(reduction="none")(preds, labels))
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def forward(self, predictions, labels):
        return self.loss_fn(predictions, labels)


    def forward(self, predictions, labels, loss_masks=None, output_token_len=None):
        """
        Computes loss based on the selected loss function.

        Args:
            predictions (torch.Tensor): Model predictions `[batch, seq_len, features]`
            labels (torch.Tensor): Ground truth labels `[batch, seq_len, features]`
            loss_masks (torch.Tensor, optional): Mask tensor `[batch, seq_len]` (1=valid, 0=ignore).
            output_token_len (int, optional): Number of output tokens per step (default: from config).

        Returns:
            torch.Tensor: Computed loss.
        """
        if output_token_len is None:
            output_token_len = self.config.output_token_len

        seq_len = predictions.shape[1] * self.config.input_token_len
        labels = labels[:, :seq_len - self.config.input_token_len + output_token_len]
        shift_labels = labels.unfold(dimension=-1, size=output_token_len, step=self.config.input_token_len)

        # Compute loss using selected function
        loss_fn = self.loss_functions[self.loss_type]
        losses = loss_fn(predictions, shift_labels)

        # Apply loss mask if provided
        if loss_masks is not None:
            losses = losses * loss_masks
            loss = losses.sum() / loss_masks.sum()
        else:
            loss = torch.mean(losses)

        return loss


class HybridTimeSeriesLoss(nn.Module):
    """
    Hybrid loss function that allows stacking multiple losses with custom weights.

    Example:
    ```python
    loss_fn = HybridTimeSeriesLoss({
        "mse": {"loss": TimeSeriesLoss("mse"), "weight": 0.4},
        "quantile": {"loss": TimeSeriesLoss("quantile"), "weight": 0.6}
    })
    ```
    """

    def __init__(self, losses: Dict[str, Dict[str, Callable]]):
        """
        Args:
            losses (dict): A dictionary where keys are loss names and values are dicts containing:
                - "loss": Instance of `TimeSeriesLoss`
                - "weight": Weighting factor for the loss
        """
        super().__init__()

        self.losses = nn.ModuleDict({name: loss_dict["loss"] for name, loss_dict in losses.items()})
        self.weights = {name: loss_dict["weight"] for name, loss_dict in losses.items()}

        assert sum(self.weights.values()) > 0, "Sum of weights must be greater than zero"

    def forward(self, preds, target):
        """
        Compute the hybrid loss as a weighted sum of all loss components.

        Args:
            preds (torch.Tensor): Model predictions
            target (torch.Tensor): Ground truth

        Returns:
            torch.Tensor: Computed hybrid loss value.
        """
        total_loss = 0.0
        for name, loss_fn in self.losses.items():
            loss = loss_fn(preds, target)
            total_loss += self.weights[name] * loss
        return total_loss