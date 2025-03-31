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
    """
    Generalized loss class for time series forecasting.
    
    Supports:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - Quantile Loss
    - MultiQuantile Loss (MQ)
    """

    def __init__(self, config, loss_type="mse", quantile=0.5):
        """
        Args:
            config: Model configuration.
            loss_type (str): Loss function to use ("mse", "rmse", "mae", "quantile", "MQ").
            quantile (float): Quantile level for quantile loss (default 0.5).
        """
        super().__init__(config)
        self.loss_type = loss_type.lower()
        self.quantiles = quantile  # Only used for quantile loss

        # Mapping of loss types
        self.loss_functions = {
            "mse": nn.MSELoss(reduction="none"),
            "mae": nn.L1Loss(reduction="none"),
            "rmse": lambda preds, labels: torch.sqrt(nn.MSELoss(reduction="none")(preds, labels)),
            "quantile": QuantileLoss(quantile = self.quantiles ),
            "mq": MQLoss(quantiles = self.quantiles),  # Placeholder for CRPS implementation
        }

        if self.loss_type not in self.loss_functions:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}. Choose from {list(self.loss_functions.keys())}.")




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