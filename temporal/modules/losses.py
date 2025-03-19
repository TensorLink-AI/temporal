import torch
import torch.nn as nn

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
    - Continuous Ranked Probability Score (CRPS)
    """

    def __init__(self, config, loss_type="mse", quantile=0.5):
        """
        Args:
            config: Model configuration.
            loss_type (str): Loss function to use ("mse", "rmse", "mae", "quantile", "crps").
            quantile (float): Quantile level for quantile loss (default 0.5).
        """
        super().__init__(config)
        self.loss_type = loss_type.lower()
        self.quantile = quantile  # Only used for quantile loss

        # Mapping of loss types
        self.loss_functions = {
            "mse": nn.MSELoss(reduction="none"),
            "mae": nn.L1Loss(reduction="none"),
            "rmse": lambda preds, labels: torch.sqrt(nn.MSELoss(reduction="none")(preds, labels)),
            "quantile": self.quantile_loss,
            "crps": self.crps_loss,  # Placeholder for CRPS implementation
        }

        if self.loss_type not in self.loss_functions:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}. Choose from {list(self.loss_functions.keys())}.")

    def quantile_loss(self, predictions, labels):
        """Computes quantile loss for given quantile level."""
        errors = labels - predictions
        return torch.max((self.quantile - 1) * errors, self.quantile * errors)

    def crps_loss(self, predictions, labels):
        """Placeholder for CRPS loss function (can be customized)."""
        return nn.MSELoss(reduction="none")(predictions, labels)  # Replace with actual CRPS implementation

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
