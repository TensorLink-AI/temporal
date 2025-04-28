import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming these loss functions are defined in temporal/losses/loss_functions.py
from temporal.modules.losses.loss_functions import (
    QuantileLoss,
    MQLoss,
    WeightedQuantileLoss,
    KernelEnergyLoss,
    EnergyDistanceLoss,
    SpectralLoss,
    FastSoftDTWLoss
)

# Assuming BaseLoss is not defined elsewhere and TimeSeriesLoss should inherit from nn.Module
class TimeSeriesLoss(nn.Module):
    def __init__(
        self,
        loss_type: str = "mse",
        quantiles: list = [0.1, 0.5, 0.9],
        output_token_len: int = 1,
        reduction: str = "mean",
        **kwargs
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.quantiles = quantiles
        self.output_token_len = output_token_len

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == "rmse":
            # Note: RMSE is typically calculated as sqrt(MSE) over the *mean* loss.
            # This implementation applies sqrt element-wise before reducing.
            # Consider torch.sqrt(F.mse_loss(..., reduction='mean')) if standard RMSE is needed.
            self.loss_fn = lambda x, y: torch.sqrt(F.mse_loss(x, y, reduction=reduction))
        elif loss_type == "quantile":
            # Assumes quantiles list is not empty and uses the first quantile
            self.loss_fn = QuantileLoss(quantile=quantiles[0], reduction=reduction)
        elif loss_type == "mq":
            self.loss_fn = MQLoss(quantiles=quantiles, reduction=reduction)
        elif loss_type == "wql":
            self.loss_fn = WeightedQuantileLoss(quantiles=quantiles, reduction=reduction)
        elif loss_type == "kernel_energy":
            # Note: KernelEnergyLoss and EnergyDistanceLoss typically work on samples, not direct predictions.
            # Ensure preds/targets are in the expected format for these losses.
            self.loss_fn = KernelEnergyLoss(reduction=reduction)
        elif loss_type == "energy":
            self.loss_fn = EnergyDistanceLoss(reduction=reduction)
        elif loss_type == "spectral":
            self.loss_fn = SpectralLoss(reduction=reduction)
        elif loss_type == "softdtw":
            gamma = kwargs.get("gamma", 1.0)
            self.loss_fn = FastSoftDTWLoss(gamma=gamma, reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Additional check for quantile losses: ensure preds has the expected shape [B, T, Q]
        if self.loss_type in ("quantile", "mq", "wql") and preds.ndim == targets.ndim:
             # If predictions don't have a quantile dimension but target doesn't either,
             # assume single output (e.g., median) and reshape predictions for QL/MQLoss.
             # This might need further refinement based on model output.
             if preds.ndim == 2: # Assume [B, T] -> reshape to [B, T, 1] for quantile loss expecting [B, T, Q]
                 preds = preds.unsqueeze(-1)

        # Ensure target shape matches prediction shape for element-wise losses if needed
        if self.loss_type not in ("quantile", "mq", "wql", "kernel_energy", "energy"): # Element-wise losses
             if preds.shape != targets.shape:
                  # Attempt to unsqueeze target if preds has an extra dim (e.g. single quantile output)
                  if preds.ndim == targets.ndim + 1 and preds.shape[-1] == 1:
                       targets = targets.unsqueeze(-1)
                  elif preds.ndim + 1 == targets.ndim and targets.shape[-1] == 1:
                       # Handle case where target has an extra dim of size 1
                       targets = targets.squeeze(-1)

             # If shapes still don't match, loss function will likely raise an error.
             # Add more specific shape checks if necessary based on expected loss inputs.

        return self.loss_fn(preds, targets)
