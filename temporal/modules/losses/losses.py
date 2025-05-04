import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming these loss functions are defined in temporal/losses/loss_functions.py
# We might need to restructure imports if BaseLoss is intended
from temporal.modules.losses.loss_functions import (
    QuantileLoss,
    MQLoss,
    WeightedQuantileLoss,
    KernelEnergyLoss,
    EnergyDistanceLoss,
    SpectralLoss,
    FastSoftDTWLoss
)
# Import the CRPS function
from temporal.losses.crps_loss_ensemble import crps_ensemble
# Import the registry decorator
from temporal.registry.core import register_module

# Assuming BaseLoss is not defined elsewhere and TimeSeriesLoss should inherit from nn.Module
class BaseLoss(nn.Module):
    """Base class for loss functions, inheriting from nn.Module."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction type: {reduction}")
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method")

    def _apply_reduction(self, loss: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """Applies reduction to the loss tensor, considering the mask."""
        if loss_mask is not None:
            # Ensure mask has same dimensions as loss for element-wise multiplication
            # Or can be broadcasted. Typically loss is [B, T] and mask is [B, T]
            if loss_mask.shape != loss.shape:
                 # Attempt broadcasting if mask is [B, T] and loss is [B, T, ...]
                 if loss_mask.ndim == loss.ndim: # Allow broadcasting if mask matches leading dims
                      mask_expanded_shape = loss_mask.shape + (1,) * (loss.ndim - loss_mask.ndim)
                      loss_mask = loss_mask.view(mask_expanded_shape).expand_as(loss)
                 elif loss_mask.shape == loss.shape[:loss_mask.ndim]: # Old logic
                     loss_mask = loss_mask.unsqueeze(-1).expand_as(loss)
                 else:
                      raise ValueError(f"Loss shape {loss.shape} and mask shape {loss_mask.shape} are incompatible.")

            loss = loss * loss_mask
            if self.reduction == "mean":
                # Compute mean only over masked elements
                return loss.sum() / loss_mask.sum().clamp(min=1e-9) # Avoid division by zero
            elif self.reduction == "sum":
                return loss.sum()
            else: # reduction == "none"
                return loss # Return masked loss per element
        else:
            # No mask, apply standard reduction
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else: # reduction == "none"
                return loss


@register_module("loss", "timeseries_generic") # Register this loss
class TimeSeriesLoss(BaseLoss): # Inherit from BaseLoss
    def __init__(
        self,
        loss_type: str = "mse",
        quantiles: list = [0.1, 0.5, 0.9],
        # output_token_len: int = 1, # This seems unused, consider removing
        reduction: str = "mean",
        **kwargs
    ):
        super().__init__(reduction=reduction) # Pass reduction to BaseLoss
        self.loss_type = loss_type
        # self.reduction = reduction # Handled by BaseLoss
        self.quantiles = quantiles
        # self.output_token_len = output_token_len

        # --- Define Loss Function based on type ---
        # We need to adapt these to potentially handle masks if BaseLoss doesn't do it automatically
        # For nn losses, we'll set their reduction to 'none' and handle reduction in forward
        nn_reduction = 'none' # Apply mask and reduction manually later

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=nn_reduction)
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss(reduction=nn_reduction)
        elif loss_type == "rmse":
             # RMSE needs mean first, then sqrt. Handle this logic in forward.
             self._mse_for_rmse = nn.MSELoss(reduction='none') # Use MSE with no reduction first
             self.loss_fn = None # Indicate special handling needed
        elif loss_type == "quantile":
            self.loss_fn = QuantileLoss(quantile=quantiles[0], reduction=nn_reduction)
        elif loss_type == "mq":
            self.loss_fn = MQLoss(quantiles=quantiles, reduction=nn_reduction)
        elif loss_type == "wql":
             # Check if WeightedQuantileLoss handles reduction internally or needs adaptation
            self.loss_fn = WeightedQuantileLoss(quantiles=quantiles, reduction=nn_reduction)
        elif loss_type == "kernel_energy":
            self.loss_fn = KernelEnergyLoss(reduction=nn_reduction)
        elif loss_type == "energy":
            self.loss_fn = EnergyDistanceLoss(reduction=nn_reduction)
        elif loss_type == "spectral":
            self.loss_fn = SpectralLoss(reduction=nn_reduction)
        elif loss_type == "softdtw":
            gamma = kwargs.get("gamma", 1.0)
            # Check if FastSoftDTWLoss handles reduction or needs adaptation
            self.loss_fn = FastSoftDTWLoss(gamma=gamma, reduction=nn_reduction)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        # --- Shape Adjustments ---
        # Ensure target has extra dim if needed for broadcasting with quantile outputs
        if self.loss_type in ("quantile", "mq", "wql"):
            if preds.ndim == targets.ndim + 1 and preds.shape[-1] == len(self.quantiles):
                 targets = targets.unsqueeze(-1) # Make target [B, T, 1] for broadcasting

        # Ensure target shape matches prediction shape for element-wise losses if needed
        elif self.loss_type in ("mse", "mae", "rmse"): # Element-wise losses
             if preds.shape != targets.shape:
                  # Attempt to unsqueeze target if preds has an extra dim (e.g. single output model)
                  if preds.ndim == targets.ndim + 1 and preds.shape[-1] == 1:
                       targets = targets.unsqueeze(-1)
                  elif preds.ndim + 1 == targets.ndim and targets.shape[-1] == 1:
                       targets = targets.squeeze(-1) # Or squeeze target if it has extra dim

             # If shapes still don't match after adjustments, the loss_fn call will likely fail
             if preds.shape != targets.shape:
                  raise ValueError(f"Shape mismatch for loss '{self.loss_type}': preds {preds.shape}, targets {targets.shape}")

        # --- Calculate Loss ---
        if self.loss_type == "rmse":
             # Calculate element-wise MSE first
             mse_loss = self._mse_for_rmse(preds, targets)
             # Apply mask *before* sqrt to avoid issues with masked zeros
             if loss_mask is not None:
                  # Ensure mask compatibility
                  if loss_mask.shape != mse_loss.shape:
                      if loss_mask.shape == mse_loss.shape[:loss_mask.ndim]:
                           loss_mask = loss_mask.unsqueeze(-1).expand_as(mse_loss)
                      else:
                           raise ValueError(f"RMSE Loss shape {mse_loss.shape} and mask shape {loss_mask.shape} are incompatible.")
                  masked_mse_loss = mse_loss * loss_mask
                  # Calculate mean MSE over *masked* elements
                  mean_masked_mse = masked_mse_loss.sum() / loss_mask.sum().clamp(min=1e-9)
                  loss = torch.sqrt(mean_masked_mse) # Final RMSE is sqrt of mean
                  # Since reduction='mean' is implied by RMSE calculation, return the scalar
                  return loss
             else:
                  # No mask, calculate standard RMSE
                  mean_mse = mse_loss.mean()
                  loss = torch.sqrt(mean_mse)
                  return loss # Return scalar RMSE

        else:
             # Calculate element-wise loss using the specific loss_fn
             elementwise_loss = self.loss_fn(preds, targets)

             # --- Apply Mask and Reduction using BaseLoss helper ---
             # Note: Quantile/MQ losses might return shape [B, T, Q]. We might need to average over Q first.
             # Let's assume loss_fn returns [B, T] or similar shape compatible with mask [B, T]
             if self.loss_type in ("mq", "wql"):
                  # MQLoss/WQLoss might return shape [B, T, Q]. Check and average over quantiles if needed.
                  if elementwise_loss.ndim > targets.ndim and elementwise_loss.shape[-1] == len(self.quantiles):
                      elementwise_loss = elementwise_loss.mean(dim=-1) # Now shape [B, T]

             return self._apply_reduction(elementwise_loss, loss_mask)


# --- New CRPS Loss Module ---
@register_module("loss", "crps") # Register this loss
class CRPSLoss(BaseLoss):
    """
    Computes the Continuous Ranked Probability Score (CRPS) using the ensemble/quantile approach.
    Uses the Probability Weighted Moments (PWM) estimator for efficiency.
    """
    def __init__(self, reduction: str = "mean", estimator: str = "pwm", axis: int = -1, **kwargs): # Added **kwargs
        """
        Args:
            reduction (str): Specifies the reduction to apply: 'none', 'mean', 'sum'.
            estimator (str): CRPS estimator ('pwm', 'nrg', 'fair'). Defaults to 'pwm'.
            axis (int): Dimension corresponding to the ensemble/quantiles in predictions. Defaults to -1.
            **kwargs: Catches unused arguments like 'quantiles' from the config.
        """
        super().__init__(reduction=reduction)
        if estimator not in ["pwm", "nrg", "fair"]:
            raise ValueError(f"Invalid estimator '{estimator}'. Choose 'pwm', 'nrg', or 'fair'.")
        self.estimator = estimator
        self.axis = axis # Store the ensemble axis

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the CRPS loss.

        Args:
            preds (torch.Tensor): Model predictions (ensemble/quantiles). Shape e.g., [B, T, K] or [B, K, T].
                                   The dimension specified by `self.axis` is treated as the ensemble.
            targets (torch.Tensor): Ground truth values. Shape e.g., [B, T] or [B, T, 1]. Must be broadcastable
                                    against `preds` after removing the ensemble dimension.
            loss_mask (torch.Tensor, optional): Boolean or float tensor for masking loss elements.
                                                Shape e.g., [B, T]. Defaults to None.

        Returns:
            torch.Tensor: The calculated CRPS loss (scalar if reduction is 'mean' or 'sum').
        """
        # Ensure target shape is suitable for crps_ensemble (expects obs shape matching forecasts except for ensemble axis)
        # Example: preds [B, T, K], targets [B, T] -> unsqueeze targets to [B, T, 1]
        # Example: preds [B, K, T], targets [B, T] -> unsqueeze targets to [B, 1, T]
        if targets.ndim == preds.ndim - 1:
            # Find the ensemble dimension in preds
            ensemble_dim_index = self.axis if self.axis >= 0 else preds.ndim + self.axis
            # Unsqueeze targets at that dimension
            targets = targets.unsqueeze(ensemble_dim_index)
        # Check if target shape is now broadcastable (matches preds shape excluding the ensemble dim)
        # Create the expected shape of targets by removing the ensemble dim from preds shape
        expected_target_shape_list = list(preds.shape)
        # Need to handle negative axis index correctly
        actual_axis = self.axis if self.axis >= 0 else preds.ndim + self.axis
        if 0 <= actual_axis < preds.ndim:
             del expected_target_shape_list[actual_axis]
        else:
             raise ValueError(f"Invalid axis {self.axis} for preds shape {preds.shape}")
        expected_target_shape = torch.Size(expected_target_shape_list)

        # Check if target shape matches the expected shape or has an extra singleton dimension
        if list(targets.shape) != expected_target_shape_list:
             # Check if target has a singleton dimension where the ensemble dim was
             target_shape_with_singleton = list(expected_target_shape)
             target_shape_with_singleton.insert(actual_axis, 1)
             if list(targets.shape) != target_shape_with_singleton:
                   raise ValueError(
                       f"Shape mismatch for CRPS: preds {preds.shape} (axis={self.axis}), "
                       f"targets {targets.shape}, expected targets shape {expected_target_shape} "
                       f"or {target_shape_with_singleton}"
                   )

        # Calculate element-wise CRPS (returns shape without ensemble dim, e.g., [B, T])
        elementwise_crps = crps_ensemble(
            observations=targets,
            forecasts=preds,
            estimator=self.estimator,
            axis=self.axis,
            reduce=False # Get per-element loss before reduction
        )

        # Apply mask (if any) and reduction using the helper method
        return self._apply_reduction(elementwise_crps, loss_mask)
