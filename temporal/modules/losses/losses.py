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
    FastSoftDTWLoss,
    SpreadPenalty, # Added SpreadPenalty
    MixtureLoss # Added MixtureLoss
)
# Import the CRPS function
from temporal.losses.crps_loss_ensemble import crps_ensemble
# Import the registry decorator
from temporal.registry.core import register_module
from typing import Optional, Tuple

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



@register_module("loss", "crps")
class CRPSLoss(BaseLoss):
    """
    Continuous Ranked Probability Score with per‐step spread penalty.
    """
    def __init__(
        self,
        reduction: str = "mean",
        estimator: str = "pwm",
        axis: int = -1,
        scaling_type: str = "none",
        scaling_dim: int = 1,
        scaling_eps: float = 1e-8,
        spread_lambda: float = 0.0,
        spread_penalty_type: str = "symmetric_log",
        spread_penalty_epsilon: float = 1e-3,
        spread_target_spread: float = 0.0,
        **kwargs
    ):
        super().__init__(reduction=reduction)
        # Validate
        if estimator not in ["pwm", "nrg", "fair"]:
            raise ValueError(f"Invalid estimator '{estimator}'.")
        if scaling_type not in ["none", "std", "minmax"]:
            raise ValueError(f"Invalid scaling_type '{scaling_type}'.")
        if spread_penalty_type not in ["log", "inverse", "symmetric_log"]:
            raise ValueError(f"Invalid spread_penalty_type '{spread_penalty_type}'.")

        self.estimator = estimator
        self.axis = axis
        self.scaling_type = scaling_type
        self.scaling_dim = scaling_dim
        self.scaling_eps = scaling_eps

        self.spread_lambda = spread_lambda
        self.spread_penalty_fn = None
        if spread_lambda > 0.0:
            # per‐step penalty: no reduction inside SpreadPenalty
            self.spread_penalty_fn = SpreadPenalty(
                penalty_type=spread_penalty_type,
                epsilon=spread_penalty_epsilon,
                target_spread=spread_target_spread,
                reduction="none"
            )

    def forward(
        self,
        preds: torch.Tensor,       # [B, T, Q] or similar
        targets: torch.Tensor,     # [B, T] or broadcastable
        loss_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Unsqueeze targets to match preds if needed
        if targets.ndim == preds.ndim - 1:
            axis = self.axis if self.axis >= 0 else preds.ndim + self.axis
            targets = targets.unsqueeze(axis)

        # Sort forecasts along ensemble/quantile axis
        preds_sorted = torch.sort(preds, dim=self.axis)[0]

        # Compute elementwise CRPS → shape [B, T, ...] without ensemble dim
        elementwise_crps = crps_ensemble(
            observations=targets,
            forecasts=preds_sorted,
            estimator=self.estimator,
            axis=self.axis,
            reduce=False
        )

        # Add per‐step spread penalty if enabled
        if self.spread_lambda > 0.0 and self.spread_penalty_fn is not None:
            # assume preds_sorted is (..., T, Q) so penalty returns (..., T)
            spread_penalty_map = self.spread_penalty_fn(preds_sorted)
            elementwise_crps = elementwise_crps + self.spread_lambda * spread_penalty_map

        # Optional scaling
        if self.scaling_type != "none":
            # compute scaling factor along scaling_dim on original targets
            dim_size = targets.size(self.scaling_dim)
            if dim_size > 1:
                if self.scaling_type == "std":
                    factor = torch.std(targets, dim=self.scaling_dim, keepdim=True, unbiased=False)
                else:  # "minmax"
                    mn = torch.min(targets, dim=self.scaling_dim, keepdim=True).values
                    mx = torch.max(targets, dim=self.scaling_dim, keepdim=True).values
                    factor = mx - mn
                factor = factor + self.scaling_eps
                elementwise_crps = elementwise_crps / factor

        # Apply mask & reduce
        return self._apply_reduction(elementwise_crps, loss_mask)

# --- New Mixture Loss Wrapper ---
@register_module("loss", "mixture")
class RegisteredMixtureLoss(BaseLoss): # Inherits from BaseLoss for consistency
    """
    Registered wrapper for MixtureLoss.
    The actual MixtureLoss function (from loss_functions.py) handles its own reduction and masking.
    This wrapper primarily serves for registration and standardized __init__ from config.
    """
    def __init__(self, reduction: str = "mean", min_df: float = 2.0, fixed_sigma: float = 1e-3, **kwargs):
        """
        Args:
            reduction (str): Specifies the reduction for MixtureLoss: 'none', 'mean', 'sum'.
            min_df (float): Minimum degrees of freedom for StudentT components.
            fixed_sigma (float): Fixed standard deviation for FixedNormal components.
            **kwargs: Catches unused arguments from the config if any.
        """
        super().__init__(reduction=reduction) 
        self.loss_fn = MixtureLoss( # Instance of the actual MixtureLoss from loss_functions.py
            reduction=reduction,
            min_df=min_df,
            fixed_sigma=fixed_sigma
        )

    def forward(self, preds: dict, targets: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the Mixture NLL loss.
        Args:
            preds (dict): Predictions from MixtureOutputHead. Expected to be a dictionary.
            targets (torch.Tensor): Ground truth values. Shape [B, T].
            loss_mask (torch.Tensor, optional): Mask for loss elements. Shape [B, T].
        Returns:
            torch.Tensor: The calculated mixture loss.
        """
        # MixtureLoss itself handles reduction and masking based on its init params
        return self.loss_fn(preds=preds, targets=targets, loss_mask=loss_mask)
