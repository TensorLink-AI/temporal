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
    Includes an option to scale the loss based on target sequence statistics.
    Optionally includes a spread penalty for regularization.
    """
    def __init__(self, reduction: str = "mean", estimator: str = "pwm", axis: int = -1,
                 scaling_type: str = "none", scaling_dim: int = 1, scaling_eps: float = 1e-8,
                 spread_lambda: float = 0.0, spread_penalty_type: str = 'log',
                 spread_penalty_epsilon: float = 1e-3,
                 **kwargs):
        """
        Args:
            reduction (str): Specifies the reduction to apply: 'none', 'mean', 'sum'.
            estimator (str): CRPS estimator ('pwm', 'nrg', 'fair'). Defaults to 'pwm'.
            axis (int): Dimension corresponding to the ensemble/quantiles in predictions. Defaults to -1.
            scaling_type (str): Type of scaling to apply to the loss: 'none', 'std', 'minmax'. Defaults to 'none'.
            scaling_dim (int): Dimension of the original target tensor along which to compute scaling statistics.
                               Defaults to 1 (typically the time dimension for targets [B, T, ...]).
            scaling_eps (float): Epsilon value added to the denominator for numerical stability during scaling.
                                 Defaults to 1e-8.
            spread_lambda (float): Coefficient for the spread penalty. If 0, penalty is not applied.
                                   Defaults to 0.0.
            spread_penalty_type (str): Type of spread penalty ('log' or 'inverse'). Defaults to 'log'.
            spread_penalty_epsilon (float): Epsilon for numerical stability in spread penalty.
                                            Defaults to 1e-3.
            **kwargs: Catches unused arguments like 'quantiles' from the config.
        """
        super().__init__(reduction=reduction)
        if estimator not in ["pwm", "nrg", "fair"]:
            raise ValueError(f"Invalid estimator '{estimator}'. Choose 'pwm', 'nrg', or 'fair'.")
        if scaling_type not in ["none", "std", "minmax"]:
            raise ValueError(f"Invalid scaling_type '{scaling_type}'. Choose 'none', 'std', or 'minmax'.")
        if spread_penalty_type not in ['log', 'inverse']:
            raise ValueError("spread_penalty_type must be 'log' or 'inverse'")

        self.estimator = estimator
        self.axis = axis # Store the ensemble axis
        self.scaling_type = scaling_type
        self.scaling_dim = scaling_dim
        self.scaling_eps = scaling_eps

        self.spread_lambda = spread_lambda
        self.spread_penalty_fn = None
        if self.spread_lambda > 0:
            self.spread_penalty_fn = SpreadPenalty(
                penalty_type=spread_penalty_type,
                epsilon=spread_penalty_epsilon,
                reduction='mean' # Penalty is mean over spread elements, then scaled by lambda
            )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the CRPS loss with optional spread penalty.

        Args:
            preds (torch.Tensor): Model predictions (ensemble/quantiles). Shape e.g., [B, T, K] or [B, K, T].
                                   The dimension specified by `self.axis` is treated as the ensemble.
                                   For spread penalty, last dimension is assumed to be quantiles.
            targets (torch.Tensor): Ground truth values. Shape e.g., [B, T] or [B, T, 1]. Must be broadcastable
                                    against `preds` after removing the ensemble dimension.
            loss_mask (torch.Tensor, optional): Boolean or float tensor for masking loss elements.
                                                Shape e.g., [B, T]. Defaults to None.

        Returns:
            torch.Tensor: The calculated CRPS loss (scalar if reduction is 'mean' or 'sum').
        """
        original_targets_for_scaling = targets # Store a reference for scaling calculations
        crps_targets = targets # This variable will be used for crps_ensemble, might be modified

        # Ensure target shape is suitable for crps_ensemble (expects obs shape matching forecasts except for ensemble axis)
        if crps_targets.ndim == preds.ndim - 1:
            ensemble_dim_index = self.axis if self.axis >= 0 else preds.ndim + self.axis
            crps_targets = crps_targets.unsqueeze(ensemble_dim_index)

        # Check if target shape is now broadcastable (matches preds shape excluding the ensemble dim)
        expected_target_shape_list = list(preds.shape)
        actual_axis = self.axis if self.axis >= 0 else preds.ndim + self.axis
        if 0 <= actual_axis < preds.ndim:
             del expected_target_shape_list[actual_axis]
        else:
             raise ValueError(f"Invalid axis {self.axis} for preds shape {preds.shape}")
        expected_target_shape = torch.Size(expected_target_shape_list)

        if list(crps_targets.shape) != expected_target_shape_list:
             target_shape_with_singleton = list(expected_target_shape)
             target_shape_with_singleton.insert(actual_axis, 1)
             if list(crps_targets.shape) != target_shape_with_singleton:
                   raise ValueError(
                       f"Shape mismatch for CRPS: preds {preds.shape} (axis={self.axis}), "
                       f"targets {crps_targets.shape} (after potential unsqueeze), expected targets shape {expected_target_shape} "
                       f"or {target_shape_with_singleton}"
                   )
        
        # Sort predictions along the ensemble/quantile axis for CRPS and spread calculation
        # CRPS ensemble often expects sorted forecasts. Spread penalty assumes sorted for q_low, q_high.
        preds_sorted = torch.sort(preds, dim=self.axis)[0]


        # Calculate element-wise CRPS (returns shape without ensemble dim, e.g., [B, T])
        elementwise_crps = crps_ensemble(
            observations=crps_targets, # Use the potentially unsqueezed targets
            forecasts=preds_sorted, # Use sorted predictions
            estimator=self.estimator,
            axis=self.axis,
            reduce=False # Get per-element loss before reduction
        )

        # --- Apply Spread Penalty ---
        if self.spread_lambda > 0 and self.spread_penalty_fn is not None:
            # SpreadPenalty expects quantiles as the last dimension (B, T, Q)
            # If self.axis is not the last dim for preds, we might need to permute or warn.
            # For now, assuming preds is (B, T, Q) if spread penalty is active,
            # which aligns with common use where axis=-1 for quantiles.
            if self.axis != -1 and self.axis != preds.ndim - 1:
                # This is a simplification. If axis is not last, SpreadPenalty might not work as intended
                # without a transpose, or SpreadPenalty needs to be axis-aware.
                # For now, we proceed assuming axis is compatible or user ensures preds are (..., Q)
                pass # Potentially add warning or transpose logic if necessary

            # We use preds_sorted here as spread penalty also benefits from sorted quantiles
            # and it's already available.
            spread_penalty_value = self.spread_penalty_fn(preds_sorted) # This returns a scalar mean penalty
            
            # Add scaled penalty to elementwise_crps.
            # Since spread_penalty_value is already a mean scalar, adding it to elementwise_crps (e.g. [B,T])
            # will broadcast. This means the penalty is uniformly applied across all batch/time elements
            # before the final reduction of the combined loss.
            elementwise_crps = elementwise_crps + (self.spread_lambda * spread_penalty_value)


        # --- Apply Scaling ---
        if self.scaling_type != "none":
            # Use original_targets_for_scaling for calculating std, min, max.
            # This tensor has the shape as it was passed into the forward method.
            # elementwise_crps has shape of original_targets_for_scaling, or compatible (e.g. [B,T])

            # Validate scaling_dim against original_targets_for_scaling
            if not (0 <= self.scaling_dim < original_targets_for_scaling.ndim):
                raise ValueError(
                    f"Invalid scaling_dim {self.scaling_dim} for original_targets_for_scaling shape {original_targets_for_scaling.shape}."
                )

            # For 'std' and 'minmax', the dimension size must be > 1
            can_scale = original_targets_for_scaling.size(self.scaling_dim) > 1
            scaling_factor = None

            if can_scale:
                if self.scaling_type == "std":
                    std_dev = torch.std(original_targets_for_scaling, dim=self.scaling_dim, keepdim=True, unbiased=False)
                    scaling_factor = std_dev
                elif self.scaling_type == "minmax":
                    min_val = torch.min(original_targets_for_scaling, dim=self.scaling_dim, keepdim=True).values
                    max_val = torch.max(original_targets_for_scaling, dim=self.scaling_dim, keepdim=True).values
                    scaling_factor = max_val - min_val
            elif self.scaling_type != "none":
                 # If scaling was intended but cannot be performed (e.g., dim size 1 for std)
                 # We simply don't scale, could also issue a warning.
                 # print(f"Warning: Scaling type '{self.scaling_type}' skipped because dimension {self.scaling_dim} has size <= 1.")
                 pass


            if scaling_factor is not None:
                scaling_factor = scaling_factor + self.scaling_eps
                # Ensure scaling_factor is broadcastable with elementwise_crps.
                # elementwise_crps shape is typically [B, T] or [B, other_dims...].
                # original_targets_for_scaling could be [B, T, F...].
                # torch.std/min/max with keepdim=True preserves rank, so broadcasting should align
                # if scaling_dim correctly targets a shared dimension.
                # E.g., elementwise_crps [B,T], original_targets_for_scaling [B,T,1].
                # If scaling_dim=1 (T dim), std_dev is [B,1,1], which broadcasts to [B,T].
                elementwise_crps = elementwise_crps / scaling_factor
        # --- End Apply Scaling ---

        # Apply mask (if any) and reduction using the helper method
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
