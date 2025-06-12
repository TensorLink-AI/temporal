import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal.modules.losses.loss_functions import (
    QuantileLoss,
    MQLoss,
    WeightedQuantileLoss,
    KernelEnergyLoss,
    EnergyDistanceLoss,
    SpectralLoss,
    FastSoftDTWLoss,
    SpreadPenalty,
    MixtureLoss
)
from temporal.losses.crps_loss_ensemble import crps_ensemble
from temporal.registry.core import register_module
from typing import Optional, Tuple

class BaseLoss(nn.Module):
    """An abstract base class for time series loss functions.

    This class provides a common interface for all loss modules, including
    standardized handling of reduction ('mean', 'sum', 'none') and optional
    masking of loss values.

    Attributes:
        reduction (str): The type of reduction to apply to the loss.
    """
    def __init__(self, reduction: str = "mean"):
        """Initializes the BaseLoss.

        Args:
            reduction (str): The reduction method. Must be one of
                'mean', 'sum', or 'none'.
        """
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction type: {reduction}")
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """The forward pass for the loss calculation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method")

    def _apply_reduction(self, loss: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """Applies masking and reduction to a calculated loss tensor.

        Args:
            loss (torch.Tensor): The raw, unreduced loss tensor.
            loss_mask (Optional[torch.Tensor]): A boolean or binary tensor used
                to mask the loss.

        Returns:
            torch.Tensor: The final loss, either as a scalar (for 'mean' or 'sum')
            or a tensor (for 'none').
        """
        if loss_mask is not None:
            if loss_mask.shape != loss.shape:
                 if loss_mask.ndim == loss.ndim:
                      mask_expanded_shape = loss_mask.shape + (1,) * (loss.ndim - loss_mask.ndim)
                      loss_mask = loss_mask.view(mask_expanded_shape).expand_as(loss)
                 elif loss_mask.shape == loss.shape[:loss_mask.ndim]:
                     loss_mask = loss_mask.unsqueeze(-1).expand_as(loss)
                 else:
                      raise ValueError(f"Loss shape {loss.shape} and mask shape {loss_mask.shape} are incompatible.")

            loss = loss * loss_mask
            if self.reduction == "mean":
                return loss.sum() / loss_mask.sum().clamp(min=1e-9)
            elif self.reduction == "sum":
                return loss.sum()
            else: # reduction == "none"
                return loss
        else:
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else: # reduction == "none"
                return loss


@register_module("loss", "timeseries_generic")
class TimeSeriesLoss(BaseLoss):
    """A generic wrapper for various standard time series loss functions.

    This module acts as a factory and wrapper, allowing for the selection of
    common loss functions like MSE, MAE, and Quantile Loss via a configuration
    string. It handles the instantiation of the appropriate underlying loss
    function and applies it during the forward pass.

    Attributes:
        loss_fn: The underlying instantiated loss function module.
    """
    def __init__(
        self,
        loss_type: str = "mse",
        quantiles: list = [0.1, 0.5, 0.9],
        reduction: str = "mean",
        **kwargs
    ):
        """Initializes the TimeSeriesLoss.

        Args:
            loss_type (str): The type of loss to use (e.g., 'mse', 'mae', 'mq').
            quantiles (list): A list of quantiles, used for quantile-based losses.
            reduction (str): The reduction method ('mean', 'sum', 'none').
            **kwargs: Catches unused arguments.
        """
        super().__init__(reduction=reduction)
        self.loss_type = loss_type
        self.quantiles = quantiles

        nn_reduction = 'none'

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=nn_reduction)
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss(reduction=nn_reduction)
        elif loss_type == "rmse":
             self._mse_for_rmse = nn.MSELoss(reduction='none')
             self.loss_fn = None
        elif loss_type == "quantile":
            self.loss_fn = QuantileLoss(quantile=quantiles[0], reduction=nn_reduction)
        elif loss_type == "mq":
            self.loss_fn = MQLoss(quantiles=quantiles, reduction=nn_reduction)
        elif loss_type == "wql":
            self.loss_fn = WeightedQuantileLoss(quantiles=quantiles, reduction=nn_reduction)
        elif loss_type == "kernel_energy":
            self.loss_fn = KernelEnergyLoss(reduction=nn_reduction)
        elif loss_type == "energy":
            self.loss_fn = EnergyDistanceLoss(reduction=nn_reduction)
        elif loss_type == "spectral":
            self.loss_fn = SpectralLoss(reduction=nn_reduction)
        elif loss_type == "softdtw":
            gamma = kwargs.get("gamma", 1.0)
            self.loss_fn = FastSoftDTWLoss(gamma=gamma, reduction=nn_reduction)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """Calculates the loss for the given predictions and targets.

        Args:
            preds (torch.Tensor): The model's predictions.
            targets (torch.Tensor): The ground truth values.
            loss_mask (Optional[torch.Tensor]): An optional mask to apply to
                the loss values.

        Returns:
            torch.Tensor: The final computed loss.
        """
        if self.loss_type in ("quantile", "mq", "wql"):
            if preds.ndim == targets.ndim + 1 and preds.shape[-1] == len(self.quantiles):
                 targets = targets.unsqueeze(-1)
        elif self.loss_type in ("mse", "mae", "rmse"):
             if preds.shape != targets.shape:
                  if preds.ndim == targets.ndim + 1 and preds.shape[-1] == 1:
                       targets = targets.unsqueeze(-1)
                  elif preds.ndim + 1 == targets.ndim and targets.shape[-1] == 1:
                       targets = targets.squeeze(-1)
             if preds.shape != targets.shape:
                  raise ValueError(f"Shape mismatch for loss '{self.loss_type}': preds {preds.shape}, targets {targets.shape}")

        if self.loss_type == "rmse":
             mse_loss = self._mse_for_rmse(preds, targets)
             if loss_mask is not None:
                  if loss_mask.shape != mse_loss.shape:
                      if loss_mask.shape == mse_loss.shape[:loss_mask.ndim]:
                           loss_mask = loss_mask.unsqueeze(-1).expand_as(mse_loss)
                      else:
                           raise ValueError(f"RMSE Loss shape {mse_loss.shape} and mask shape {loss_mask.shape} are incompatible.")
                  masked_mse_loss = mse_loss * loss_mask
                  mean_masked_mse = masked_mse_loss.sum() / loss_mask.sum().clamp(min=1e-9)
                  loss = torch.sqrt(mean_masked_mse)
                  return loss
             else:
                  mean_mse = mse_loss.mean()
                  loss = torch.sqrt(mean_mse)
                  return loss
        else:
             elementwise_loss = self.loss_fn(preds, targets)
             if self.loss_type in ("mq", "wql"):
                  if elementwise_loss.ndim > targets.ndim and elementwise_loss.shape[-1] == len(self.quantiles):
                      elementwise_loss = elementwise_loss.mean(dim=-1)
             return self._apply_reduction(elementwise_loss, loss_mask)


@register_module("loss", "crps")
class CRPSLoss(BaseLoss):
    """Computes the Continuous Ranked Probability Score (CRPS).

    CRPS is a proper scoring rule that generalizes the Mean Absolute Error (MAE)
    to probabilistic forecasts. It measures the difference between the predicted
    cumulative distribution function (CDF) and the empirical CDF of the target.

    This implementation supports different estimators for CRPS and an optional
    spread penalty to regularize the variance of the forecast distribution.

    Attributes:
        estimator (str): The CRPS estimator ('pwm', 'nrg', 'fair').
        axis (int): The axis representing the ensemble/quantile dimension.
        scaling_type (str): The type of scaling to apply to the loss.
        spread_lambda (float): The coefficient for the spread penalty.
        spread_penalty_fn (Optional[SpreadPenalty]): The spread penalty function
            module, if enabled.
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
        """Initializes the CRPSLoss module.

        Args:
            reduction (str): The final reduction method for the loss.
            estimator (str): The CRPS estimator to use.
            axis (int): The dimension corresponding to the ensemble/quantiles.
            scaling_type (str): The type of scaling to apply to the loss values
                ('none', 'std', 'minmax').
            scaling_dim (int): The dimension along which to compute scaling factors.
            scaling_eps (float): A small epsilon to add for numerical stability
                during scaling.
            spread_lambda (float): The coefficient for the spread penalty. If 0,
                the penalty is disabled.
            spread_penalty_type (str): The type of spread penalty function.
            spread_penalty_epsilon (float): Epsilon for the spread penalty function.
            spread_target_spread (float): A target spread value for the penalty
                function.
            **kwargs: Catches unused arguments.
        """
        super().__init__(reduction=reduction)
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
            self.spread_penalty_fn = SpreadPenalty(
                penalty_type=spread_penalty_type,
                epsilon=spread_penalty_epsilon,
                target_spread=spread_target_spread,
                reduction="none"
            )

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculates the CRPS loss.

        Args:
            preds (torch.Tensor): The predicted ensemble/quantiles, shape
                `[B, T, Q]`.
            targets (torch.Tensor): The ground truth values, shape `[B, T]`.
            loss_mask (Optional[torch.Tensor]): An optional mask to apply to
                the loss values.

        Returns:
            torch.Tensor: The final computed loss.
        """
        if targets.ndim == preds.ndim - 1:
            axis = self.axis if self.axis >= 0 else preds.ndim + self.axis
            targets = targets.unsqueeze(axis)

        preds_sorted = torch.sort(preds, dim=self.axis)[0]

        elementwise_crps = crps_ensemble(
            observations=targets,
            forecasts=preds_sorted,
            estimator=self.estimator,
            axis=self.axis,
            reduce=False
        )

        if self.spread_lambda > 0.0 and self.spread_penalty_fn is not None:
            spread_penalty_map = self.spread_penalty_fn(preds_sorted)
            elementwise_crps = elementwise_crps + self.spread_lambda * spread_penalty_map

        if self.scaling_type != "none":
            dim_size = targets.size(self.scaling_dim)
            if dim_size > 1:
                if self.scaling_type == "std":
                    factor = torch.std(targets, dim=self.scaling_dim, keepdim=True, unbiased=False)
                else:
                    mn = torch.min(targets, dim=self.scaling_dim, keepdim=True).values
                    mx = torch.max(targets, dim=self.scaling_dim, keepdim=True).values
                    factor = mx - mn
                factor = factor + self.scaling_eps
                elementwise_crps = elementwise_crps / factor

        return self._apply_reduction(elementwise_crps, loss_mask)


@register_module("loss", "mixture")
class RegisteredMixtureLoss(BaseLoss):
    """A registered wrapper for the `MixtureLoss` function.

    This module serves as a bridge between the model's configuration system
    and the `MixtureLoss` implementation. It allows `MixtureLoss` to be
    instantiated from a configuration dictionary via the registry. The actual
    loss computation, including reduction and masking, is handled by the
    underlying `MixtureLoss` instance.

    Attributes:
        loss_fn (MixtureLoss): The instantiated `MixtureLoss` object.
    """
    def __init__(self, reduction: str = "mean", min_df: float = 2.0, fixed_sigma: float = 1e-3, **kwargs):
        """Initializes the RegisteredMixtureLoss wrapper.

        Args:
            reduction (str): Specifies the reduction for MixtureLoss: 'none', 'mean', 'sum'.
            min_df (float): Minimum degrees of freedom for StudentT components.
            fixed_sigma (float): Fixed standard deviation for FixedNormal components.
            **kwargs: Catches unused arguments from the config if any.
        """
        super().__init__(reduction=reduction)
        self.loss_fn = MixtureLoss(
            reduction=reduction,
            min_df=min_df,
            fixed_sigma=fixed_sigma
        )

    def forward(self, preds: dict, targets: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """Calculates the Mixture Negative Log-Likelihood loss.

        Args:
            preds (dict): A dictionary of predictions from the `MixtureOutputHead`.
            targets (torch.Tensor): Ground truth values. Shape [B, T].
            loss_mask (Optional[torch.Tensor]): An optional mask for the loss
                elements, shape `[B, T]`.

        Returns:
            torch.Tensor: The final computed mixture loss.
        """
        return self.loss_fn(preds=preds, targets=targets, loss_mask=loss_mask)
