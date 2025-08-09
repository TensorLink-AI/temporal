from typing import Dict, List, Optional, Union
import math
import torch
import torch.nn as nn
from torch.distributions import Normal

from temporal.losses.crps_loss_ensemble import crps_ensemble
from temporal.modules.losses.loss_functions import (
    EnergyDistanceLoss,
    FastSoftDTWLoss,
    KernelEnergyLoss,
    MQLoss,
    MixtureLoss,
    QuantileLoss,
    SpectralLoss,
    SpreadPenalty,
    WeightedQuantileLoss,
)
from temporal.registry.core import register_module


class BaseLoss(nn.Module):
    """
    An abstract base class for time series loss functions.

    This class provides a common interface for all loss modules, including
    standardized handling of reduction ('mean', 'sum', 'none') and optional
    masking of loss values.

    Attributes:
        reduction (str): The type of reduction to apply to the loss.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Initializes the BaseLoss.

        Args:
            reduction (str): The reduction method. Must be one of
                'mean', 'sum', or 'none'.
        """
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction type: {reduction}")
        self.reduction = reduction

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """The forward pass for the loss calculation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method")

    def _apply_reduction(
        self, loss: torch.Tensor, loss_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Applies masking and reduction to a calculated loss tensor.

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
                    mask_expanded_shape = loss_mask.shape + (
                        1,
                    ) * (loss.ndim - loss_mask.ndim)
                    loss_mask = loss_mask.view(mask_expanded_shape).expand_as(loss)
                elif loss_mask.shape == loss.shape[: loss_mask.ndim]:
                    loss_mask = loss_mask.unsqueeze(-1).expand_as(loss)
                else:
                    raise ValueError(
                        f"Loss shape {loss.shape} and mask shape {loss_mask.shape} are incompatible."
                    )

            loss = loss * loss_mask
            if self.reduction == "mean":
                return loss.sum() / loss_mask.sum().clamp(min=1e-9)
            elif self.reduction == "sum":
                return loss.sum()
            else:  # reduction == "none"
                return loss
        else:
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:  # reduction == "none"
                return loss


@register_module("loss", "timeseries_generic")
class TimeSeriesLoss(BaseLoss):
    """
    A generic wrapper for various standard time series loss functions.

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
        **kwargs,
    ):
        """
        Initializes the TimeSeriesLoss.

        Args:
            loss_type (str): The type of loss to use (e.g., 'mse', 'mae', 'mq').
            quantiles (list): A list of quantiles, used for quantile-based losses.
            reduction (str): The reduction method ('mean', 'sum', 'none').
            **kwargs: Catches unused arguments.
        """
        super().__init__(reduction=reduction)
        self.loss_type = loss_type
        self.quantiles = quantiles

        nn_reduction = "none"

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=nn_reduction)
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss(reduction=nn_reduction)
        elif loss_type == "rmse":
            self._mse_for_rmse = nn.MSELoss(reduction="none")
            self.loss_fn = None
        elif loss_type == "quantile":
            self.loss_fn = QuantileLoss(quantile=quantiles[0], reduction=nn_reduction)
        elif loss_type == "mq":
            self.loss_fn = MQLoss(quantiles=quantiles, reduction=nn_reduction)
        elif loss_type == "wql":
            self.loss_fn = WeightedQuantileLoss(
                quantiles=quantiles, reduction=nn_reduction
            )
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

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calculates the loss for the given predictions and targets.

        Args:
            preds (torch.Tensor): The model's predictions.
            targets (torch.Tensor): The ground truth values.
            loss_mask (Optional[torch.Tensor]): An optional mask to apply to
                the loss values.

        Returns:
            torch.Tensor: The final computed loss.
        """
        if self.loss_type in ("quantile", "mq", "wql"):
            if preds.ndim == targets.ndim + 1 and preds.shape[-1] == len(
                self.quantiles
            ):
                targets = targets.unsqueeze(-1)
        elif self.loss_type in ("mse", "mae", "rmse"):
            if preds.shape != targets.shape:
                if preds.ndim == targets.ndim + 1 and preds.shape[-1] == 1:
                    targets = targets.unsqueeze(-1)
                elif preds.ndim + 1 == targets.ndim and targets.shape[-1] == 1:
                    targets = targets.squeeze(-1)
            if preds.shape != targets.shape:
                raise ValueError(
                    f"Shape mismatch for loss '{self.loss_type}': preds {preds.shape}, targets {targets.shape}"
                )

        if self.loss_type == "rmse":
            mse_loss = self._mse_for_rmse(preds, targets)
            if loss_mask is not None:
                if loss_mask.shape != mse_loss.shape:
                    if loss_mask.shape == mse_loss.shape[: loss_mask.ndim]:
                        loss_mask = loss_mask.unsqueeze(-1).expand_as(mse_loss)
                    else:
                        raise ValueError(
                            f"RMSE Loss shape {mse_loss.shape} and mask shape {loss_mask.shape} are incompatible."
                        )
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
                if (
                    elementwise_loss.ndim > targets.ndim
                    and elementwise_loss.shape[-1] == len(self.quantiles)
                ):
                    elementwise_loss = elementwise_loss.mean(dim=-1)
            return self._apply_reduction(elementwise_loss, loss_mask)


@register_module("loss", "crps")
class CRPSLoss(BaseLoss):
    """
    Computes the Continuous Ranked Probability Score (CRPS).

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
        **kwargs,
    ):
        """
        Initializes the CRPSLoss module.

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
                reduction="none",
            )

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the CRPS loss.

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
            reduce=False,
        )

        if self.spread_lambda > 0.0 and self.spread_penalty_fn is not None:
            spread_penalty_map = self.spread_penalty_fn(preds_sorted)
            elementwise_crps = (
                elementwise_crps + self.spread_lambda * spread_penalty_map
            )

        if self.scaling_type != "none":
            dim_size = targets.size(self.scaling_dim)
            if dim_size > 1:
                if self.scaling_type == "std":
                    factor = torch.std(
                        targets, dim=self.scaling_dim, keepdim=True, unbiased=False
                    )
                else:
                    mn = torch.min(targets, dim=self.scaling_dim, keepdim=True).values
                    mx = torch.max(targets, dim=self.scaling_dim, keepdim=True).values
                    factor = mx - mn
                factor = factor + self.scaling_eps
                elementwise_crps = elementwise_crps / factor

        return self._apply_reduction(elementwise_crps, loss_mask)


@register_module("loss", "nll")
class NegativeLogLikelihoodLoss(BaseLoss):
    """
    Computes the Negative Log Likelihood (NLL) loss for probabilistic forecasts.
    This loss function is designed to work with output heads that predict
    parameters of a probability distribution.
    It supports:
    - Gaussian distributions (parameters: mean, log_std).
    - Mixture distributions (delegates to internal MixtureLoss).
    - Other single-component distributions can be added by extending the forward method.
    Attributes:
        distribution_type (str): The type of distribution ("gaussian", "mixture").
        mixture_loss_fn (Optional[MixtureLoss]): An instance of MixtureLoss if
            `distribution_type` is "mixture".
    """

    def __init__(
        self,
        distribution_type: str,
        reduction: str = "mean",
        # Parameters for MixtureLoss, passed through if distribution_type is "mixture"
        min_df: float = 2.0,
        fixed_sigma: float = 1e-3,
        min_log_sigma: float = -20.0,
        max_log_sigma: float = 20.0,
        **kwargs,
    ):
        """
        Initializes the NegativeLogLikelihoodLoss.
        Args:
            distribution_type (str): The type of distribution whose NLL to calculate.
                Supported: "gaussian", "mixture".
            reduction (str): The reduction method ('mean', 'sum', 'none').
            min_df (float): Minimum degrees of freedom for StudentT components in MixtureLoss.
                Only used if `distribution_type` is "mixture".
            fixed_sigma (float): Fixed standard deviation for FixedNormal components in MixtureLoss.
                Only used if `distribution_type` is "mixture".
            min_log_sigma (float): Minimum value for the log of the standard deviation.
            max_log_sigma (float): Maximum value for the log of the standard deviation.
            **kwargs: Catches any additional arguments.
        """
        super().__init__(reduction=reduction)
        self.distribution_type = distribution_type
        self.mixture_loss_fn = None

        if distribution_type == "gaussian":
            self.min_log_sigma = min_log_sigma
            self.max_log_sigma = max_log_sigma
        elif distribution_type == "mixture":
            # Delegate to the existing MixtureLoss for consistency and robust handling of mixtures.
            # Important: The internal MixtureLoss instance should use "none" reduction,
            # so `_apply_reduction` of this NLLLoss can apply the final desired reduction.
            self.mixture_loss_fn = MixtureLoss(
                reduction="none",  # Ensure element-wise NLL from MixtureLoss for our _apply_reduction
                min_df=min_df,
                fixed_sigma=fixed_sigma,
            )
        else:
            raise ValueError(
                f"Unsupported distribution_type for NLLLoss: {distribution_type}. "
                "Supported types are 'gaussian', 'mixture'."
            )

    def forward(
        self,
        preds: Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[str]]]],
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the Negative Log Likelihood loss.
        Args:
            preds (Union[torch.Tensor, Dict]): The model's probabilistic predictions.
                - If `distribution_type` is "gaussian": Tensor of shape `[B, T, F*2]`
                  containing concatenated mean and log_std for each feature.
                - If `distribution_type` is "mixture": Dictionary of parameters
                  from `MixtureOutputHead`.
            targets (torch.Tensor): The ground truth values. Shape `[B, T, F]`
                for multivariate, or `[B, T]` for univariate.
            loss_mask (Optional[torch.Tensor]): An optional mask to apply to
                the loss values. Shape `[B, T]` for univariate targets, or
                `[B, T, F]` for multivariate targets. The mask will be expanded
                to match the element-wise NLL loss shape before reduction.
        Returns:
            torch.Tensor: The final computed NLL loss, reduced according to `self.reduction`.
        """
        elementwise_nll = None

        if self.distribution_type == "gaussian":
            # Validate preds format for Gaussian
            if not isinstance(preds, torch.Tensor) or preds.ndim not in [2, 3]:
                raise TypeError(
                    f"Expected preds to be a Tensor for 'gaussian' distribution_type, but got {type(preds)}"
                )

            # Infer number of features (F) from preds and reshape to [B, T, F, 2]
            if preds.ndim == 2:  # Assumes [Batch, F*2] for a single timestep or feature (e.g., from an output head processing last step)
                num_features = preds.shape[-1] // 2
                preds_reshaped = preds.unsqueeze(1).view(
                    preds.shape[0], 1, num_features, 2
                )
            elif preds.ndim == 3:  # Assumes [Batch, Time, F*2]
                num_features = preds.shape[-1] // 2
                preds_reshaped = preds.view(
                    preds.shape[0], preds.shape[1], num_features, 2
                )
            else:
                raise ValueError(
                    f"Unsupported preds dimension for Gaussian: {preds.ndim}. Expected 2D or 3D."
                )

            # Extract mu and log_sigma
            mu = preds_reshaped[..., 0]
            log_sigma_unclamped = preds_reshaped[..., 1]
            log_sigma = torch.clamp(log_sigma_unclamped, min=self.min_log_sigma, max=self.max_log_sigma)


            # Ensure targets matches the shape of mu/sigma for element-wise log_prob calculation
            # targets needs to be [B, T, F] to match mu/sigma for log_prob.
            if (
                targets.ndim == mu.ndim - 1 and mu.shape[-1] == 1
            ):  # univariate case: targets [B,T], mu [B,T,1]
                targets_expanded = targets.unsqueeze(-1)  # [B, T] -> [B, T, 1]
            elif targets.ndim == mu.ndim:  # multivariate case: targets [B,T,F], mu [B,T,F]
                targets_expanded = targets
            else:  # Handle cases where targets might be [B,F] for a single timestep, or other mismatches
                if (
                    targets.ndim == 2 and mu.ndim == 3 and mu.shape[1] == 1
                ):  # targets [B,F], mu [B,1,F]
                    targets_expanded = targets.unsqueeze(1)  # [B,F] -> [B,1,F]
                else:
                    raise ValueError(
                        f"Target shape {targets.shape} incompatible with Gaussian parameters mu/sigma shape {mu.shape}. "
                        "Expected targets to match features (F) and broadcast across time (T)."
                    )

            inv_sigma_sq = torch.exp(-2 * log_sigma)
            squared_error = (targets_expanded - mu) ** 2
            
            elementwise_nll = (
                0.5 * squared_error * inv_sigma_sq
                + log_sigma
                + 0.5 * math.log(2 * math.pi)
            )

        elif self.distribution_type == "mixture":
            # Validate preds format for Mixture
            if not isinstance(preds, Dict):
                raise TypeError(
                    f"Expected preds to be a Dict for 'mixture' distribution_type, but got {type(preds)}"
                )
            if self.mixture_loss_fn is None:
                raise RuntimeError(
                    "MixtureLoss function not initialized for 'mixture' distribution_type."
                )

            # Delegate to the internal MixtureLoss. It's already configured for reduction="none"
            # so its output will be element-wise NLL.
            # The MixtureLoss expects `targets` to be [B, T] or [B, T, F].
            elementwise_nll = self.mixture_loss_fn(
                preds=preds,
                targets=targets,
                loss_mask=None,  # Pass None here, as our `_apply_reduction` will handle the mask
            )
            # The `elementwise_nll` from MixtureLoss will be of shape [B, T] (univariate) or [B, T, F] (multivariate).

        else:
            # This case should ideally not be reached due to __init__ validation
            raise ValueError(
                f"Internal error: Unhandled distribution_type: {self.distribution_type}"
            )

        # Apply reduction and loss_mask using the BaseLoss's helper
        # `elementwise_nll` is already of shape [B, T] or [B, T, F] (element-wise per sample/time/feature)
        return self._apply_reduction(elementwise_nll, loss_mask)


import math
from typing import Optional



def huber_transform(x: torch.Tensor, delta: float) -> torch.Tensor:
    """
    Huber‐style transform: quadratic for |x|<=delta, linear beyond.
    """
    abs_x = x.abs()
    quad  = 0.5 * abs_x.pow(2)
    lin   = delta * (abs_x - 0.5 * delta)
    return torch.where(abs_x <= delta, quad, lin)

@register_module("loss", "crps_huber")
class CRPSHuberLoss(BaseLoss):
    """
    CRPS Loss with an optional Huber‐style transform.

    Args:
        estimator:    which finite‐ensemble estimator to use ('pwm','nrg','fair')
        axis:         ensemble dimension in preds (default last)
        huber_delta:  if >0, applies a Huber transform with this knee;
                      if 0 or None, no transform (pure CRPS)
        spread_lambda / _type / _eps / _target:
                      exactly as in your original CRPSLoss
        reduction:    one of 'mean','sum','none'
    """
    def __init__(
        self,
        estimator: str = "pwm",
        axis: int = -1,
        huber_delta: Optional[float] = None,
        spread_lambda: float = 0.0,
        spread_penalty_type: str = "symmetric_log",
        spread_penalty_epsilon: float = 1e-3,
        spread_target_spread: float = 0.0,
        reduction: str = "mean",
        **kwargs
    ):
        super().__init__(reduction=reduction)
        if estimator not in ("pwm","nrg","fair"):
            raise ValueError(f"Unknown CRPS estimator '{estimator}'")
        self.estimator = estimator
        self.axis      = axis
        self.huber_delta = float(huber_delta) if huber_delta and huber_delta > 0.0 else None

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
        # ensure targets has the ensemble dim
        if targets.ndim == preds.ndim - 1:
            ax = self.axis if self.axis >= 0 else preds.ndim + self.axis
            targets = targets.unsqueeze(ax)

        # sort along ensemble dim
        preds_sorted = torch.sort(preds, dim=self.axis)[0]

        # raw per‐sample CRPS
        crps_raw = crps_ensemble(
            observations=targets,
            forecasts=preds_sorted,
            estimator=self.estimator,
            axis=self.axis,
            reduce=False
        )

        # optional spread penalty
        if self.spread_lambda > 0.0:
            spread_pen = self.spread_penalty_fn(preds_sorted)  # shape = crps_raw.shape
            crps_raw = crps_raw + self.spread_lambda * spread_pen

        # optional Huber transform
        if self.huber_delta is not None:
            crps_raw = huber_transform(crps_raw, delta=self.huber_delta)

        # final reduction / masking
        return self._apply_reduction(crps_raw, loss_mask)


@register_module("loss", "timeflow")
class TimeFlowLoss(BaseLoss):
    """
    The TimeFlow loss module for temporal forecasting.
    Given target sequences and a conditioning vector z, computes a diffusion-style MSE loss.
    This loss is stateful and contains its own neural network.
    """

    def forward(
        self,
        preds: torch.Tensor, # Expected to be the conditioning vector `z`
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 4) weighted MSE over channels
        weights = 1.0 / torch.arange(
            1, targets.size(-1) + 1,
            device=targets.device, dtype=torch.float32
        )
        err = (preds - targets) ** 2
        
        # This loss assumes channel-last, so weights are applied to the last dim.
        err = weights * err
        loss = err.sum(dim=-1)

        # 5) apply sequence mask (if any) and reduction
        return self._apply_reduction(loss, loss_mask)
