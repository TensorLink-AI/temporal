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
        quantiles: Optional[List[float]]= None,
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


import math
import torch
from typing import Optional, Union, Dict, List


@register_module("loss", "nll")
class NegativeLogLikelihoodLoss(BaseLoss):
    """
    Negative Log Likelihood (NLL) for probabilistic heads.

    Supports
    --------
    - distribution_type="gaussian":
        preds: concatenated [mu, log_sigma] with shape [B,T,2F] or [B,2F]
        targets: [B,T,F] or [B,T] (univariate) or [B,F] with T==1
    - distribution_type="student_t":
        preds: concatenated [mu, log_scale, log_df] with shape [B,T,3F] or [B,3F]
        targets: as above
    - distribution_type="mixture":
        preds: Dict returned by MixtureOutputHead (univariate MDN)
               This class delegates elementwise NLL to MixtureLoss(reduction="none"),
               then applies the usual mask+reduction here.

    Masking & Reduction
    -------------------
    - loss_mask may be [B,T] or [B,T,F]; broadcasting is handled by BaseLoss._apply_reduction.
    - final reduction is controlled by `reduction` ("mean" | "sum" | "none").
    """

    def __init__(
        self,
        distribution_type: str,
        reduction: str = "mean",

        # Gaussian clamps
        min_log_sigma: float = -20.0,
        max_log_sigma: float =  20.0,

        # Student-t clamps/floors
        min_log_scale: float = -7.0,
        max_log_scale: float =  5.0,
        min_log_df: float    = -2.0,
        max_log_df: float    =  6.0,
        sigma_floor: float   = 1e-4,   # minimum positive scale after exp()
        df_floor: float      = 1.001,  # ensure mean exists; use >2.0 if you require finite variance

        # Mixture passthrough knobs (kept for compatibility)
        min_df: float = 2.0,
        fixed_sigma: float = 1e-3,

        **kwargs,
    ):
        super().__init__(reduction=reduction)
        self.distribution_type = distribution_type.lower().strip()

        # Gaussian settings
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)

        # Student-t settings
        self.min_log_scale = float(min_log_scale)
        self.max_log_scale = float(max_log_scale)
        self.min_log_df    = float(min_log_df)
        self.max_log_df    = float(max_log_df)
        self.sigma_floor   = float(sigma_floor)
        self.df_floor      = float(df_floor)

        # Mixture loss delegate (elementwise); we apply reduction here
        self.mixture_loss_fn = None
        if self.distribution_type == "mixture":
            if MixtureLoss is None:
                raise ImportError(
                    "MixtureLoss could not be imported; ensure it is available under "
                    "`temporal.modules.losses.losses.MixtureLoss`."
                )
            self.mixture_loss_fn = MixtureLoss(
                reduction="none",
                min_df=min_df,
                fixed_sigma=fixed_sigma,
            )

        if self.distribution_type not in {"gaussian", "student_t", "mixture"}:
            raise ValueError(
                f"Unsupported distribution_type={distribution_type!r}. "
                "Choose from: 'gaussian', 'student_t', 'mixture'."
            )

    # ------------------------- shape helpers -------------------------

    @staticmethod
    def _ensure_btfx2(preds: torch.Tensor) -> torch.Tensor:
        """
        Gaussian params to [B,T,F,2] from [B,2F] or [B,T,2F].
        """
        if preds.ndim == 2:
            B, twoF = preds.shape
            if twoF % 2 != 0:
                raise ValueError(f"Gaussian preds last dim must be 2F, got {twoF}.")
            F = twoF // 2
            return preds.unsqueeze(1).view(B, 1, F, 2)
        if preds.ndim == 3:
            B, T, twoF = preds.shape
            if twoF % 2 != 0:
                raise ValueError(f"Gaussian preds last dim must be 2F, got {twoF}.")
            F = twoF // 2
            return preds.view(B, T, F, 2)
        raise ValueError(f"Gaussian preds must be 2D or 3D, got shape {tuple(preds.shape)}")

    @staticmethod
    def _ensure_btfx3(preds: torch.Tensor) -> torch.Tensor:
        """
        Student-t params to [B,T,F,3] from [B,3F] or [B,T,3F].
        """
        if preds.ndim == 2:
            B, threeF = preds.shape
            if threeF % 3 != 0:
                raise ValueError(f"Student-t preds last dim must be 3F, got {threeF}.")
            F = threeF // 3
            return preds.unsqueeze(1).view(B, 1, F, 3)
        if preds.ndim == 3:
            B, T, threeF = preds.shape
            if threeF % 3 != 0:
                raise ValueError(f"Student-t preds last dim must be 3F, got {threeF}.")
            F = threeF // 3
            return preds.view(B, T, F, 3)
        raise ValueError(f"Student-t preds must be 2D or 3D, got shape {tuple(preds.shape)}")

    @staticmethod
    def _match_target_shape(targets: torch.Tensor, *, F: int, T: int) -> torch.Tensor:
        """
        Normalize targets to [B,T,F].

        Accepts:
          - [B,T] (univariate)  -> [B,T,1] if F==1
          - [B,T,F]             -> as-is
          - [B,F] with T==1     -> [B,1,F]
        """
        if targets.ndim == 3:
            if targets.size(-1) != F:
                raise ValueError(f"Targets feature dim {targets.size(-1)} != F={F}.")
            if targets.size(1) != T and T != 1:
                # We allow mismatch only when T==1 (broadcasting single step)
                raise ValueError(f"Targets time dim {targets.size(1)} != T={T}.")
            return targets

        if targets.ndim == 2:
            B, dim = targets.shape
            if F == 1:              # univariate => [B,T] or [B,1]
                if dim == T:        # [B,T] → [B,T,1]
                    return targets.unsqueeze(-1)
                if dim == 1:        # [B,1] → [B,1,1]
                    return targets.unsqueeze(-1)
                raise ValueError(f"Univariate target expected [B,T] or [B,1], got [B,{dim}]")
            # Multivariate: allow [B,F] with T==1
            if dim == F and T == 1:
                return targets.unsqueeze(1)  # [B,1,F]
            raise ValueError(f"Incompatible targets shape {tuple(targets.shape)} for F={F}, T={T}.")

        raise ValueError(f"Incompatible targets shape {tuple(targets.shape)}; expected [B,T,F] or [B,T] or [B,F] with T==1.")

    @staticmethod
    def _squeeze_univariate_bt1(targets: torch.Tensor) -> torch.Tensor:
        """
        For univariate mixture loss: convert [B,T,1] -> [B,T]. If already [B,T], pass through.
        """
        if targets.ndim == 3 and targets.size(-1) == 1:
            return targets.squeeze(-1)
        return targets

    # ------------------------- forward -------------------------

    def forward(
        self,
        preds: Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[str]]]],
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute NLL with mask+reduction.

        Parameters
        ----------
        preds   : Tensor or Dict
            - Gaussian/Student-t: concatenated parameter tensor.
            - Mixture: dict from MixtureOutputHead (univariate).
        targets : Tensor
            Ground-truth values. See helpers for accepted shapes.
        loss_mask : Optional[Tensor]
            Optional [B,T] or [B,T,F] mask; broadcast handled in BaseLoss._apply_reduction.
        """
        if self.distribution_type == "gaussian":
            p = self._ensure_btfx2(preds)  # [B,T,F,2]
            mu        = p[..., 0]
            log_sigma = p[..., 1].clamp(min=self.min_log_sigma, max=self.max_log_sigma)
            B, T, F = mu.shape

            y = self._match_target_shape(targets, F=F, T=T)  # [B,T,F]

            inv_sigma_sq = torch.exp(-2.0 * log_sigma)
            sq_err = (y - mu) ** 2

            elementwise_nll = (
                0.5 * sq_err * inv_sigma_sq
                + log_sigma
                + 0.5 * math.log(2.0 * math.pi)
            )  # [B,T,F]

            return self._apply_reduction(elementwise_nll, loss_mask)

        if self.distribution_type == "student_t":
            p = self._ensure_btfx3(preds)  # [B,T,F,3]
            mu        = p[..., 0]
            log_scale = p[..., 1].clamp(min=self.min_log_scale, max=self.max_log_scale)
            log_df    = p[..., 2].clamp(min=self.min_log_df,    max=self.max_log_df)
            B, T, F = mu.shape

            y = self._match_target_shape(targets, F=F, T=T)   # [B,T,F]

            scale = torch.exp(log_scale).clamp_min(self.sigma_floor)  # [B,T,F]
            nu    = torch.exp(log_df) + self.df_floor                 # [B,T,F]

            # log pdf of Student's t (elementwise)
            z2 = ((y - mu) / scale) ** 2
            log_base = (
                torch.lgamma((nu + 1.0) / 2.0)
                - torch.lgamma(nu / 2.0)
                - 0.5 * (torch.log(nu) + math.log(math.pi))
                - torch.log(scale)
            )
            log_prob = log_base - 0.5 * (nu + 1.0) * torch.log1p(z2 / nu)
            elementwise_nll = -log_prob  # [B,T,F]

            return self._apply_reduction(elementwise_nll, loss_mask)

        # mixture
        if not isinstance(preds, dict):
            raise TypeError("For 'mixture' distribution_type, preds must be a Dict from MixtureOutputHead.")
        if self.mixture_loss_fn is None:
            raise RuntimeError("MixtureLoss not initialized for 'mixture' distribution_type.")

        # MixtureOutputHead is univariate; ensure targets are [B,T], not [B,T,1].
        t = self._squeeze_univariate_bt1(targets)
        # Delegate to MixtureLoss to get elementwise [B,T] (or [B,T,1]) NLL
        elementwise = self.mixture_loss_fn(preds=preds, targets=t, loss_mask=None)

        # If mixture loss returns [B,T,1], make it [B,T,1] consistently for reduction.
        if elementwise.ndim == 2:
            elementwise = elementwise.unsqueeze(-1)  # [B,T] -> [B,T,1]

        return self._apply_reduction(elementwise, loss_mask)

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
