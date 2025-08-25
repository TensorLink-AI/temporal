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
    Continuous Ranked Probability Score (CRPS) for ensemble/quantile forecasts.

    Accepted `preds` inputs:
      • Tensor: [B,T,S] or [B,T,F,S]  (S = ensemble/paths/quantiles)
      • Dict: one of {"paths","samples","members","quantiles"} -> Tensor
      • ForecastBundle-like: uses .quantiles (Tensor [B,T,F,Q]) if present,
        else .params["paths"] when available.

    Targets:
      • [B,T] (univariate) or [B,T,F] (multivariate). We broadcast F=1 as needed.

    Internally we normalize predictions to [B,T,F,S] with S as the last dim, sort
    along S, and call `crps_ensemble(..., axis=-1, reduce=False)`. The output
    elementwise CRPS is [B,T,F], then masking/scaling/reduction are applied.
    """

    def __init__(
        self,
        reduction: str = "mean",
        estimator: str = "pwm",             # "pwm" | "nrg" | "fair"
        axis: int = -1,                     # user-specified ensemble axis (respected then moved to last)
        scaling_type: str = "none",         # "none" | "std" | "minmax"
        scaling_dim: int = 1,
        scaling_eps: float = 1e-8,
        spread_lambda: float = 0.0,
        spread_penalty_type: str = "symmetric_log",  # "log" | "inverse" | "symmetric_log"
        spread_penalty_epsilon: float = 1e-3,
        spread_target_spread: float = 0.0,
        **kwargs,
    ):
        super().__init__(reduction=reduction)

        if estimator not in {"pwm", "nrg", "fair"}:
            raise ValueError(f"Invalid estimator '{estimator}'.")
        if scaling_type not in {"none", "std", "minmax"}:
            raise ValueError(f"Invalid scaling_type '{scaling_type}'.")
        if spread_penalty_type not in {"log", "inverse", "symmetric_log"}:
            raise ValueError(f"Invalid spread_penalty_type '{spread_penalty_type}'.")

        self.estimator = estimator
        self.user_axis = axis
        self.scaling_type = scaling_type
        self.scaling_dim = scaling_dim
        self.scaling_eps = scaling_eps
        self.spread_lambda = float(spread_lambda)

        self.spread_penalty_fn = None
        if self.spread_lambda > 0.0:
            self.spread_penalty_fn = SpreadPenalty(
                penalty_type=spread_penalty_type,
                epsilon=spread_penalty_epsilon,
                target_spread=spread_target_spread,
                reduction="none",
            )

    # ---------------- shape / extraction helpers ----------------

    @staticmethod
    def _choose_dict_key(d: Dict[str, torch.Tensor]) -> str:
        # preference order
        for k in ("paths", "samples", "members", "quantiles"):
            if k in d and torch.is_tensor(d[k]):
                return k
        # fallback to first tensor value
        for k, v in d.items():
            if torch.is_tensor(v):
                return k
        raise KeyError("No tensor field found in preds dict (expected 'paths'/'samples'/'members'/'quantiles').")

    @staticmethod
    def _as_btfs(x: torch.Tensor, axis: int) -> torch.Tensor:
        """
        Normalize to [B,T,F,S], moving ensemble axis to last.
        Accepts [B,T,S], [B,S,T], [B,T,F,S], [B,F,T,S], etc.
        We don't try to guess which dim is which beyond using `axis` as ensemble;
        we preserve order of non-ensemble dims and append a singleton F if needed.
        """
        if x.ndim < 3:
            raise ValueError(f"preds tensor must be 3D or 4D, got {tuple(x.shape)}")

        # Resolve axis in current ndims
        axis = axis if axis >= 0 else x.ndim + axis
        if not (0 <= axis < x.ndim):
            raise ValueError(f"axis={axis} out of bounds for tensor with ndim={x.ndim}")

        # Move ensemble to last
        if axis != x.ndim - 1:
            perm = [i for i in range(x.ndim) if i != axis] + [axis]
            x = x.permute(*perm)

        # Now ensemble is last; we want [B,T,F,S].
        if x.ndim == 3:
            # Treat as [B,T,S] → add F=1 → [B,T,1,S]
            B, T, S = x.shape
            x = x.view(B, T, 1, S)
            return x

        if x.ndim == 4:
            # We must decide which is T and which is F among the first 3 dims.
            # Convention: assume dims are [B, T, F, S] already if second dim > 1.
            # If the third dim is clearly time (common pattern [B,F,T,S]), swap.
            B, D1, D2, S = x.shape
            # Heuristic: if D2 > D1, likely [B,F,T,S] → swap D1 and D2
            if D2 > D1:
                x = x.permute(0, 2, 1, 3)  # [B,T,F,S]
            # otherwise leave as-is, assuming [B,T,F,S]
            return x

        # For ndim > 4 we don't support (unlikely for time series heads)
        raise ValueError(f"Unsupported preds tensor shape {tuple(x.shape)}; expected 3D or 4D.")

    @staticmethod
    def _targets_to_btf(targets: torch.Tensor, F: int, T: int) -> torch.Tensor:
        """
        Normalize targets to [B,T,F].
        Accepts [B,T], [B,T,F], [B,F] with T==1. Broadcast F=1 if needed.
        """
        if targets.ndim == 3:
            if targets.size(1) != T:
                raise ValueError(f"Targets time dim {targets.size(1)} != T={T}.")
            if targets.size(2) == F:
                return targets
            if targets.size(2) == 1 and F > 1:
                return targets.expand(-1, -1, F)
            raise ValueError(f"Targets feature dim {targets.size(2)} incompatible with F={F}.")

        if targets.ndim == 2:  # [B,T] (univariate)
            if targets.size(1) != T:
                raise ValueError(f"Targets time dim {targets.size(1)} != T={T}.")
            if F == 1:
                return targets.unsqueeze(-1)  # [B,T,1]
            # broadcast scalar per-time target across features if caller really has F>1
            return targets.unsqueeze(-1).expand(-1, -1, F)

        if targets.ndim == 2 and T == 1:  # [B,F] with T==1 (rare)
            if targets.size(1) == F:
                return targets.unsqueeze(1)
            if targets.size(1) == 1 and F > 1:
                return targets.unsqueeze(1).expand(-1, 1, F)

        raise ValueError(f"Incompatible targets shape {tuple(targets.shape)}; expected [B,T] or [B,T,F].")

    def _extract_forecasts_tensor(
        self,
        preds: Union[torch.Tensor, Dict[str, torch.Tensor], object],
        axis: int,
    ) -> torch.Tensor:
        """
        Returns a forecasts tensor normalized to [B,T,F,S], ensemble last.
        Supports Tensor, dict (common DistPred), or ForecastBundle-like object.
        """
        # Tensor directly
        if torch.is_tensor(preds):
            return self._as_btfs(preds, axis)

        # Dict (DistPred / custom)
        if isinstance(preds, dict):
            key = self._choose_dict_key(preds)
            if not torch.is_tensor(preds[key]):
                raise TypeError(f"preds['{key}'] must be a Tensor.")
            return self._as_btfs(preds[key], axis)

        # ForecastBundle-like duck typing
        # Prefer quantiles if present (Tensor [B,T,F,Q])
        if hasattr(preds, "quantiles") and torch.is_tensor(getattr(preds, "quantiles")):
            q = preds.quantiles  # [B,T,F,Q] or [B,T,Q]
            if q.ndim == 3:      # [B,T,Q] → [B,T,1,Q]
                q = q.unsqueeze(2)
            # quantile axis already last; treat as [B,T,F,S]
            return q

        # Else try params["paths"]
        if hasattr(preds, "params"):
            params = getattr(preds, "params")
            if isinstance(params, dict):
                for k in ("paths", "samples", "members", "quantiles"):
                    if k in params and torch.is_tensor(params[k]):
                        return self._as_btfs(params[k], axis)

        raise TypeError(
            "CRPSLoss could not extract forecasts: pass a Tensor [B,T,S]/[B,T,F,S], "
            "a dict with a tensor under 'paths'/'samples'/'members'/'quantiles', "
            "or a ForecastBundle-like object with .quantiles tensor or .params['paths']."
        )

    # ---------------- forward ----------------

    def forward(
        self,
        preds: Union[torch.Tensor, Dict[str, torch.Tensor], object],
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute CRPS with optional spread penalty, scaling, and reduction.

        Returns
        -------
        torch.Tensor
            Reduced loss per your `reduction` setting.
        """
        # 1) Extract and normalize forecasts to [B,T,F,S] with S last.
        forecasts = self._extract_forecasts_tensor(preds, axis=self.user_axis)  # [B,T,F,S]
        B, T, F, S = forecasts.shape

        # 2) Normalize targets to [B,T,F]
        targets_btf = self._targets_to_btf(targets, F=F, T=T)  # [B,T,F]

        # 3) Sort along ensemble dim (S last)
        forecasts_sorted = torch.sort(forecasts, dim=-1)[0]    # [B,T,F,S]

        # 4) crps_ensemble expects observations with singleton on ensemble axis
        obs = targets_btf.unsqueeze(-1)                        # [B,T,F,1]
        elementwise_crps = crps_ensemble(
            observations=obs,
            forecasts=forecasts_sorted,
            estimator=self.estimator,
            axis=-1,
            reduce=False,
        )  # -> [B,T,F]

        # 5) Optional spread penalty (expects ensemble last; handles 3D/4D)
        if self.spread_lambda > 0.0 and self.spread_penalty_fn is not None:
            spread_penalty_map = self.spread_penalty_fn(forecasts_sorted)  # [B,T,F]
            elementwise_crps = elementwise_crps + self.spread_lambda * spread_penalty_map

        # 6) Optional scaling on targets
        if self.scaling_type != "none":
            dim_size = targets_btf.size(self.scaling_dim)
            if dim_size > 1:
                if self.scaling_type == "std":
                    factor = torch.std(
                        targets_btf, dim=self.scaling_dim, keepdim=True, unbiased=False
                    )
                else:  # "minmax"
                    mn = torch.min(targets_btf, dim=self.scaling_dim, keepdim=True).values
                    mx = torch.max(targets_btf, dim=self.scaling_dim, keepdim=True).values
                    factor = mx - mn
                elementwise_crps = elementwise_crps / (factor + self.scaling_eps)

        # 7) Final mask + reduction (BaseLoss handles broadcasting [B,T] / [B,T,F])
        return self._apply_reduction(elementwise_crps, loss_mask)



import torch.nn.functional as Fnn

@register_module("loss", "nll")
class NegativeLogLikelihoodLoss(BaseLoss):
    """
    NLL for probabilistic heads.

    Supports:
      - "gaussian"   : preds concat [mu, log_sigma]   -> [B,T,2C] or [B,2C]
      - "student_t"  : preds concat [mu, log_scale, log_df] -> [B,T,3C] or [B,3C]
      - "mixture"    : dict from MixtureOutputHead (univariate); mixture NLL computed inline

    targets: [B,T,C] or [B,T] (univariate) or [B,C] with T==1
    loss_mask: [B,T] or [B,T,C] (broadcasted in BaseLoss._apply_reduction)
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
        sigma_floor: float   = 1e-4,
        df_floor: float      = 1.001,

        # Mixture numeric floors & fixed-sigma option
        mix_min_sigma: float = 1e-6,
        mix_min_scale: float = 1e-6,
        mix_min_df: float    = 1.001,
        mix_lognorm_min_y: float = 1e-9,
        mix_nb_min_r: float = 1e-6,
        mix_nb_eps_p: float = 1e-6,
        mix_fixed_sigma: float = 1e-3,

        **kwargs,
    ):
        super().__init__(reduction=reduction)
        self.distribution_type = distribution_type.lower().strip()

        # Gaussian
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)

        # Student-t
        self.min_log_scale = float(min_log_scale)
        self.max_log_scale = float(max_log_scale)
        self.min_log_df    = float(min_log_df)
        self.max_log_df    = float(max_log_df)
        self.sigma_floor   = float(sigma_floor)
        self.df_floor      = float(df_floor)

        # Mixture numeric options
        self.mix_min_sigma = float(mix_min_sigma)
        self.mix_min_scale = float(mix_min_scale)
        self.mix_min_df    = float(mix_min_df)
        self.mix_lognorm_min_y = float(mix_lognorm_min_y)
        self.mix_nb_min_r  = float(mix_nb_min_r)
        self.mix_nb_eps_p  = float(mix_nb_eps_p)
        self.mix_fixed_sigma = float(mix_fixed_sigma)

        if self.distribution_type not in {"gaussian", "student_t", "mixture"}:
            raise ValueError(
                f"Unsupported distribution_type={distribution_type!r}. "
                "Choose from: 'gaussian', 'student_t', 'mixture'."
            )

    # -------------------- shape helpers (Gaussian/Student-t) --------------------

    @staticmethod
    def _ensure_btCx2(preds: torch.Tensor) -> torch.Tensor:
        # [B,2C] or [B,T,2C] -> [B,T,C,2]
        if preds.ndim == 2:
            B, twoC = preds.shape
            if twoC % 2 != 0:
                raise ValueError(f"Gaussian preds last dim must be 2C, got {twoC}.")
            C = twoC // 2
            return preds.unsqueeze(1).view(B, 1, C, 2)
        if preds.ndim == 3:
            B, T, twoC = preds.shape
            if twoC % 2 != 0:
                raise ValueError(f"Gaussian preds last dim must be 2C, got {twoC}.")
            C = twoC // 2
            return preds.view(B, T, C, 2)
        raise ValueError(f"Gaussian preds must be 2D or 3D, got shape {tuple(preds.shape)}")

    @staticmethod
    def _ensure_btCx3(preds: torch.Tensor) -> torch.Tensor:
        # [B,3C] or [B,T,3C] -> [B,T, C, 3]
        if preds.ndim == 2:
            B, threeC = preds.shape
            if threeC % 3 != 0:
                raise ValueError(f"Student-t preds last dim must be 3C, got {threeC}.")
            C = threeC // 3
            return preds.unsqueeze(1).view(B, 1, C, 3)
        if preds.ndim == 3:
            B, T, threeC = preds.shape
            if threeC % 3 != 0:
                raise ValueError(f"Student-t preds last dim must be 3C, got {threeC}.")
            C = threeC // 3
            return preds.view(B, T, C, 3)
        raise ValueError(f"Student-t preds must be 2D or 3D, got shape {tuple(preds.shape)}")

    @staticmethod
    def _match_target_shape(targets: torch.Tensor, *, C: int, T: int) -> torch.Tensor:
        # Normalize to [B,T,C]
        if targets.ndim == 3:
            if targets.size(-1) != C:
                raise ValueError(f"Targets feature dim {targets.size(-1)} != C={C}.")
            if targets.size(1) != T and T != 1:
                raise ValueError(f"Targets time dim {targets.size(1)} != T={T}.")
            return targets
        if targets.ndim == 2:
            B, dim = targets.shape
            if C == 1:
                if dim == T:  # [B,T] -> [B,T,1]
                    return targets.unsqueeze(-1)
                if dim == 1:  # [B,1] -> [B,1,1]
                    return targets.unsqueeze(-1)
                raise ValueError(f"Univariate target expected [B,T] or [B,1], got [B,{dim}]")
            if dim == C and T == 1:  # [B,C] with T==1
                return targets.unsqueeze(1)
            raise ValueError(f"Incompatible targets shape {tuple(targets.shape)} for C={C}, T={T}.")
        raise ValueError(f"Incompatible targets shape {tuple(targets.shape)}; "
                         f"expected [B,T,C] or [B,T] or [B,C] with T==1.")

    # -------------------- mixture helpers --------------------

    _KNOWN_MIX = {"gaussian", "fixed_gaussian", "student_t", "log_normal", "neg_binomial"}
    _ALIASES = {
        "normal": "gaussian",
        "fixed_normal": "fixed_gaussian",
        "gauss": "gaussian",
        "studentt": "student_t",
        "student-t": "student_t",
        "lognormal": "log_normal",
        "log-norm": "log_normal",
        "negative_binomial": "neg_binomial",
        "negativebinomial": "neg_binomial",
        "nb": "neg_binomial",
    }

    @classmethod
    def _canon(cls, names: List[str]) -> List[str]:
        out = []
        for n in names:
            key = n.strip().lower().replace(" ", "").replace("-", "_")
            out.append(cls._ALIASES.get(key, key))
        return out

    @staticmethod
    def _btm(logits: torch.Tensor, T: int, M: int) -> torch.Tensor:
        # [B,M] or [B,1,M] or [B,T,M] -> [B,T,M]
        if logits.ndim == 2:   # [B,M]
            return logits.unsqueeze(1).expand(-1, T, -1)
        if logits.ndim == 3:
            return logits if logits.size(1) == T else logits.expand(-1, T, -1)
        raise ValueError(f"mixture_logits must be [B,M] or [B,T,M], got {tuple(logits.shape)}")

    @staticmethod
    def _bt1(x: torch.Tensor, T: int) -> torch.Tensor:
        # [B], [B,1], [B,T] -> [B,T]
        if x.ndim == 1:
            return x.view(x.size(0), 1).expand(-1, T)
        if x.ndim == 2:
            return x if x.size(1) == T else x.expand(-1, T)
        raise ValueError(f"Param must be [B], [B,1], or [B,T], got {tuple(x.shape)} for T={T}")

    @staticmethod
    def _targets_bt(y: torch.Tensor, T: int) -> torch.Tensor:
        # [B,T], [B,T,1], [B], [B,1] -> [B,T]
        if y.ndim == 2 and y.size(1) == T:
            return y
        if y.ndim == 3 and y.size(-1) == 1 and y.size(1) == T:
            return y.squeeze(-1)
        if y.ndim == 2 and y.size(1) == 1:
            return y.expand(-1, T)
        if y.ndim == 1:
            return y.view(y.size(0), 1).expand(-1, T)
        raise ValueError(f"targets must be [B,T], [B,T,1], [B], or [B,1]; got {tuple(y.shape)}")

    # log-prob pieces for mixture
    def _log_prob_gaussian(self, y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(self.mix_min_sigma)
        z = (y - mu) / sigma
        return -0.5 * z * z - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)

    def _log_prob_student_t(self, y: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        scale = scale.clamp_min(self.mix_min_scale)
        nu = df.clamp_min(self.mix_min_df)
        z2 = ((y - mu) / scale) ** 2
        log_base = (
            torch.lgamma((nu + 1.0) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * (torch.log(nu) + math.log(math.pi))
            - torch.log(scale)
        )
        return log_base - 0.5 * (nu + 1.0) * torch.log1p(z2 / nu)

    def _log_prob_lognormal(self, y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(self.mix_min_sigma)
        y_pos = y.clamp_min(self.mix_lognorm_min_y)
        z = (torch.log(y_pos) - mu) / sigma
        return -0.5 * z * z - torch.log(y_pos) - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)

    def _log_prob_negbin(self, y: torch.Tensor, r: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        r = r.clamp_min(self.mix_nb_min_r)
        p = p.clamp(self.mix_nb_eps_p, 1.0 - self.mix_nb_eps_p)
        return (
            torch.lgamma(y + r)
            - torch.lgamma(r)
            - torch.lgamma(y + 1.0)
            + r * torch.log(p)
            + y * torch.log1p(-p)
        )

    # -------------------- forward --------------------

    def forward(
        self,
        preds: Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[str]]]],
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # ---- Gaussian ----
        if self.distribution_type == "gaussian":
            p = self._ensure_btCx2(preds)  # [B,T,C,2]
            mu        = p[..., 0]
            log_sigma = p[..., 1].clamp(min=self.min_log_sigma, max=self.max_log_sigma)
            B, T, C = mu.shape

            y = self._match_target_shape(targets, C=C, T=T)  # [B,T,C]
            inv_sigma_sq = torch.exp(-2.0 * log_sigma)
            sq_err = (y - mu) ** 2

            elementwise_nll = (
                0.5 * sq_err * inv_sigma_sq
                + log_sigma
                + 0.5 * math.log(2.0 * math.pi)
            )  # [B,T,C]
            return self._apply_reduction(elementwise_nll, loss_mask)

        # ---- Student-t ----
        if self.distribution_type == "student_t":
            p = self._ensure_btCx3(preds)  # [B,T,C,3]
            mu        = p[..., 0]
            log_scale = p[..., 1].clamp(min=self.min_log_scale, max=self.max_log_scale)
            log_df    = p[..., 2].clamp(min=self.min_log_df,    max=self.max_log_df)
            B, T, C = mu.shape

            y = self._match_target_shape(targets, C=C, T=T)   # [B,T,C]
            scale = torch.exp(log_scale).clamp_min(self.sigma_floor)
            nu    = torch.exp(log_df) + self.df_floor

            z2 = ((y - mu) / scale) ** 2
            log_base = (
                torch.lgamma((nu + 1.0) / 2.0)
                - torch.lgamma(nu / 2.0)
                - 0.5 * (torch.log(nu) + math.log(math.pi))
                - torch.log(scale)
            )
            log_prob = log_base - 0.5 * (nu + 1.0) * torch.log1p(z2 / nu)
            elementwise_nll = -log_prob  # [B,T,C]
            return self._apply_reduction(elementwise_nll, loss_mask)

        # ---- Mixture (MDN, univariate) ----
        if not isinstance(preds, dict):
            raise TypeError("For 'mixture', preds must be a Dict from MixtureOutputHead.")

        if "components" not in preds or "mixture_logits" not in preds:
            raise ValueError("Mixture preds must contain 'components' and 'mixture_logits'.")

        components_raw = preds["components"]
        if not isinstance(components_raw, (list, tuple)) or len(components_raw) == 0:
            raise ValueError("'components' must be a non-empty list.")
        comps = self._canon(list(components_raw))
        for c in comps:
            if c not in self._KNOWN_MIX:
                raise ValueError(f"Unknown distribution component: {c}")

        logits = preds["mixture_logits"]
        if logits.ndim == 2:   # [B,M]
            B, M = logits.shape
            T = targets.size(1) if targets.ndim >= 2 else 1
        elif logits.ndim == 3: # [B,T,M]
            B, T, M = logits.shape
        else:
            raise ValueError(f"mixture_logits must be [B,M] or [B,T,M], got {tuple(logits.shape)}")

        # normalize shapes
        log_w = Fnn.log_softmax(self._btm(logits, T=T, M=M), dim=-1)  # [B,T,M]
        y = self._targets_bt(targets, T=T)                             # [B,T]

        # helper to fetch either 'gaussian_*' or legacy 'normal_*'
        def need(key: str, alt: Optional[str] = None) -> torch.Tensor:
            if key in preds:
                return preds[key]
            if alt is not None and alt in preds:
                return preds[alt]
            raise KeyError(f"Missing required key '{key}'" + (f" (or '{alt}')" if alt else "") + " in mixture preds.")

        # accumulate per-component log-probs
        log_probs: List[torch.Tensor] = []
        for j, cname in enumerate(comps):
            if cname == "gaussian":
                mu = self._bt1(need("gaussian_mu", "normal_mu"), T)
                sigma = self._bt1(need("gaussian_sigma", "normal_sigma"), T)
                lp = self._log_prob_gaussian(y, mu, sigma)

            elif cname == "fixed_gaussian":
                mu = self._bt1(need("gaussian_mu", "normal_mu"), T)
                sigma = torch.full_like(mu, self.mix_fixed_sigma)
                lp = self._log_prob_gaussian(y, mu, sigma)

            elif cname == "student_t":
                mu    = self._bt1(need("student_mu"), T)
                scale = self._bt1(need("student_scale"), T)
                df    = self._bt1(need("student_df"), T)
                lp = self._log_prob_student_t(y, mu, scale, df)

            elif cname == "log_normal":
                mu    = self._bt1(need("lognorm_mu"), T)
                sigma = self._bt1(need("lognorm_sigma"), T)
                lp = self._log_prob_lognormal(y, mu, sigma)

            elif cname == "neg_binomial":
                r = self._bt1(need("nb_r"), T)
                p = self._bt1(need("nb_p"), T)
                lp = self._log_prob_negbin(y, r, p)

            else:
                raise ValueError(f"Unknown distribution component: {cname}")

            log_probs.append(lp + log_w[..., j])  # [B,T]

        # mixture log-likelihood via log-sum-exp
        stacked = torch.stack(log_probs, dim=-1)     # [B,T,M]
        log_mix = torch.logsumexp(stacked, dim=-1)   # [B,T]
        nll = -log_mix                                # [B,T]

        # make it [B,T,1] so _apply_reduction can broadcast masks like [B,T,C]
        elementwise = nll.unsqueeze(-1)               # [B,T,1]
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
