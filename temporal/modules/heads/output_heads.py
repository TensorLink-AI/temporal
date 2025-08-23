import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Union, Callable, Tuple

from temporal.registry.core import register_module
from temporal.modules.heads.base_output_head import BaseOutputHead

# Loss imports kept for context; not used here.
from temporal.modules.losses.losses import TimeSeriesLoss, CRPSLoss
from temporal.modules.losses.loss_functions import QuantileLoss

# =========================
# Shared sampling helpers
# =========================

def _apply_top_p(probs: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return probs
    sorted_probs, idx = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum <= top_p
    mask[..., 0] = True
    filtered = torch.zeros_like(probs).scatter_(-1, idx, sorted_probs * mask)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return filtered

def _temperature_scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(temperature, 1e-6)

def _temperature_scale_std(std: torch.Tensor, temperature: float) -> torch.Tensor:
    return std * max(temperature, 1e-6)

def _safe_softplus(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0, min_value: float = 1e-4) -> torch.Tensor:
    return F.softplus(x, beta=beta, threshold=threshold) + min_value

def _blend_with_mean(sample: torch.Tensor, mean: torch.Tensor, eta: Optional[float]) -> torch.Tensor:
    if eta is None:
        return sample
    eta = float(max(0.0, min(1.0, eta)))
    return eta * sample + (1.0 - eta) * mean

def _add_jitter(y: torch.Tensor, rel_std: float, min_abs: float) -> torch.Tensor:
    if rel_std <= 0 and min_abs <= 0:
        return y
    scale = torch.maximum(torch.abs(y) * rel_std, torch.tensor(min_abs, device=y.device, dtype=y.dtype))
    return y + torch.randn_like(y) * scale

def _ensure_bt1f(y: torch.Tensor) -> torch.Tensor:
    """
    Ensure output shape is [B, 1, F] (add missing dims if univariate or T not present).
    """
    if y.ndim == 2:  # [B, F] -> [B,1,F]
        return y.unsqueeze(1)
    if y.ndim == 3:
        return y  # [B, T, F] assumed T==1 in AR loop
    # handle [B,] -> [B,1,1]
    if y.ndim == 1:
        return y.view(y.shape[0], 1, 1)
    return y  # leave as-is if already [B,1,F]

# ----- Quantile inverse-CDF utilities -----

def _enforce_monotone_quantiles(q: torch.Tensor) -> torch.Tensor:
    # sort along last dim
    return torch.sort(q, dim=-1).values

def _inverse_cdf_from_quantiles(
    quantiles: torch.Tensor,  # [..., Q]
    u: Optional[torch.Tensor] = None,
    tail_extrapolation: str = "hold",
) -> torch.Tensor:
    *batch, Q = quantiles.shape
    device, dtype = quantiles.device, quantiles.dtype
    if u is None:
        u = torch.rand(*batch, 1, device=device, dtype=dtype)

    probs = torch.linspace(0.5 / Q, 1 - 0.5 / Q, Q, device=device, dtype=dtype)  # [Q]
    # searchsorted expects ascending
    # Expand probs to batch
    probs_b = probs.view(*([1] * len(batch)), Q).expand(*batch, Q)
    idx = torch.searchsorted(probs_b, u).clamp(1, Q - 1)
    gather = lambda t, i: torch.gather(t, -1, i)

    p_lo = gather(probs_b, idx - 1)
    p_hi = gather(probs_b, idx)
    q_lo = gather(quantiles, idx - 1)
    q_hi = gather(quantiles, idx)

    w = (u - p_lo) / (p_hi - p_lo).clamp_min(1e-8)
    x = q_lo + w * (q_hi - q_lo)

    if tail_extrapolation == "linear":
        below = u < probs_b[..., :1]
        above = u > probs_b[..., -1:]
        x_below = quantiles[..., :1] - (probs_b[..., :1] - u) * (quantiles[..., 1:2] - quantiles[..., :1]) / (probs_b[..., 1:2] - probs_b[..., :1]).clamp_min(1e-8)
        x_above = quantiles[..., -1:] + (u - probs_b[..., -1:]) * (quantiles[..., -1:] - quantiles[..., -2:-1]) / (probs_b[..., -1:] - probs_b[..., -2:-1]).clamp_min(1e-8)
        x = torch.where(below, x_below, x)
        x = torch.where(above, x_above, x)

    return x

# =========================
# Linear
# =========================

@register_module("output_head", "linear")
class LinearOutputHead(BaseOutputHead):
    """
    Simple linear projection head for point forecasts.
    """
    def __init__(self, hidden_size: int, output_size: int = 1, **kwargs):
        super().__init__()
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # [B,T,F]

    def sample(
        self,
        y_hat: torch.Tensor,                      # [B,1,F] from forward()
        *,
        temperature: float = 1.2,
        jitter_std: float = 0.05,
        min_abs_jitter: float = 1e-3,
        eta_blend_mean: float = 0.85,
    ) -> torch.Tensor:
        # Add tempered jitter, then blend back to mean
        scale = torch.maximum(torch.abs(y_hat) * jitter_std, torch.tensor(min_abs_jitter, device=y_hat.device, dtype=y_hat.dtype))
        scale = _temperature_scale_std(scale, temperature)
        eps = torch.randn_like(y_hat) * scale
        sample = y_hat + eps
        return _ensure_bt1f(_blend_with_mean(sample, y_hat, eta_blend_mean))

    def get_loss_fn(self) -> Optional[Callable]:
        return None

# =========================
# Gaussian
# =========================

@register_module("output_head", "gaussian")
class GaussianHead(BaseOutputHead):
    """
    Gaussian output head.
    - forward: returns concatenated [mu, log_sigma] with shape [B, T, 2F]
    - predict("mean"|"median"): returns [B, T, F]
    - sample(...): reparam sampling, returns [B, T, F] (or [B, 1, F] if T==1)
    - sample_quantiles(qs): returns [B, T, F, Q]
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int = 1,              # number of features F
        *,
        min_log_sigma: float = -7.0,       # clamp range for stability
        max_log_sigma: float = 5.0,
        sigma_floor: float = 1e-4,         # minimum stdev after exp()
        init_log_sigma: Optional[float] = None,  # optional bias init for log_sigma
        **kwargs,
    ):
        super().__init__()
        self.feature_size = int(output_size)
        self.proj = nn.Linear(hidden_size, 2 * self.feature_size)
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)
        self.sigma_floor = float(sigma_floor)

        # Optional: initialize the log_sigma half with a bias
        if init_log_sigma is not None:
            with torch.no_grad():
                # bias layout: [mu(F), log_sigma(F)]
                if self.proj.bias is None:
                    self.proj.bias = nn.Parameter(torch.zeros(2 * self.feature_size))
                self.proj.bias[self.feature_size:] = float(init_log_sigma)

    # ---- internals ---------------------------------------------------------

    def _split_params(self, y: torch.Tensor):
        """
        y: [B, T, 2F] or [B, 1, 2F]
        returns (mu, log_sigma) both [B, T, F] (or [B, 1, F])
        """
        mu, log_sigma = y.chunk(2, dim=-1)
        # clamp in-place for stability
        log_sigma = log_sigma.clamp_(self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma

    # ---- API ---------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, hidden_size]
        returns: [B, T, 2F]  (concat of mu and log_sigma)
        """
        out = self.proj(x)                  # [B, T, 2F]
        # clamp log_sigma region
        mu, log_sigma = self._split_params(out)
        return torch.cat([mu, log_sigma], dim=-1)

    def predict(self, params: torch.Tensor, method: str = "mean") -> torch.Tensor:
        """
        params: [B, T, 2F] or [B, 1, 2F]
        method: "mean" | "median"
        returns: [B, T, F] (or [B, 1, F])
        """
        mu, log_sigma = self._split_params(params)
        if method not in ("mean", "median"):
            raise ValueError(f"GaussianHead.predict: unsupported method '{method}'. Use 'mean' or 'median'.")
        # For a Gaussian, mean == median
        return mu

    def sample(
        self,
        params: torch.Tensor,                     # [B, T, 2F] or [B, 1, 2F]
        *,
        method: str = "reparam",                  # {"reparam","mean"}
        temperature: float = 1.0,
        min_std: Optional[float] = None,
        eta_blend_mean: float = 0.0,              # 0..1: blend sample towards mean
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Returns a sample with same time shape as params: [B, T, F] or [B, 1, F]
        """
        mu, log_sigma = self._split_params(params)
        if method == "mean" or temperature == 0.0:
            return mu

        sigma = torch.exp(log_sigma)
        sigma = sigma.clamp_min(self.sigma_floor if min_std is None else float(min_std))
        if temperature != 1.0:
            sigma = sigma * float(temperature)

        eps = _randn_like(mu, generator=generator)
        y = mu + eps * sigma
        if eta_blend_mean:
            y = (1.0 - float(eta_blend_mean)) * y + float(eta_blend_mean) * mu
        return y

    def sample_quantiles(self, params: torch.Tensor, quantile_levels: List[float]) -> torch.Tensor:
        """
        params: [B, T, 2F] or [B, 1, 2F]
        returns: [B, T, F, Q]
        """
        mu, log_sigma = self._split_params(params)
        sigma = torch.exp(log_sigma)

        q = torch.tensor(quantile_levels, dtype=mu.dtype, device=mu.device)  # [Q]
        # standard normal quantiles
        z = torch.distributions.Normal(0.0, 1.0).icdf(q)                     # [Q]

        # Broadcast: [B,T,F,1] + [1,1,1,Q] -> [B,T,F,Q]
        return mu.unsqueeze(-1) + sigma.unsqueeze(-1) * z.view(1, 1, 1, -1)

    def get_loss_fn(self) -> Optional[Callable]:
        return None


# =========================
# Quantile Regression
# =========================

@register_module("output_head", "quantile_regression")
class QuantileRegressionOutputHead(BaseOutputHead):
    """
    Multi-quantile regression head.
    """
    def __init__(self, hidden_size: int, output_size: int, num_quantiles: int, feature_size: int = 1, **kwargs):
        super().__init__()
        if output_size != num_quantiles * feature_size:
            raise ValueError(
                f"Output size mismatch: output_size ({output_size}) must equal "
                f"num_quantiles ({num_quantiles}) * feature_size ({feature_size})."
            )
        self.num_quantiles = num_quantiles
        self.feature_size = feature_size
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected_output = self.proj(x)
        if self.feature_size > 1:
            return projected_output.view(*projected_output.shape[:-1], self.feature_size, self.num_quantiles)  # [B,T,F,Q]
        else:
            return projected_output  # [B,T,Q]

    def sample(
        self,
        q_pred: torch.Tensor,                     # [B,1,Q] or [B,1,F,Q]
        *,
        temperature: float = 1.0,
        jitter_u: float = 0.05,
        enforce_monotone: bool = True,
        tail_extrapolation: str = "hold",
        eta_blend_median: float = 0.3,
    ) -> torch.Tensor:
        if q_pred.ndim == 4:  # [B,1,F,Q]
            # inverse-CDF per feature
            B, T, F, Q = q_pred.shape
            q_use = _enforce_monotone_quantiles(q_pred) if enforce_monotone else q_pred
            # u ~ around 0.5 with width ~ jitter_u*temperature
            if jitter_u > 0:
                spread = min(0.5, jitter_u * max(temperature, 1e-6))
                u = (0.5 - spread) + torch.rand(B, T, F, 1, device=q_pred.device, dtype=q_pred.dtype) * (2 * spread)
                u = u.clamp(0.0, 1.0)
            else:
                u = torch.rand(B, T, F, 1, device=q_pred.device, dtype=q_pred.dtype)
            x = _inverse_cdf_from_quantiles(q_use, u=u, tail_extrapolation=tail_extrapolation)  # [B,1,F,1]
            median = q_use[..., Q // 2: Q // 2 + 1]  # [B,1,F,1]
            x = _blend_with_mean(x, median, eta_blend_median)
            return _ensure_bt1f(x.squeeze(-1))  # [B,1,F]
        else:  # [B,1,Q] -> treat as univariate F=1
            B, T, Q = q_pred.shape
            q_use = _enforce_monotone_quantiles(q_pred) if enforce_monotone else q_pred
            if jitter_u > 0:
                spread = min(0.5, jitter_u * max(temperature, 1e-6))
                u = (0.5 - spread) + torch.rand(B, T, 1, device=q_pred.device, dtype=q_pred.dtype) * (2 * spread)
                u = u.clamp(0.0, 1.0)
            else:
                u = torch.rand(B, T, 1, device=q_pred.device, dtype=q_pred.dtype)
            x = _inverse_cdf_from_quantiles(q_use, u=u, tail_extrapolation=tail_extrapolation)  # [B,1,1]
            median = q_use[..., Q // 2: Q // 2 + 1]
            x = _blend_with_mean(x, median, eta_blend_median)
            return _ensure_bt1f(x)  # [B,1,1]

    def predict(self, x: torch.Tensor, method: str = "median") -> torch.Tensor:
        if method != "median":
            raise ValueError("QuantileRegressionOutputHead only supports 'median' prediction.")
        median_idx = self.num_quantiles // 2
        if x.ndim == 4:
            return x[..., median_idx]  # [B,T,F]
        return x[..., median_idx:median_idx + 1]  # [B,T,1]

    def get_loss_fn(self) -> Optional[Callable]:
        return None

# =========================
# DistPred (Path-based)
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Tuple

from temporal.registry.core import register_module
from temporal.modules.heads.base_output_head import BaseOutputHead


@register_module("output_head", "distpred")
class DistPredHead(BaseOutputHead):
    """
    DistPred outputs K candidate paths per feature; we sample a path per step.

    forward(...) returns a dict:
        {
          "paths": [B, T, C, K]   (or [B, T, 1, K] if univariate),
          "path_logits": [B, T, K]   # optional, if with_path_logits=True
        }

    .sample(...) supports sticky or mixture selection and returns:
        (y_next, new_state)
        where y_next is [B, 1, C] and new_state holds sticky path indices.
    """

    # --------------------------- helpers (scoped) ---------------------------

    @staticmethod
    def _apply_top_p(probs: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
        if top_p is None or top_p >= 1.0:
            return probs
        sorted_probs, idx = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum <= top_p
        mask[..., 0] = True
        filtered = torch.zeros_like(probs).scatter_(-1, idx, sorted_probs * mask)
        filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return filtered

    @staticmethod
    def _temperature_scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return logits / max(float(temperature), 1e-6)

    @staticmethod
    def _add_jitter(y: torch.Tensor, rel_std: float, min_abs: float) -> torch.Tensor:
        if rel_std <= 0 and min_abs <= 0:
            return y
        scale = torch.maximum(torch.abs(y) * rel_std, torch.tensor(min_abs, device=y.device, dtype=y.dtype))
        return y + torch.randn_like(y) * scale

    @staticmethod
    def _ensure_bt1f(x: torch.Tensor, feature_size: int) -> torch.Tensor:
        """
        Normalize to [B, 1, C]. Accepts [B,1,C]/[B,1,1]/[B,C].
        """
        if x.ndim == 3:  # [B,1,C]
            if x.size(-1) == 1 and feature_size > 1:
                x = x.expand(x.size(0), x.size(1), feature_size)
            return x
        if x.ndim == 2:  # [B,C] -> [B,1,C]
            x = x.unsqueeze(1)
            if x.size(-1) == 1 and feature_size > 1:
                x = x.expand(x.size(0), x.size(1), feature_size)
            return x
        if x.ndim == 1:  # [B] -> [B,1,1]
            x = x.view(x.shape[0], 1, 1)
            if feature_size > 1:
                x = x.expand(x.size(0), x.size(1), feature_size)
            return x
        raise ValueError(f"Expected feedback tensor with 1–3 dims, got {x.shape}")

    # --------------------------- init / forward -----------------------------

    def __init__(self, hidden_size: int, output_size: int, **kwargs):
        super().__init__()
        if 'num_outputs' not in kwargs:
            raise ValueError("DistPredHead requires 'num_outputs' to be specified in the configuration.")
        self.num_outputs = kwargs['num_outputs']
        self.feature_size = kwargs.get('feature_size', 1)

        expected_output_size = self.num_outputs * self.feature_size
        if output_size != expected_output_size:
            raise ValueError(
                f"DistPredHead output size mismatch: output_size ({output_size}) "
                f"!= num_outputs ({self.num_outputs}) * feature_size ({self.feature_size})"
            )

        self.proj = nn.Linear(hidden_size, output_size)
        self.use_tanh = kwargs.get('use_tanh', False)
        self.tanh_scale = kwargs.get('tanh_scale', 10.0)

        # Optional scoring head for paths
        self.with_path_logits = kwargs.get('with_path_logits', False)
        if self.with_path_logits:
            self.logit_proj = nn.Linear(hidden_size, self.num_outputs)

        # Default state key for sticky path indices
        self.state_key_default = kwargs.get('state_key', "distpred.path_idx")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, T, H] -> returns dict with:
          paths: [B, T, C, K]
          path_logits: [B, T, K] (optional)
        """
        y = self.proj(x)  # [B, T, C*K]
        if self.use_tanh:
            y = self.tanh_scale * torch.tanh(y)
        y = y.view(*y.shape[:-1], self.feature_size, self.num_outputs)  # [B, T, C, K]

        out = {"paths": y}
        if self.with_path_logits:
            out["path_logits"] = self.logit_proj(x)  # [B, T, K]
        return out

    # --------------------------- quantiles / predict -------------------------

    def predict(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]],
                method: Union[str, float, int] = "median") -> torch.Tensor:
        """
        Collapse [B,T,C,K] (or [B,T,K]) over K. Accepts dict from forward().
        Returns [B,T,C] (or [B,T,1] for univariate).
        """
        # Unwrap dicts
        if isinstance(x, dict):
            if "paths" not in x:
                raise TypeError("DistPredHead.predict expected a tensor or a dict with key 'paths'.")
            x = x["paths"]

        is_multi = (x.ndim == 4)  # [B,T,C,K]
        K = x.shape[-1]

        def _reduce_mean(z):    return z.mean(dim=-1)
        def _reduce_median(z):  return torch.median(z, dim=-1).values
        def _reduce_quantile(z, q: float):
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"quantile must be in [0,1], got {q}")
            return torch.quantile(z, q=q, dim=-1, interpolation="linear")

        if isinstance(method, str):
            if method == "mean":
                out = _reduce_mean(x)
            elif method == "median":
                out = _reduce_median(x)
            else:
                raise ValueError(f"Unsupported method string: {method!r}")
        elif isinstance(method, float):
            out = _reduce_quantile(x, method)
        elif isinstance(method, int):
            idx = method if method >= 0 else K + method
            if not (0 <= idx < K):
                raise IndexError(f"Index {method} out of range for K={K}")
            out = x[..., idx]
        else:
            raise TypeError(f"method must be str|float|int, not {type(method)}")

        # ensure [B,T,C]
        if out.ndim == 2:
            out = out.unsqueeze(-1)  # [B,T] -> [B,T,1]
        return out

    def sample_quantiles(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]],
                         quantile_levels: List[float]) -> torch.Tensor:
        """
        Empirical quantiles from ensemble.
        Accepts tensor or dict from forward().
        Returns: [B,T,C,Q] (univariate -> C=1)
        """
        # Unwrap dicts
        if isinstance(x, dict):
            if "paths" not in x:
                raise TypeError("DistPredHead.sample_quantiles expected a tensor or a dict with key 'paths'.")
            x = x["paths"]  # [B,T,C,K] or [B,T,K]

        # Normalize shapes
        if x.ndim == 3:           # [B,T,K] -> [B,T,1,K]
            x = x.unsqueeze(-2)
        elif x.ndim != 4:         # must be [B,T,C,K]
            raise ValueError(f"Unexpected DistPred paths shape: {x.shape}")

        # Sort along K
        sorted_x, _ = torch.sort(x, dim=-1)  # [B,T,C,K]

        q_tensor = torch.tensor(quantile_levels, device=x.device, dtype=x.dtype).view(1, 1, 1, -1)  # [1,1,1,Q]
        K = sorted_x.shape[-1]
        indices = (q_tensor * (K - 1)).round().long()                    # [1,1,1,Q]
        indices = indices.expand(sorted_x.shape[0], sorted_x.shape[1], sorted_x.shape[2], -1)  # [B,T,C,Q]

        return torch.gather(sorted_x, -1, indices)  # [B,T,C,Q]

    # --------------------------- sampling (sticky / mixture) -----------------

    @torch.no_grad()
    def sample(
        self,
        head_out: Dict[str, torch.Tensor],        # output of forward()
        *,
        state: Optional[Dict[str, torch.Tensor]] = None,
        temperature: float = 1.0,
        top_p: Optional[float] = 0.9,
        stickiness: float = 0.9,
        reselection_hazard: Optional[float] = 0.02,
        mode: str = "sticky",                     # {'sticky','mixture'}
        mixture_sharpness: float = 1.0,
        dirichlet_alpha: Optional[float] = None,
        jitter_rel_std: float = 0.0,
        jitter_min_abs: float = 1e-3,
        state_key: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        One AR step. Expects T=1 in head_out tensors.
        Returns:
          y_next: [B,1,C]
          new_state: dict with updated path indices
        """
        paths = head_out["paths"]                 # [B,1,C,K] or [B,1,K]
        if paths.ndim == 3:
            paths = paths.unsqueeze(-2)          # -> [B,1,1,K]
        B, T, C, K = paths.shape                 # <-- use C (channel/feature), not F
        assert T == 1, "DistPredHead.sample expects T=1 during AR decoding."

        device, dtype = paths.device, paths.dtype
        path_logits = head_out.get("path_logits", None)
        if path_logits is not None:
            # [B,1,K] -> [B,K] or handle [B,T,K] with T=1
            if path_logits.ndim == 3 and path_logits.shape[1] == 1:
                path_logits = path_logits.squeeze(1)  # [B,K]
        else:
            path_logits = torch.zeros(B, K, device=device, dtype=dtype)

        scaled = self._temperature_scale_logits(path_logits, temperature)
        probs  = F.softmax(scaled, dim=-1)       # [B,K]
        probs  = self._apply_top_p(probs, top_p)

        state_key = state_key or self.state_key_default
        new_state = {} if state is None else dict(state)
        if state is not None and state_key in state:
            prev_idx = state[state_key].long()   # [B]
            one_hot  = F.one_hot(prev_idx, num_classes=K).to(dtype)
            sticky_prior = stickiness * one_hot + (1.0 - stickiness) * probs
            sticky_prior = sticky_prior / sticky_prior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            if reselection_hazard and reselection_hazard > 0.0:
                hazard = (torch.rand(B, device=device) < reselection_hazard).to(dtype).unsqueeze(-1)
                probs = hazard * probs + (1.0 - hazard) * sticky_prior
            else:
                probs = sticky_prior

        if mode == "mixture":
            if dirichlet_alpha and dirichlet_alpha > 0:
                noise = torch.distributions.Dirichlet(
                    torch.full((K,), dirichlet_alpha, device=device, dtype=dtype)
                ).sample((B,))
                probs = (probs + noise) / (probs + noise).sum(dim=-1, keepdim=True).clamp_min(1e-12)
            if mixture_sharpness != 1.0:
                logits = torch.log(probs.clamp_min(1e-12)) * mixture_sharpness
                probs  = F.softmax(logits, dim=-1)
            # Weighted sum over paths
            w = probs.view(B, 1, 1, K)
            y_next = (w * paths).sum(dim=-1)     # [B,1,C]
            if jitter_rel_std > 0:
                y_next = self._add_jitter(y_next, jitter_rel_std, jitter_min_abs)
            new_state[state_key] = probs.argmax(dim=-1)
            return self._ensure_bt1f(y_next, self.feature_size), new_state

        # Hard (sticky) selection
        cat = torch.distributions.Categorical(probs=probs)
        idx = cat.sample()                        # [B]
        new_state[state_key] = idx
        gather_idx = idx.view(B, 1, 1, 1).expand(B, 1, C, 1)
        y_next = paths.gather(dim=-1, index=gather_idx).squeeze(-1)   # [B,1,C]
        if jitter_rel_std > 0:
            y_next = self._add_jitter(y_next, jitter_rel_std, jitter_min_abs)
        return self._ensure_bt1f(y_next, self.feature_size), new_state


# =========================
# Mixture (MDN)
# =========================

@register_module("output_head", "mixture")
class MixtureOutputHead(BaseOutputHead):
    """
    Mixture Density head (univariate per component).
    Supports components in DIST_PARAM_COUNTS.
    forward(x) -> dict with per-component params and 'mixture_logits' [B,T,M].
    sample(params) -> one-step sample [B,1,1].
    predict(params) -> mixture mean [B,T,1].
    sample_quantiles(params, qs) -> MC quantiles [B,T,1,Q].
    """

    DIST_PARAM_COUNTS = {
        "student_t": {"df": 1, "mu": 1, "scale": 1},
        "log_normal": {"mu": 1, "sigma": 1},
        "neg_binomial": {"r": 1, "p": 1},
        "normal": {"mu": 1, "sigma": 1},
        "fixed_normal": {"mu": 1}
    }
    DIST_OUTPUT_KEYS = {
        "student_t": {"df": "student_df", "mu": "student_mu", "scale": "student_scale"},
        "log_normal": {"mu": "lognorm_mu", "sigma": "lognorm_sigma"},
        "neg_binomial": {"r": "nb_r", "p": "nb_p"},
        "normal": {"mu": "normal_mu", "sigma": "normal_sigma"},
        "fixed_normal": {"mu": "normal_mu"}
    }

    def __init__(self, hidden_size: int, components: List[str], **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.components = components
        self.num_components = len(components)
        if self.num_components <= 0:
            raise ValueError("MixtureOutputHead requires at least one component.")

        total_params_dim = 0
        self.param_indices: Dict[str, Tuple[int, int]] = {}
        current_idx = 0

        for dist_name in self.components:
            if dist_name not in self.DIST_PARAM_COUNTS:
                raise ValueError(f"Unknown distribution component '{dist_name}'. Supported: {list(self.DIST_PARAM_COUNTS.keys())}")
            param_counts_for_dist = self.DIST_PARAM_COUNTS[dist_name]
            output_keys_for_dist = self.DIST_OUTPUT_KEYS[dist_name]
            for param_key, count in param_counts_for_dist.items():
                out_key = output_keys_for_dist[param_key]
                self.param_indices[out_key] = (current_idx, current_idx + count)
                current_idx += count
                total_params_dim += count

        self.mixture_logits_indices = (current_idx, current_idx + self.num_components)
        total_params_dim += self.num_components

        self.output_projection = nn.Linear(hidden_size, total_params_dim)
        if "output_size" in kwargs and kwargs["output_size"] != total_params_dim:
            raise ValueError(
                f"MixtureOutputHead: provided 'output_size' ({kwargs['output_size']}) "
                f"does not match derived total param dim ({total_params_dim})."
            )

    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[str]]]:
        flat = self.output_projection(x)  # [B,T,Dp]
        out: Dict[str, Union[torch.Tensor, List[str]]] = {"components": self.components}
        # mixture logits
        s, e = self.mixture_logits_indices
        out["mixture_logits"] = flat[..., s:e]  # [B,T,M]
        # component params
        processed = set()
        for dist_name in self.components:
            keymap = self.DIST_OUTPUT_KEYS[dist_name]
            for pkey in self.DIST_PARAM_COUNTS[dist_name]:
                out_key = keymap[pkey]
                if out_key in processed:
                    continue
                a, b = self.param_indices[out_key]
                tensor = flat[..., a:b]
                out[out_key] = tensor.squeeze(-1) if tensor.shape[-1] == 1 else tensor
                processed.add(out_key)
        return out

    # ---------- helpers to normalize param shapes to [B,T,1] ----------

    @staticmethod
    def _bt1(t: torch.Tensor) -> torch.Tensor:
        # Accept [B,T], [B,1], [B,T,1], [B] and normalize to [B,T,1]
        if t.ndim == 3 and t.shape[-1] == 1:
            return t  # [B,T,1]
        if t.ndim == 2:
            return t.unsqueeze(-1)  # [B,T,1]
        if t.ndim == 1:
            return t.view(t.shape[0], 1, 1)  # [B,1,1]
        raise ValueError(f"Unexpected tensor shape for param: {t.shape}")

    def _get_params_per_component(self, params: Dict[str, torch.Tensor], min_std: float) -> List[Dict[str, torch.Tensor]]:
        """Return list of per-component param dicts, each tensors [B,T,1]."""
        comps = params["components"]
        out: List[Dict[str, torch.Tensor]] = []
        for cname in comps:
            cp: Dict[str, torch.Tensor] = {}
            if cname == "normal":
                mu = self._bt1(params["normal_mu"])
                sigma = self._bt1(params["normal_sigma"])
                cp["mu"] = mu
                cp["sigma"] = _safe_softplus(sigma, min_value=min_std)
            elif cname == "fixed_normal":
                mu = self._bt1(params["normal_mu"])
                cp["mu"] = mu
                cp["sigma"] = torch.full_like(mu, min_std)
            elif cname == "student_t":
                df = self._bt1(params["student_df"])
                mu = self._bt1(params["student_mu"])
                sc = self._bt1(params["student_scale"])
                cp["df"] = 2.0 + F.softplus(df)                      # ensure df>2 for variance
                cp["mu"] = mu
                cp["scale"] = _safe_softplus(sc, min_value=min_std)
            elif cname == "log_normal":
                mu = self._bt1(params["lognorm_mu"])
                sig = self._bt1(params["lognorm_sigma"])
                cp["mu"] = mu
                cp["sigma"] = _safe_softplus(sig, min_value=min_std)
            elif cname == "neg_binomial":
                r = self._bt1(params["nb_r"])
                p = self._bt1(params["nb_p"]).clamp(1e-6, 1 - 1e-6)
                cp["r"] = _safe_softplus(r, min_value=1e-6)          # total_count > 0
                cp["p"] = p
            else:
                raise NotImplementedError(f"Mixture component '{cname}' not implemented.")
            out.append(cp)
        return out

    # ---------- point prediction (mixture mean) ----------

    def predict(self, params: Dict[str, torch.Tensor], method: str = "mean") -> torch.Tensor:
        if method not in ("mean", "median"):
            raise ValueError(f"Unsupported method: {method}")
        logits = params["mixture_logits"]
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)      # [B,1,M]
        w = torch.softmax(logits, dim=-1)      # [B,T,M]

        comp_params = self._get_params_per_component(params, min_std=1e-4)  # [B,T,1] tensors
        # mixture mean by component type
        means = []
        for cp, cname in zip(comp_params, params["components"]):
            if cname in ("normal", "fixed_normal", "student_t"):
                means.append(cp["mu"])  # [B,T,1]
            elif cname == "log_normal":
                means.append(torch.exp(cp["mu"] + 0.5 * (cp["sigma"]**2)))
            elif cname == "neg_binomial":
                # mean under (r,p) (#failures until r successes with prob p of success) = r*(1-p)/p
                means.append(cp["r"] * (1.0 - cp["p"]) / cp["p"])
        means = torch.stack(means, dim=-1)     # [B,T,1,M]
        y = (w.unsqueeze(-2) * means).sum(dim=-1)  # [B,T,1]
        return y

    # ---------- quantiles via Monte Carlo ----------

    @torch.no_grad()
    def sample_quantiles(
        self,
        params: Dict[str, Union[torch.Tensor, List[str]]],
        quantile_levels: List[float],
        *,
        mc_samples: int = 512,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        mixture_sharpness: float = 1.0,
        dirichlet_alpha: Optional[float] = None,
        min_std: float = 1e-4,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Monte Carlo quantiles from the mixture.
        Args:
          params: dict from forward(); expects 'mixture_logits' and per-component params.
          quantile_levels: list in (0,1)
          mc_samples: number of MC draws per time step
          temperature/top_p/mixture_sharpness/dirichlet_alpha: sampling controls on mixture probs
          min_std: floor for std/scale
          seed: optional RNG seed for reproducibility
        Returns:
          Tensor [B, T, 1, Q]
        """
        q = torch.tensor(quantile_levels, dtype=params["mixture_logits"].dtype, device=params["mixture_logits"].device)
        if not torch.all((q > 0.0) & (q < 1.0)):
            raise ValueError(f"Quantiles must be in (0,1). Got {quantile_levels}")

        logits = params["mixture_logits"]             # [B,T,M] or [B,1,M] or [B,M]
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)              # [B,1,M]
        B, T, M = logits.shape

        # mixture probabilities with controls
        scaled = logits
        if mixture_sharpness != 1.0:
            scaled = scaled * float(mixture_sharpness)
        if temperature != 1.0:
            scaled = _temperature_scale_logits(scaled, temperature)
        w = F.softmax(scaled, dim=-1)                 # [B,T,M]
        if top_p is not None and top_p < 1.0:
            # apply top-p independently at each (B,T)
            w = w.view(B * T, M)
            w = _apply_top_p(w, top_p)
            w = w.view(B, T, M)
        if dirichlet_alpha and dirichlet_alpha > 0:
            alpha = torch.full((B, T, M), float(dirichlet_alpha), device=logits.device, dtype=logits.dtype)
            noise = torch.distributions.Dirichlet(alpha).sample()
            w = (w + noise) / (w + noise).sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # set up RNG
        gen = None
        if seed is not None:
            gen = torch.Generator(device=logits.device).manual_seed(int(seed))

        # per-component parameter tensors [B,T,1]
        comp_params = self._get_params_per_component(params, min_std=min_std)

        # sample component indices: shape [B,T,mc]
        cat = torch.distributions.Categorical(probs=w.view(B * T, M))
        choices = cat.sample((mc_samples,))  # [mc, B*T]
        choices = choices.permute(1, 0).contiguous().view(B, T, mc_samples)  # [B,T,mc]

        # generate samples per component then fill according to choices
        samples = torch.empty(B, T, mc_samples, device=logits.device, dtype=logits.dtype)

        for m_idx, (cname, cp) in enumerate(zip(params["components"], comp_params)):
            sel = (choices == m_idx)  # [B,T,mc]
            if not sel.any():
                continue

            if cname in ("normal", "fixed_normal"):
                mu = cp["mu"]     # [B,T,1]
                sigma = cp["sigma"]
                z = torch.randn(B, T, mc_samples, device=logits.device, dtype=logits.dtype, generator=gen)
                val = mu + sigma * z  # broadcast
                samples[sel] = val[sel]

            elif cname == "student_t":
                mu = cp["mu"]; scale = cp["scale"]; df = cp["df"]
                # chi-square via Gamma(df/2, rate=1/2)
                gamma = torch.distributions.Gamma(concentration=(df/2).expand(B, T, 1), rate=torch.tensor(0.5, device=logits.device, dtype=logits.dtype))
                U = gamma.rsample((mc_samples,))  # [mc,B,T,1]
                U = U.permute(1, 2, 0, 3).squeeze(-1).contiguous()  # [B,T,mc]
                Z = torch.randn(B, T, mc_samples, device=logits.device, dtype=logits.dtype, generator=gen)
                t0 = Z * torch.sqrt(df / U)
                val = mu + scale * t0
                samples[sel] = val[sel]

            elif cname == "log_normal":
                mu = cp["mu"]; sigma = cp["sigma"]
                z = torch.randn(B, T, mc_samples, device=logits.device, dtype=logits.dtype, generator=gen)
                ln = mu + sigma * z
                val = torch.exp(ln)
                samples[sel] = val[sel]

            elif cname == "neg_binomial":
                # torch NB: total_count=r, probs=1-p (prob of success on each trial)
                total_count = cp["r"].clamp_min(1e-6)
                probs = (1.0 - cp["p"]).clamp(1e-6, 1 - 1e-6)
                # sample elementwise
                # build NB with broadcast params; sample mc i.i.d.
                nb = torch.distributions.NegativeBinomial(total_count=total_count.expand(B, T, 1),
                                                          probs=probs.expand(B, T, 1))
                val = nb.sample((mc_samples,)).permute(1, 2, 0, 3).squeeze(-1).contiguous()  # [B,T,mc]
                samples[sel] = val[sel]

            else:
                raise NotImplementedError(f"Quantile sampling for '{cname}' not implemented.")

        # quantiles over MC dimension
        qv = torch.quantile(samples, q.to(samples.dtype), dim=-1)  # [B,T,Q]
        return qv.unsqueeze(-2)  # [B,T,1,Q]

    def get_loss_fn(self) -> Optional[Callable]:
        return None




def _randn_like(x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


@register_module("output_head", "student_t")
class StudentTHead(BaseOutputHead):
    """
    Student's t output head (no torch.special.betainc / StudentT.icdf dependency).

    forward(x): concat params [mu, log_scale, log_df] -> [B, T, 3F]
    predict("mean"|"median") -> [B, T, F]
    sample(...)              -> [B, T, F]
    sample_quantiles(qs)     -> [B, T, F, Q] (via vectorized bisection)
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int = 1,
        *,
        min_log_scale: float = -7.0,
        max_log_scale: float =  5.0,
        min_log_df: float    = -2.0,
        max_log_df: float    =  6.0,
        sigma_floor: float   = 1e-4,
        df_floor: float      = 1.001,
        init_log_scale: Optional[float] = None,
        init_log_df: Optional[float]    = None,
        **kwargs,
    ):
        super().__init__()
        self.feature_size = int(output_size)
        self.proj = nn.Linear(hidden_size, 3 * self.feature_size)

        self.min_log_scale = float(min_log_scale)
        self.max_log_scale = float(max_log_scale)
        self.min_log_df    = float(min_log_df)
        self.max_log_df    = float(max_log_df)

        self.sigma_floor = float(sigma_floor)
        self.df_floor    = float(df_floor)

        if init_log_scale is not None or init_log_df is not None:
            with torch.no_grad():
                if self.proj.bias is None:
                    self.proj.bias = nn.Parameter(torch.zeros(3 * self.feature_size))
                if init_log_scale is not None:
                    self.proj.bias[self.feature_size : 2 * self.feature_size] = float(init_log_scale)
                if init_log_df is not None:
                    self.proj.bias[2 * self.feature_size : 3 * self.feature_size] = float(init_log_df)

    # -------------------- internals --------------------

    def _split_params(self, y: torch.Tensor):
        """ y: [B,T,3F] -> (mu, log_scale, log_df) each [B,T,F], with clamps. """
        mu, log_scale, log_df = torch.split(y, self.feature_size, dim=-1)
        log_scale = log_scale.clamp_(self.min_log_scale, self.max_log_scale)
        log_df    = log_df.clamp_(self.min_log_df,    self.max_log_df)
        return mu, log_scale, log_df

    def _scale_df(self, log_scale: torch.Tensor, log_df: torch.Tensor):
        """ Positive scale & df with floors. """
        scale = torch.exp(log_scale).clamp_min(self.sigma_floor)
        nu    = torch.exp(log_df) + self.df_floor
        return scale, nu

    # Regularized incomplete beta I_x(a,b) via Lentz’s continued fraction (vectorized).
    def _betainc_reg(self, a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, iters: int = 200, eps: float = 3e-7):
        device, dtype = x.device, x.dtype
        tiny = torch.finfo(dtype).tiny
        x = x.clamp(torch.tensor(1e-12, device=device, dtype=dtype),
                    torch.tensor(1.0 - 1e-12, device=device, dtype=dtype))

        lg_ab = torch.lgamma(a + b)
        lg_a  = torch.lgamma(a)
        lg_b  = torch.lgamma(b)

        use_direct = x <= (a + 1.0) / (a + b + 2.0)

        def _betacf(aa: torch.Tensor, bb: torch.Tensor, xx: torch.Tensor):
            qab = aa + bb
            qap = aa + 1.0
            qam = aa - 1.0

            c = torch.ones_like(xx)
            d = 1.0 - (qab * xx) / qap
            d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
            d = 1.0 / d
            h = d.clone()

            for m in range(1, iters + 1):
                m2 = 2 * m
                # even step
                num = m * (bb - m) * xx
                den = (qam + m2) * (aa + m2)
                aa_term = num / den
                d = 1.0 + aa_term * d
                d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
                c = 1.0 + aa_term / c
                c = torch.where(c.abs() < tiny, torch.full_like(c, tiny), c)
                d = 1.0 / d
                h = h * d * c

                # odd step
                num = -(aa + m) * (qab + m) * xx
                den = (aa + m2) * (qap + m2)
                aa_term = num / den
                d = 1.0 + aa_term * d
                d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
                c = 1.0 + aa_term / c
                c = torch.where(c.abs() < tiny, torch.full_like(c, tiny), c)
                d = 1.0 / d
                delta = d * c
                h = h * delta

                if torch.all((delta - 1.0).abs() < eps):
                    break
            return h

        # compute I_x(a,b) with symmetry for stability
        bt1 = torch.exp(lg_ab - lg_a - lg_b + a * torch.log(x) + b * torch.log1p(-x))
        Ix_direct = bt1 * _betacf(a, b, x) / a

        bt2 = torch.exp(lg_ab - lg_a - lg_b + b * torch.log1p(-x) + a * torch.log(x))
        Ix_symm = 1.0 - (bt2 * _betacf(b, a, 1.0 - x) / b)

        return torch.where(use_direct, Ix_direct, Ix_symm).clamp(0.0, 1.0)

    # CDF for standard t via regularized incomplete beta.
    def _student_t_cdf(self, x: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        x_abs = x.abs()
        z = nu / (nu + x_abs * x_abs)
        a = nu / 2.0
        b = torch.as_tensor(0.5, device=nu.device, dtype=nu.dtype)
        I = self._betainc_reg(a, b, z)  # regularized
        cdf_pos = 1.0 - 0.5 * I
        return torch.where(x >= 0, cdf_pos, 0.5 * I)

    # Inverse CDF via vectorized bisection (no special funcs).
    def _student_t_icdf_bisect(self, q: torch.Tensor, nu: torch.Tensor, *, iters: int = 48):
        q = q.clamp(1e-9, 1 - 1e-9)
        q_pos = torch.where(q < 0.5, 1.0 - q, q)

        while nu.dim() < q_pos.dim():
            nu = nu.unsqueeze(-1)

        lo = torch.zeros_like(q_pos)
        hi = torch.ones_like(q_pos)

        for _ in range(18):
            c = self._student_t_cdf(hi, nu)
            mask = c < q_pos
            if not mask.any():
                break
            hi = torch.where(mask, hi * 2.0, hi)

        for _ in range(iters):
            mid = (lo + hi) / 2.0
            c = self._student_t_cdf(mid, nu)
            go_right = c < q_pos
            lo = torch.where(go_right, mid, lo)
            hi = torch.where(go_right, hi, mid)

        t_pos = (lo + hi) / 2.0
        return torch.where(q < 0.5, -t_pos, t_pos)

    # -------------------- API -------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)  # [B,T,3F]
        mu, log_scale, log_df = self._split_params(out)
        return torch.cat([mu, log_scale, log_df], dim=-1)

    def predict(self, params: torch.Tensor, method: str = "mean") -> torch.Tensor:
        mu, log_scale, log_df = self._split_params(params)
        if method not in ("mean", "median"):
            raise ValueError("StudentTHead.predict: use 'mean' or 'median'.")
        return mu  # mean == median (df_floor>1)

    def sample(
        self,
        params: torch.Tensor,
        *,
        method: str = "reparam",
        temperature: float = 1.0,
        eta_blend_mean: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        mu, log_scale, log_df = self._split_params(params)
        scale, nu = self._scale_df(log_scale, log_df)

        if method == "mean" or temperature == 0.0:
            return mu

        if temperature != 1.0:
            scale = scale * float(temperature)

        # U ~ Chi2(nu) via Gamma(nu/2, rate=1/2); rsample matches broadcast shape
        gamma = torch.distributions.Gamma(concentration=nu / 2.0,
                                          rate=torch.tensor(0.5, device=nu.device, dtype=nu.dtype))
        U = gamma.rsample()  # [B,T,F]

        # Z ~ N(0,1)
        Z = _randn_like(mu, generator=generator)

        T0 = Z * torch.sqrt(nu / U)  # Z / sqrt(U/nu)
        y = mu + scale * T0
        if eta_blend_mean:
            y = (1.0 - float(eta_blend_mean)) * y + float(eta_blend_mean) * mu
        return y

    def sample_quantiles(self, params: torch.Tensor, quantile_levels: List[float]) -> torch.Tensor:
        mu, log_scale, log_df = self._split_params(params)
        scale, nu = self._scale_df(log_scale, log_df)

        q = torch.tensor(quantile_levels, dtype=mu.dtype, device=mu.device)
        if not torch.all((q > 0.0) & (q < 1.0)):
            raise ValueError(f"Quantiles must be in (0,1), got {quantile_levels}")

        q4 = q.view(1, 1, 1, -1).expand(*mu.shape, q.numel())
        nu4 = nu.unsqueeze(-1).expand_as(q4)

        t = self._student_t_icdf_bisect(q4, nu4, iters=48)  # [B,T,F,Q]
        return mu.unsqueeze(-1) + scale.unsqueeze(-1) * t

    def get_loss_fn(self) -> Optional[Callable]:
        return None