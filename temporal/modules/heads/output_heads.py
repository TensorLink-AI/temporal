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

        eps = torch.randn_like(mu, generator=generator)
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
    Mixture Density head. Supports components in DIST_PARAM_COUNTS.
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

    @torch.no_grad()
    def sample(
        self,
        params: Dict[str, Union[torch.Tensor, List[str]]],  # forward() output
        *,
        temperature: float = 1.0,
        top_p: Optional[float] = 0.9,
        min_std: float = 1e-4,
        eta_blend_mean: float = 0.3,
    ) -> torch.Tensor:
        """
        Samples one value per step. Assumes univariate per component.
        Returns [B,1,1].
        """
        logits = params["mixture_logits"]  # [B,1,M] or [B,T,M]
        if logits.ndim == 3 and logits.shape[1] == 1:
            logits = logits.squeeze(1)      # [B,M]
        elif logits.ndim == 2:
            pass  # [B,M]
        else:
            # take last timestep if T>1
            logits = logits[..., -1, :] if logits.ndim == 3 else logits

        B, M = logits.shape
        device, dtype = logits.device, logits.dtype

        scaled = _temperature_scale_logits(logits, temperature)
        w = F.softmax(scaled, dim=-1)  # [B,M]
        w = _apply_top_p(w, top_p)
        cat = torch.distributions.Categorical(probs=w)
        k = cat.sample()  # [B]

        # helper to gather [B,1] from param dict (handles [B,1] or [B,T])
        def _get_param(name: str) -> Optional[torch.Tensor]:
            if name not in params:
                return None
            t = params[name]  # [B,1] or [B,T] or [B,1,*] — we assume univariate
            if t.ndim == 3 and t.shape[-1] == 1:
                t = t.squeeze(-1)  # [B,1]
            # Make component-specific by broadcasting if needed (we assume each param is per-component via slicing at forward)
            # Here, our forward produced per-component params concatenated, not per-component stacks.
            # We modeled each as global per component count=1; i.e., every component has its own slot in flat vector.
            # Above we've sliced each parameter into its own tensor with shape [B,T] or [B,1].
            return t

        # We need per-component params, but forward() produced each param as a single scalar channel, not [B,T,M].
        # Given this design, each component’s param is in its own named tensor, not stacked along M.
        # So we reconstruct per-component tensors by reading names with suffix and selecting the chosen one.
        # However, our design names (e.g., "student_mu", "normal_sigma") are single tensors per *component type*, not per *mixture index*.
        # Simpler robust path: if multiple components exist, we assume they are in order in `components`
        # and each component contributes its own params (one set per component in the same time step).
        # We will build lists of tensors [B,1] per component by reading in order.

        comps: List[str] = params["components"]  # list of names
        # Build per-component param dicts
        comp_params: List[Dict[str, torch.Tensor]] = []
        ptrs: Dict[str, int] = {}  # track usage when same key repeats
        for cname in comps:
            cp: Dict[str, torch.Tensor] = {}
            if cname == "normal":
                # expected keys: normal_mu, normal_sigma
                mu = params["normal_mu"]; sigma = params["normal_sigma"]
                cp["mu"] = mu if mu.ndim == 2 else mu[:, -1:]
                sig = sigma if sigma.ndim == 2 else sigma[:, -1:]
                cp["sigma"] = _safe_softplus(sig, min_value=min_std)
            elif cname == "fixed_normal":
                mu = params["normal_mu"]
                cp["mu"] = mu if mu.ndim == 2 else mu[:, -1:]
                cp["sigma"] = torch.full_like(cp["mu"], min_std)
            elif cname == "student_t":
                df = params["student_df"]; mu = params["student_mu"]; sc = params["student_scale"]
                cp["df"] = 2.0 + F.softplus(df if df.ndim == 2 else df[:, -1:])
                cp["mu"] = mu if mu.ndim == 2 else mu[:, -1:]
                cp["scale"] = _safe_softplus(sc if sc.ndim == 2 else sc[:, -1:], min_value=min_std)
            elif cname == "log_normal":
                mu = params["lognorm_mu"]; sig = params["lognorm_sigma"]
                cp["mu"] = mu if mu.ndim == 2 else mu[:, -1:]
                cp["sigma"] = _safe_softplus(sig if sig.ndim == 2 else sig[:, -1:], min_value=min_std)
            elif cname == "neg_binomial":
                r = params["nb_r"]; p = params["nb_p"]
                cp["r"] = _safe_softplus(r if r.ndim == 2 else r[:, -1:], min_value=1e-6)
                cp["p"] = torch.sigmoid(p if p.ndim == 2 else p[:, -1:])
                cp["p"] = cp["p"].clamp(1e-6, 1 - 1e-6)
            else:
                raise NotImplementedError(f"Mixture sampling for '{cname}' not implemented.")
            comp_params.append(cp)

        # Gather chosen component params and sample
        out = torch.empty(B, 1, device=device, dtype=dtype)
        for i in range(B):
            comp_name = comps[k[i].item()]
            p = comp_params[k[i].item()]
            if comp_name in ("normal", "fixed_normal"):
                z = torch.randn_like(out[i:i+1])
                out[i:i+1] = p["mu"][i:i+1] + p["sigma"][i:i+1] * z
            elif comp_name == "student_t":
                z = torch.randn_like(out[i:i+1])
                # Chi-square sampling
                chi = torch.distributions.Chi2(df=p["df"][i:i+1]).sample()
                t = z / torch.sqrt(chi / p["df"][i:i+1])
                out[i:i+1] = p["mu"][i:i+1] + p["scale"][i:i+1] * t
            elif comp_name == "log_normal":
                z = torch.randn_like(out[i:i+1])
                ln = p["mu"][i:i+1] + p["sigma"][i:i+1] * z
                out[i:i+1] = torch.exp(ln)
            elif comp_name == "neg_binomial":
                # sample counts
                # PyTorch parameterization uses total_count (r) and probs (1-p) OR logits
                # We'll use probs=q where q=1-p is "success" prob for failures until r successes.
                total_count = p["r"][i:i+1].clamp_min(1e-6)
                probs = 1.0 - p["p"][i:i+1]
                nb = torch.distributions.NegativeBinomial(total_count=total_count, probs=probs)
                out[i:i+1] = nb.sample()
            else:
                raise NotImplementedError
        # Optional small mean-blend to avoid extreme tails
        # Compute mixture mean (approx) for blend
        mix_mean = torch.zeros_like(out)
        for j, cname in enumerate(comps):
            wj = w[:, j:j+1]  # [B,1]
            if cname in ("normal", "fixed_normal"):
                mix_mean += wj * comp_params[j]["mu"]
            elif cname == "student_t":
                mix_mean += wj * comp_params[j]["mu"]
            elif cname == "log_normal":
                # mean of lognormal = exp(mu + 0.5*sigma^2)
                mu = comp_params[j]["mu"]; sig = comp_params[j]["sigma"]
                mix_mean += wj * torch.exp(mu + 0.5 * sig * sig)
            elif cname == "neg_binomial":
                # mean = r * (p) / (1 - p)   under our p=success prob per trial?
                # We used probs=(1-p) in torch; mean in our paramization (r, p) for #failures until r successes is r*(1-p)/p
                r = comp_params[j]["r"]; pprob = comp_params[j]["p"]
                mix_mean += wj * (r * (1 - pprob) / pprob)
        out = _blend_with_mean(out, mix_mean, eta_blend_mean)
        return _ensure_bt1f(out.unsqueeze(-1))  # [B,1,1]

    def get_loss_fn(self) -> Optional[Callable]:
        return None




@register_module("output_head", "student_t")
class StudentTHead(BaseOutputHead):
    """
    Student's t output head.

    Parameters produced by forward():
      - mu         : location                       [B, T, F]
      - log_scale  : log of scale (>0)              [B, T, F]
      - log_df     : log of degrees of freedom ν    [B, T, F]
    Returned as a single concat tensor: [B, T, 3F]  in the order [mu, log_scale, log_df].

    API:
      - predict("mean"|"median") -> [B, T, F]
      - sample(...)              -> [B, T, F]
      - sample_quantiles(qs)     -> [B, T, F, Q]

    Notes:
      - We enforce numerical stability with clamps and floors:
          ν = exp(log_df) + df_floor   (df_floor ≥ 1.001 ensures mean exists)
          s = exp(log_scale)           (then clamped with sigma_floor)
      - For a symmetric t, median == mu and mean == mu (when ν>1).
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int = 1,             # number of features F
        *,
        # log-parameter clamping
        min_log_scale: float = -7.0,
        max_log_scale: float =  5.0,
        min_log_df: float    = -2.0,      # exp(-2)=~0.135 → +df_floor ensures > 1.136
        max_log_df: float    =  6.0,      # exp(6)=403 → plenty large
        # floors
        sigma_floor: float   = 1e-4,      # minimum scale after exp()
        df_floor: float      = 1.001,     # ensures mean exists
        # optional bias inits (convenient for stable starts)
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

        # Optional: initialize the log-scale / log-df halves with biases
        if init_log_scale is not None or init_log_df is not None:
            with torch.no_grad():
                if self.proj.bias is None:
                    self.proj.bias = nn.Parameter(torch.zeros(3 * self.feature_size))
                # layout: [mu(F), log_scale(F), log_df(F)]
                if init_log_scale is not None:
                    self.proj.bias[self.feature_size : 2 * self.feature_size] = float(init_log_scale)
                if init_log_df is not None:
                    self.proj.bias[2 * self.feature_size : 3 * self.feature_size] = float(init_log_df)

    # -------------------- internals --------------------

    def _split_params(self, y: torch.Tensor):
        """
        y: [B, T, 3F]  -> (mu, log_scale, log_df), each [B, T, F] (or [B,1,F])
        Applies in-place clamps to log_* for stability.
        """
        mu, log_scale, log_df = torch.split(y, self.feature_size, dim=-1)
        log_scale = log_scale.clamp_(self.min_log_scale, self.max_log_scale)
        log_df    = log_df.clamp_(self.min_log_df,    self.max_log_df)
        return mu, log_scale, log_df

    def _scale_df(self, log_scale: torch.Tensor, log_df: torch.Tensor):
        """
        Convert log params to positive scale, df with floors.
          s  = clamp(exp(log_scale), sigma_floor, +inf)
          nu = exp(log_df) + df_floor
        """
        scale = torch.exp(log_scale).clamp_min(self.sigma_floor)
        nu    = torch.exp(log_df) + self.df_floor
        return scale, nu

    # -------------------- API -------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, hidden_size]
        returns: [B, T, 3F] in order [mu, log_scale, log_df], with log_* clamped
        """
        out = self.proj(x)  # [B, T, 3F]
        mu, log_scale, log_df = self._split_params(out)
        return torch.cat([mu, log_scale, log_df], dim=-1)

    def predict(self, params: torch.Tensor, method: str = "mean") -> torch.Tensor:
        """
        params: [B, T, 3F] (or [B, 1, 3F])
        returns: [B, T, F]
        """
        mu, log_scale, log_df = self._split_params(params)
        # For Student's t, mean == median == mu (we enforce nu>1 via df_floor)
        if method not in ("mean", "median"):
            raise ValueError(f"StudentTHead.predict: unsupported method '{method}'. Use 'mean' or 'median'.")
        return mu

    def sample(
        self,
        params: torch.Tensor,                     # [B, T, 3F] or [B, 1, 3F]
        *,
        method: str = "reparam",                  # {"reparam","mean"}
        temperature: float = 1.0,                 # scales the 'scale' parameter
        eta_blend_mean: float = 0.0,              # 0..1: blend sample toward mean
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Reparameterized sampling using the chi-square construction:
          Z ~ N(0,1),  U ~ Chi2(nu),  T = Z / sqrt(U/nu)
          Y = mu + scale * T
        Returns: [B, T, F]
        """
        mu, log_scale, log_df = self._split_params(params)
        scale, nu = self._scale_df(log_scale, log_df)

        if method == "mean" or temperature == 0.0:
            return mu

        # temperature scales the dispersion
        if temperature != 1.0:
            scale = scale * float(temperature)

        # U ~ Chi2(nu)  via Gamma(k=nu/2, rate=1/2)
        gamma = torch.distributions.Gamma(concentration=nu / 2.0, rate=torch.tensor(0.5, device=nu.device, dtype=nu.dtype))
        U = gamma.rsample(sample_shape=mu.shape[:-1]) if generator is None else gamma.rsample(mu.shape[:-1], generator=generator)
        # NOTE: Gamma.rsample accepts sample_shape first; above form ensures shape [B,T,F]

        # Z ~ N(0,1)
        Z = torch.randn_like(mu, generator=generator)

        T0 = Z * torch.sqrt(nu / U)   # Z / sqrt(U/nu)
        y  = mu + scale * T0
        if eta_blend_mean:
            y = (1.0 - float(eta_blend_mean)) * y + float(eta_blend_mean) * mu
        return y

    def sample_quantiles(self, params: torch.Tensor, quantile_levels: List[float]) -> torch.Tensor:
        """
        params: [B, T, 3F]
        returns: [B, T, F, Q]
        """
        mu, log_scale, log_df = self._split_params(params)
        scale, nu = self._scale_df(log_scale, log_df)

        q = torch.tensor(quantile_levels, dtype=mu.dtype, device=mu.device)  # [Q]
        if not torch.all((q > 0.0) & (q < 1.0)):
            raise ValueError(f"Quantiles must be in (0,1), got {quantile_levels}")

        # Standard t quantiles, then affine transform
        t_base = torch.distributions.StudentT(df=nu)
        z = t_base.icdf(q)  # [B, T, F, Q] via broadcasting
        return mu.unsqueeze(-1) + scale.unsqueeze(-1) * z

    def get_loss_fn(self) -> Optional[Callable]:
        return None

