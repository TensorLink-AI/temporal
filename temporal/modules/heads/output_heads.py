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
        init_log_sigma: Optional[float] = None,
        allow_lazy_infer: bool = True,     # <<< new: auto-fix in_features on first forward
        **kwargs,
    ):
        super().__init__()
        self.feature_size = int(output_size)
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)
        self.sigma_floor   = float(sigma_floor)
        self.allow_lazy_infer = bool(allow_lazy_infer)

        # If hidden_size looks bogus (e.g., 0/1 while your decoder is 256/512/etc),
        # we’ll still build something, but we’ll fix it on first forward if allowed.
        in_features = max(int(hidden_size), 1)
        self.proj = nn.Linear(in_features, 2 * self.feature_size)

        if init_log_sigma is not None:
            with torch.no_grad():
                if self.proj.bias is None:
                    self.proj.bias = nn.Parameter(torch.zeros(2 * self.feature_size))
                # bias layout: [mu(F), log_sigma(F)]
                self.proj.bias[self.feature_size:] = float(init_log_sigma)

    # ---- internals ---------------------------------------------------------

    def _maybe_fix_in_features(self, x_last_dim: int):
        if self.proj.in_features == x_last_dim:
            return
        if not self.allow_lazy_infer:
            raise RuntimeError(
                f"GaussianHead expected hidden_size={self.proj.in_features} but got {x_last_dim}. "
                f"Construct the head with hidden_size=decoder_hidden_size (e.g. 512) and "
                f"output_size=feature_size."
            )
        # Lazy re-init to match the true hidden size
        old_out = self.proj.out_features
        new = nn.Linear(x_last_dim, old_out)
        # (Optional) leave weights random; copying isn’t possible due to size change.
        self.proj = new

    def _split_params(self, y: torch.Tensor):
        mu, log_sigma = y.chunk(2, dim=-1)
        # Use the out-of-place version
        log_sigma = log_sigma.clamp(self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma

    # ---- API ---------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, hidden_size]
        returns: [B, T, 2F]  (concat of mu and log_sigma)
        """
        # Ensure the linear matches the actual hidden size
        self._maybe_fix_in_features(x.size(-1))
        out = self.proj(x)  # [B, T, 2F]
        mu, log_sigma = self._split_params(out)
        return torch.cat([mu, log_sigma], dim=-1)

    def predict(self, params: torch.Tensor, method: str = "mean") -> torch.Tensor:
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
        mu, log_sigma = self._split_params(params)
        if method == "mean" or temperature == 0.0:
            return mu

        sigma = torch.exp(log_sigma)
        sigma = sigma.clamp_min(self.sigma_floor if min_std is None else float(min_std))
        if temperature != 1.0:
            sigma = sigma * float(temperature)

        if generator is None:
            eps = torch.randn_like(mu)
        else:
            eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
        y = mu + eps * sigma
        if eta_blend_mean:
            y = (1.0 - float(eta_blend_mean)) * y + float(eta_blend_mean) * mu
        return y

    def sample_quantiles(self, params: torch.Tensor, quantile_levels: List[float]) -> torch.Tensor:
        mu, log_sigma = self._split_params(params)
        sigma = torch.exp(log_sigma)
        q = torch.tensor(quantile_levels, dtype=mu.dtype, device=mu.device)  # [Q]
        z = torch.distributions.Normal(0.0, 1.0).icdf(q)                     # [Q]
        return mu.unsqueeze(-1) + sigma.unsqueeze(-1) * z.view(1, 1, 1, -1)  # [B,T,F,Q]



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


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# from temporal.modules.registry import register_module   # keep your existing import
# from temporal.modules.heads.base import BaseOutputHead  # keep your existing import


@register_module("output_head", "mixture")
class MixtureOutputHead(BaseOutputHead):
    """
    Univariate Mixture Density head (heterogeneous components supported).
    forward(...) -> dict of params (same keys as your original)
    predict(...) -> mixture mean [B,T,1]
    sample(...)  -> one-step sample   [B,1,1]
    sample_quantiles(...) -> MC quantiles [B,T,1,Q] (post-loop friendly)
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

        self.feature_size = int(kwargs.get("feature_size", 1))
        if self.feature_size != 1:
            raise NotImplementedError("This MDN is univariate (feature_size=1).")

        # Build flat projection size
        total_params_dim = 0
        self.param_indices: Dict[str, Tuple[int, int]] = {}
        cur = 0
        for dist_name in self.components:
            if dist_name not in self.DIST_PARAM_COUNTS:
                raise ValueError(f"Unknown component '{dist_name}'.")
            for pkey, cnt in self.DIST_PARAM_COUNTS[dist_name].items():
                out_key = self.DIST_OUTPUT_KEYS[dist_name][pkey]
                self.param_indices[out_key] = (cur, cur + cnt)
                cur += cnt
                total_params_dim += cnt
        self.mixture_logits_indices = (cur, cur + self.num_components)
        total_params_dim += self.num_components

        self.output_projection = nn.Linear(hidden_size, total_params_dim)
        if "output_size" in kwargs and kwargs["output_size"] != total_params_dim:
            raise ValueError(
                f"MixtureOutputHead: provided output_size={kwargs['output_size']} "
                f"!= derived={total_params_dim}"
            )

    # ---------- helpers ----------
    def _weights(self, params: Dict[str, torch.Tensor], *, temperature: float = 1.0, top_p: Optional[float] = None) -> torch.Tensor:
        """Return mixture weights w: [B,T,M]."""
        logits = params["mixture_logits"]  # [B,T,M] (preferred) or [B,1,M] or [B,M]
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)  # -> [B,1,M]
        scaled = _temperature_scale_logits(logits, temperature)
        w = F.softmax(scaled, dim=-1)    # [B,T,M]
        return _apply_top_p(w, top_p)

    def _as_bt1(self, t: torch.Tensor) -> torch.Tensor:
        """Ensure [B,T,1]. Accept [B,T], [B,1], [B] -> broadcast to [B,T,1] as needed."""
        if t.ndim == 3:
            return t  # [B,T,1]
        if t.ndim == 2:
            return t.unsqueeze(-1)  # [B,T] -> [B,T,1]
        if t.ndim == 1:
            return t.view(t.size(0), 1, 1)
        raise ValueError(f"Unexpected tensor shape {tuple(t.shape)}")

    def _get_params_per_component(self, params: Dict[str, torch.Tensor], *, min_std: float = 1e-4) -> List[Dict[str, torch.Tensor]]:
        """List of per-component dicts; each param shaped [B,T,1]."""
        out: List[Dict[str, torch.Tensor]] = []
        for cname in params["components"]:
            cp: Dict[str, torch.Tensor] = {}
            if cname == "normal":
                mu = self._as_bt1(params["normal_mu"])
                sig = self._as_bt1(params["normal_sigma"])
                cp["mu"] = mu
                cp["sigma"] = _safe_softplus(sig, min_value=min_std)
            elif cname == "fixed_normal":
                mu = self._as_bt1(params["normal_mu"])
                cp["mu"] = mu
                cp["sigma"] = torch.full_like(mu, min_std)
            elif cname == "student_t":
                df = self._as_bt1(params["student_df"])
                mu = self._as_bt1(params["student_mu"])
                sc = self._as_bt1(params["student_scale"])
                cp["df"] = 2.0 + F.softplus(df)               # > 2 (finite mean)
                cp["mu"] = mu
                cp["scale"] = _safe_softplus(sc, min_value=min_std)
            elif cname == "log_normal":
                mu = self._as_bt1(params["lognorm_mu"])
                sig = self._as_bt1(params["lognorm_sigma"])
                cp["mu"] = mu
                cp["sigma"] = _safe_softplus(sig, min_value=min_std)
            elif cname == "neg_binomial":
                r = self._as_bt1(params["nb_r"])
                p = self._as_bt1(params["nb_p"])
                cp["r"] = _safe_softplus(r, min_value=1e-6)
                cp["p"] = torch.sigmoid(p).clamp(1e-6, 1 - 1e-6)
            else:
                raise NotImplementedError(cname)
            out.append(cp)
        return out

    # ---------- forward / predict ----------
    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[str]]]:
        flat = self.output_projection(x)  # [B,T,Dp]
        out: Dict[str, Union[torch.Tensor, List[str]]] = {"components": self.components}
        s, e = self.mixture_logits_indices
        out["mixture_logits"] = flat[..., s:e]  # [B,T,M]
        seen = set()
        for dist_name in self.components:
            for pkey in self.DIST_PARAM_COUNTS[dist_name]:
                out_key = self.DIST_OUTPUT_KEYS[dist_name][pkey]
                if out_key in seen:
                    continue
                a, b = self.param_indices[out_key]
                t = flat[..., a:b]                     # [B,T,1] (or [B,T] if cnt==1 then squeeze below)
                out[out_key] = t.squeeze(-1) if t.shape[-1] == 1 else t
                seen.add(out_key)
        return out

    def predict(self, params: Dict[str, torch.Tensor], method: str = "mean") -> torch.Tensor:
        """Mixture mean (univariate) -> [B,T,1]."""
        if method not in ("mean", "median"):
            raise ValueError("MixtureOutputHead.predict supports 'mean' or 'median'.")
        w = self._weights(params)                                    # [B,T,M]
        comps = self._get_params_per_component(params)               # list of {param:[B,T,1]}
        means = []
        for cname, cp in zip(params["components"], comps):
            if cname in ("normal", "fixed_normal", "student_t"):
                means.append(cp["mu"])                               # [B,T,1]
            elif cname == "log_normal":
                means.append(torch.exp(cp["mu"] + 0.5 * cp["sigma"]**2))
            elif cname == "neg_binomial":
                r, p = cp["r"], cp["p"]
                means.append(r * (1 - p) / p)
        mean = torch.zeros_like(means[0])
        for j, m in enumerate(means):
            mean = mean + w[..., j:j+1] * m                          # [B,T,1]
        return mean

    @torch.no_grad()
    def sample_quantiles(
        self,
        params: Dict[str, torch.Tensor],
        quantile_levels: List[float],
        *,
        num_mc: int = 256,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        min_std: float = 1e-4,
    ) -> torch.Tensor:
        """
        Monte Carlo mixture quantiles (univariate MDN).
        Returns: [B, T, 1, Q]
        """
        logits = params["mixture_logits"]
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)  # -> [B,1,M]
        B, T, M = logits.shape
        device, dtype = logits.device, logits.dtype

        q = torch.as_tensor(quantile_levels, device=device, dtype=dtype)
        if not torch.all((q > 0) & (q < 1)):
            raise ValueError(f"quantile_levels must be in (0,1); got {quantile_levels}")
        Q = int(q.numel())

        # Mixture weights [B,T,M]
        w = self._weights(params, temperature=temperature, top_p=top_p)
        if w.shape != (B, T, M):
            raise RuntimeError(f"weights shape mismatch: expected {(B,T,M)}, got {tuple(w.shape)}")

        comps = self._get_params_per_component(params, min_std=min_std)  # each param ~ [B,T,1]
        if len(comps) != M:
            raise RuntimeError(f"components length {len(comps)} != M {M}")

        S = int(num_mc)
        samples_per_comp: List[torch.Tensor] = []

        # Helper: ensure param tensors broadcast with [B,T,S] without creating extra dims
        def _bt_1_to_bts(t: torch.Tensor) -> torch.Tensor:
            # Accept [B,T], [B,T,1], or [B,T,S]; return one that broadcasts with [B,T,S]
            if t.ndim == 2:
                return t.unsqueeze(-1)                        # [B,T,1]
            if t.ndim == 3:
                return t                                      # [B,T,1] or [B,T,S]
            raise RuntimeError(f"Unexpected param shape {tuple(t.shape)}")

        for cname, cp in zip(params["components"], comps):
            if cname in ("normal", "fixed_normal"):
                mu    = _bt_1_to_bts(cp["mu"])               # [B,T,1] or [B,T,S]
                sigma = _bt_1_to_bts(cp["sigma"])            # [B,T,1] or [B,T,S]
                z = torch.randn(B, T, S, device=device, dtype=dtype)             # [B,T,S]
                y = mu + sigma * z                                             # -> [B,T,S]

            elif cname == "student_t":
                mu    = _bt_1_to_bts(cp["mu"])
                scale = _bt_1_to_bts(cp["scale"])
                df    = _bt_1_to_bts(cp["df"])                                  # [B,T,1] or [B,T,S]
                z = torch.randn(B, T, S, device=device, dtype=dtype)             # [B,T,S]
                gamma = torch.distributions.Gamma(
                    concentration=df / 2.0,
                    rate=torch.tensor(0.5, device=device, dtype=dtype),
                )
                # Gamma.rsample((S,)) -> [S,B,T,1] (broadcasted), permute -> [B,T,S]
                U = gamma.rsample(sample_shape=(S,)).permute(1, 2, 0, 3).squeeze(-1)  # [B,T,S]
                t0 = z * torch.sqrt(df.expand_as(U) / U)                               # [B,T,S]
                y  = mu + scale * t0                                                   # [B,T,S]

            elif cname == "log_normal":
                mu    = _bt_1_to_bts(cp["mu"])
                sigma = _bt_1_to_bts(cp["sigma"])
                z = torch.randn(B, T, S, device=device, dtype=dtype)             # [B,T,S]
                ln = mu + sigma * z                                              # **no unsqueeze on z**
                y  = torch.exp(ln)                                               # [B,T,S]

            elif cname == "neg_binomial":
                r = _bt_1_to_bts(cp["r"])                                        # [B,T,1] or [B,T,S]
                p = _bt_1_to_bts(cp["p"])                                        # [B,T,1] or [B,T,S]
                nb = torch.distributions.NegativeBinomial(total_count=r, probs=p)
                y = nb.sample((S,)).permute(1, 2, 0, 3).squeeze(-1)              # [B,T,S]

            else:
                raise NotImplementedError(f"component '{cname}' not implemented")

            if y.shape != (B, T, S):
                raise RuntimeError(f"component '{cname}' produced {tuple(y.shape)}; expected {(B,T,S)}")
            samples_per_comp.append(y)

        # Stack per-component -> [B,T,M,S]
        Y_all = torch.stack(samples_per_comp, dim=2)

        # Draw component indices for each of the S MC samples -> [B,T,S]
        cat = torch.distributions.Categorical(probs=w)           # probs [B,T,M]
        k = cat.sample((S,)).permute(1, 2, 0)                    # [B,T,S]

        # Gather: work in [B,T,S,M] then pick last dim with k
        Y = torch.gather(
            Y_all.permute(0, 1, 3, 2),                            # [B,T,S,M]
            dim=-1,
            index=k.unsqueeze(-1)                                 # [B,T,S,1]
        ).squeeze(-1)                                            # -> [B,T,S]

        # Quantiles across S -> [B,T,Q] -> [B,T,1,Q]
        qs = [float(v) for v in quantile_levels]                 # ensure scalars
        q_slices = [torch.quantile(Y, v, dim=-1) for v in qs]    # each [B,T]
        qvals = torch.stack(q_slices, dim=-1)                    # [B,T,Q]

        # Final shape for bundle consistency
        return qvals.unsqueeze(-2)                               # [B,T,1,Q]

    # ---------- one-step sample for AR feedback ----------
    @torch.no_grad()
    def sample(
        self,
        params: Dict[str, Union[torch.Tensor, List[str]]],
        *,
        temperature: float = 1.0,
        top_p: Optional[float] = 0.9,
        min_std: float = 1e-4,
        eta_blend_mean: float = 0.3,
    ) -> torch.Tensor:
        """
        Sample a single step for AR feedback. Returns [B,1,1].
        """
        logits = params["mixture_logits"]
        if logits.ndim == 3:
            logits_last = logits[:, -1, :]     # [B,M]
        elif logits.ndim == 2:
            logits_last = logits               # [B,M]
        else:
            raise ValueError(f"Unexpected logits shape {tuple(logits.shape)}")

        B, M = logits_last.shape
        device, dtype = logits_last.device, logits_last.dtype

        w_last = F.softmax(_temperature_scale_logits(logits_last, temperature), dim=-1)  # [B,M]
        w_last = _apply_top_p(w_last, top_p)
        k = torch.distributions.Categorical(probs=w_last).sample()  # [B]

        comps = self._get_params_per_component(params, min_std=min_std)

        def _last(t):  # [B,*,1] -> [B,1,1]
            return t[:, -1:, :] if t.size(1) > 1 else t

        out = torch.empty(B, 1, 1, device=device, dtype=dtype)  # [B,1,1]
        for i in range(B):
            cname = params["components"][k[i].item()]
            cp = comps[k[i].item()]
            if cname in ("normal", "fixed_normal"):
                z = torch.randn_like(out[i:i+1])
                out[i:i+1] = _last(cp["mu"])[i:i+1] + _last(cp["sigma"])[i:i+1] * z
            elif cname == "student_t":
                z = torch.randn_like(out[i:i+1])
                df = _last(cp["df"])[i:i+1]
                chi = torch.distributions.Chi2(df=df).sample()
                t  = z / torch.sqrt(chi / df)
                out[i:i+1] = _last(cp["mu"])[i:i+1] + _last(cp["scale"])[i:i+1] * t
            elif cname == "log_normal":
                z = torch.randn_like(out[i:i+1])
                ln = _last(cp["mu"])[i:i+1] + _last(cp["sigma"])[i:i+1] * z
                out[i:i+1] = torch.exp(ln)
            elif cname == "neg_binomial":
                total_count = _last(cp["r"])[i:i+1].clamp_min(1e-6)
                probs = 1.0 - _last(cp["p"])[i:i+1]
                nb = torch.distributions.NegativeBinomial(total_count=total_count, probs=probs)
                out[i:i+1] = nb.sample()
            else:
                raise NotImplementedError

        # optional mean-blend
        mix_mean = torch.zeros_like(out)                        # [B,1,1]
        for j, cname in enumerate(params["components"]):
            cp = comps[j]
            if cname in ("normal", "fixed_normal", "student_t"):
                m = _last(cp["mu"])                             # [B,1,1]
            elif cname == "log_normal":
                mu, sig = _last(cp["mu"]), _last(cp["sigma"])
                m = torch.exp(mu + 0.5 * sig * sig)
            elif cname == "neg_binomial":
                r, p = _last(cp["r"]), _last(cp["p"])
                m = r * (1 - p) / p
            wj = w_last[:, j:j+1].unsqueeze(-1)                # [B,1,1]
            mix_mean += wj * m
        out = _blend_with_mean(out, mix_mean, eta=eta_blend_mean)
        return out  # [B,1,1]

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

    def _split_params(self, y):
        """ y: [B,T,3F] -> (mu, log_scale, log_df) each [B,T,F], with clamps. """
        mu, log_scale, log_df = torch.split(y, self.feature_size, dim=-1)
        # Use out-of-place clamp to create new tensors
        log_scale = log_scale.clamp(self.min_log_scale, self.max_log_scale)
        log_df    = log_df.clamp(self.min_log_df,    self.max_log_df)
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

