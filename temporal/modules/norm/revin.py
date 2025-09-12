# Licensed under the Apache License, Version 2.0

import torch
from torch import nn
from typing import Optional
from temporal.registry.core import register_module


@register_module("normalization", "revin")
class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time-series.
        It is described in https://openreview.net/forum?id=cGDAkQo1C0p

    This implementation assumes the input tensor is of shape (N, L, C), where
    N is the batch size, L is the sequence length, and C is the number of features.
    Normalization is applied over the L dimension.

    Args:
        num_features (int): The number of features or channels (C).
        eps (float): A value added for numerical stability. Default: 1e-5.
        affine (bool): If True, this module has learnable affine parameters. Default: True.
        subtract_last (bool): If True, subtracts the last element of the sequence from the input. Default: False.
    Input shape: (N, L, C)  [batch, length, features]

    Modes:
      • forward(..., mode='norm', update_stats=True): fit + apply stats
      • forward(..., mode='norm', update_stats=False): apply stored stats only
      • forward(..., mode='denorm'): invert using stored stats
      • transform(...): apply stored stats (no update)
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

        # Per-batch, per-sample statistics (set on .forward(mode='norm', update_stats=True))
        self.mean: Optional[torch.Tensor] = None   # [N,1,C]
        self.stdev: Optional[torch.Tensor] = None  # [N,1,C]
        self.last: Optional[torch.Tensor] = None   # [N,1,C] when subtract_last=True

    def _init_params(self):
        # Broadcastable to [N,L,C]
        self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_features))
        self.affine_bias   = nn.Parameter(torch.zeros(1, 1, self.num_features))

    def forward(self, x: torch.Tensor, mode: str, mask: Optional[torch.Tensor] = None, update_stats: bool = True):
        """
        Args:
            x:    [N,L,C]
            mode: 'norm' | 'denorm'
            mask: [N,L] (1 = valid, 0 = pad)
            update_stats: when mode='norm':
                True  -> fit + apply stats
                False -> apply stored stats only (no update)
        """
        if mode == 'norm':
            if update_stats:
                self._get_statistics(x, mask)
                x = self._normalize(x)
            else:
                x = self.transform(x, mask=mask)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Unknown mode {mode}")
        return x

    # ---- stats ----
    def _get_statistics(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        N, L, C = x.shape

        if self.subtract_last:
            # choose last valid step per batch
            if mask is None:
                idx = torch.full((N, 1, C), L - 1, device=x.device, dtype=torch.long)
            else:
                valid_len = mask.to(dtype=torch.long, device=x.device).sum(dim=1).clamp(min=1)  # [N]
                idx = (valid_len - 1).view(N, 1, 1).expand(N, 1, C)                             # [N,1,C]
            self.last = x.gather(dim=1, index=idx).detach()                                     # [N,1,C]

            x_shifted = x - self.last
            if mask is None:
                self.mean = torch.zeros_like(self.last)
                var = x_shifted.var(dim=1, keepdim=True, unbiased=False)
            else:
                m = mask.to(x.dtype).unsqueeze(-1)  # [N,L,1]
                num = m.sum(dim=1, keepdim=True).clamp(min=1.0)
                self.mean = torch.zeros_like(self.last)
                var = ((x_shifted * m) ** 2).sum(dim=1, keepdim=True) / num

            self.stdev = torch.sqrt(var + self.eps).detach().clamp_min(self.eps)
            return

        # standard path (no subtract_last)
        self.last = None
        if mask is None:
            self.mean  = x.mean(dim=1, keepdim=True).detach()
            var       = x.var(dim=1, keepdim=True, unbiased=False)
            self.stdev = torch.sqrt(var + self.eps).detach().clamp_min(self.eps)
        else:
            m = mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)  # [N,L,1]
            masked_sum = (x * m).sum(dim=1, keepdim=True)
            num = m.sum(dim=1, keepdim=True)
            all_pad = (num == 0)
            num = num.clamp(min=1.0)
            self.mean = (masked_sum / num).detach()

            var = (((x - self.mean) * m) ** 2).sum(dim=1, keepdim=True) / num
            self.stdev = torch.sqrt(var + self.eps).detach().clamp_min(self.eps)

            self.mean  = torch.where(all_pad, torch.zeros_like(self.mean), self.mean)
            self.stdev = torch.where(all_pad, torch.ones_like(self.stdev), self.stdev)

    # ---- forward / inverse ----
    def _normalize(self, x: torch.Tensor):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor):
        if self.stdev is None or self.mean is None:
            raise RuntimeError("RevIN must be normalized before denormalizing.")

        # reshape stats if x has extra dims (e.g., quantiles)
        stdev = self.stdev
        mean  = self.mean
        aw    = self.affine_weight if self.affine else None
        ab    = self.affine_bias   if self.affine else None
        last  = self.last

        if x.ndim > stdev.ndim:
            extra = (1,) * (x.ndim - stdev.ndim)
            stdev = stdev.view(stdev.shape + extra)
            mean  = mean.view(mean.shape + extra)
            if aw is not None: aw = aw.view(aw.shape + extra)
            if ab is not None: ab = ab.view(ab.shape + extra)
            if last is not None: last = last.view(last.shape + extra)

        if self.affine:
            x = x - ab
            x = x / torch.clamp(aw, min=1e-6)  # FIX: clamp instead of +eps*eps

        x = x * stdev
        if self.subtract_last:
            x = x + last
        else:
            x = x + mean
        return x

    def transform(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Apply using stored stats (no update)."""
        if self.stdev is None or self.mean is None:
            raise RuntimeError("RevIN must be normalized (fit stats) before transform().")
        if self.subtract_last:
            x_t = x - self.last
        else:
            x_t = x - self.mean
        x_t = x_t / self.stdev
        if self.affine:
            x_t = x_t * self.affine_weight + self.affine_bias

        if mask is not None:
            m = mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)  # [N,L,1]
            return torch.where(m.bool(), x_t, x)
        return x_t

    def inverse_transform(self, x: torch.Tensor):
        return self._denormalize(x)


@register_module("normalization", "revin2d")
class RevIN2d(nn.Module):
    """
    Reversible Instance Normalization for 2D tensors.
    Input: (N, C, H, W). Mask is (N, H, W) with 1=valid.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

        self.mean: Optional[torch.Tensor] = None  # [N,C,1,1]
        self.stdev: Optional[torch.Tensor] = None # [N,C,1,1]

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1, 1))
        self.affine_bias   = nn.Parameter(torch.zeros(1, self.num_features, 1, 1))

    def forward(self, x: torch.Tensor, mode: str, mask: Optional[torch.Tensor] = None, update_stats: bool = True):
        if mode == 'norm':
            if update_stats:
                self._get_statistics(x, mask)
                x = self._normalize(x)
            else:
                x = self.transform(x, mask=mask)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.subtract_last:
            raise NotImplementedError("`subtract_last` is not implemented for RevIN2d.")
        dims = (2, 3)
        if mask is None:
            self.mean  = x.mean(dim=dims, keepdim=True).detach()
            var        = x.var(dim=dims, keepdim=True, unbiased=False)
            self.stdev = torch.sqrt(var + self.eps).detach().clamp_min(self.eps)
        else:
            m = mask.to(dtype=x.dtype, device=x.device).unsqueeze(1)  # [N,1,H,W]
            masked_sum = (x * m).sum(dim=dims, keepdim=True)
            num = m.sum(dim=dims, keepdim=True)
            all_pad = (num == 0)
            num = num.clamp(min=1.0)

            self.mean = (masked_sum / num).detach()
            var = (((x - self.mean) * m) ** 2).sum(dim=dims, keepdim=True) / num
            self.stdev = torch.sqrt(var + self.eps).detach().clamp_min(self.eps)

            self.mean  = torch.where(all_pad, torch.zeros_like(self.mean), self.mean)
            self.stdev = torch.where(all_pad, torch.ones_like(self.stdev), self.stdev)

    def _normalize(self, x: torch.Tensor):
        if self.subtract_last:
            raise NotImplementedError("`subtract_last` is not implemented for RevIN2d.")
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor):
        if self.stdev is None or self.mean is None:
            raise RuntimeError("RevIN2d must be normalized before denormalizing.")
        if self.affine:
            x = x - self.affine_bias
            x = x / torch.clamp(self.affine_weight, min=1e-6)  # FIX
        x = x * self.stdev + self.mean
        return x

    def transform(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.stdev is None or self.mean is None:
            raise RuntimeError("RevIN2d must be normalized before transform().")
        x_t = (x - self.mean) / self.stdev
        if self.affine:
            x_t = x_t * self.affine_weight + self.affine_bias
        if mask is not None:
            m = mask.to(dtype=x.dtype, device=x.device).unsqueeze(1)
            return torch.where(m.bool(), x_t, x)
        return x_t

    def inverse_transform(self, x: torch.Tensor):
        return self._denormalize(x)
