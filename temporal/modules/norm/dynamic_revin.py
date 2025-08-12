import torch
from torch import nn
from typing import Dict, Union, Optional

from temporal.registry.core import register_module

@register_module("normalization", "dynamic_revin")
class DynamicRevIN(nn.Module):
    """
    Dynamic RevIN for channels-last tensors [B, L, F] with switchable affine:
      - 'fixed'            : learnable γ, β (broadcast [1,1,F])
      - {'type':'dynamic', 'mapper':'linear' | 'mlp', 'hidden_dim':16, 'gamma_positive':True}

    Modes:
      - mode='norm'      : compute stats on x, compute affine, normalize & store stats/affine
      - mode='transform' : normalize using last stored stats/affine (no recompute)
      - mode='denorm'    : invert last stored affine + stats to original scale
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine_mode: Union[str, Dict] = 'fixed'):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.cfg = affine_mode if isinstance(affine_mode, dict) else {'type':'fixed'}
        self.affine_type = self.cfg.get('type', 'fixed')
        self.gamma_positive = bool(self.cfg.get('gamma_positive', True))

        # ----- affine params / mapper -----
        if self.affine_type == 'fixed':
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta  = nn.Parameter(torch.zeros(1, 1, num_features))
        elif self.affine_type == 'dynamic':
            mapper = self.cfg.get('mapper', 'linear')
            if mapper == 'linear':
                self.mapper = nn.Linear(2, 2)
            elif mapper == 'mlp':
                hid = int(self.cfg.get('hidden_dim', 16))
                self.mapper = nn.Sequential(
                    nn.Linear(2, hid), nn.ReLU(), nn.Linear(hid, 2)
                )
            else:
                raise ValueError("dynamic mapper must be 'linear' or 'mlp'")
            # init mapper to identity: gamma=1, beta=0
            with torch.no_grad():
                last = self.mapper[-1] if isinstance(self.mapper, nn.Sequential) else self.mapper
                last.weight.zero_()
                last.bias.zero_()
                if self.gamma_positive:
                    last.bias[0].fill_(0.541324854)
                else:
                    last.bias[0].fill_(1.0)
                last.bias[1].fill_(0.0)

        else:
            raise ValueError("affine_mode must be 'fixed' or dict(type='dynamic', ...)")

        # ----- runtime buffers (NOT saved) -----
        self.register_buffer('mean',  None, persistent=False)     # [B,1,F]
        self.register_buffer('stdev', None, persistent=False)     # [B,1,F]
        self.register_buffer('last_gamma', None, persistent=False) # [B,1,F]
        self.register_buffer('last_beta',  None, persistent=False) # [B,1,F]

    def forward(self, x: torch.Tensor, mode: str, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mode == 'norm':
            return self._norm_compute(x, mask)
        elif mode == 'denorm':
            return self._denorm(x)
        else:
            raise NotImplementedError(f"mode must be 'norm' | 'denorm'")

    def _compute_stats(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is None:
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            var = torch.var(x, dim=1, unbiased=False, keepdim=True)
            self.stdev = (var + self.eps).sqrt().detach().clamp_min(self.eps)
        else:
            mask = mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
            masked_sum = torch.sum(x * mask, dim=1, keepdim=True)
            num_non_masked = torch.sum(mask, dim=1, keepdim=True)

            all_pad = (num_non_masked == 0)
            num_non_masked = torch.clamp(num_non_masked, min=1)
            
            self.mean = (masked_sum / num_non_masked).detach()
            
            variance = torch.sum(((x - self.mean) * mask)**2, dim=1, keepdim=True) / num_non_masked
            self.stdev = torch.sqrt(variance + self.eps).detach().clamp_min(self.eps)

            self.mean  = torch.where(all_pad, torch.zeros_like(self.mean), self.mean)
            self.stdev = torch.where(all_pad, torch.ones_like(self.stdev), self.stdev)


    def _compute_affine_shared(self):
        B, _, F = self.mean.shape
        stats = torch.stack([self.mean, self.stdev], dim=-1).squeeze(1)   # [B,F,2]
        out = self.mapper(stats.view(-1, 2)).view(B, F, 2)        # [B,F,2]
        gamma = out[..., 0].unsqueeze(1)                                  # [B,1,F]
        beta  = out[..., 1].unsqueeze(1)                                  # [B,1,F]
        if self.gamma_positive:
            gamma = torch.nn.functional.softplus(gamma)
        return gamma, beta

    def _norm_compute(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # compute stats & affine ON x, then normalize and store affine
        self._compute_stats(x, mask)
        if self.affine_type == 'fixed':
            gamma = self.gamma.expand(x.size(0), -1, -1)  # [B,1,F]
            beta  = self.beta.expand(x.size(0), -1, -1)
        else:
            gamma, beta = self._compute_affine_shared()
        self.last_gamma = gamma.to(dtype=x.dtype, device=x.device)
        self.last_beta  = beta.to(dtype=x.dtype, device=x.device)
        x_norm = (x - self.mean) / self.stdev
        x_norm = x_norm * self.last_gamma + self.last_beta

        if mask is not None:
             mask_expanded = mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
             return torch.where(mask_expanded.bool(), x_norm, x)
        return x_norm


    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        # invert affine then de-standardize
        assert self.mean is not None and self.stdev is not None, "No stored stats."
        if self.last_gamma is None or self.last_beta is None:
            # fall back to fixed params if available
            gamma = (self.gamma if self.affine_type == 'fixed' else torch.ones_like(self.mean)).to(x)
            beta  = (self.beta  if self.affine_type == 'fixed' else torch.zeros_like(self.mean)).to(x)
        else:
            gamma, beta = self.last_gamma.to(x), self.last_beta.to(x)
        x = (x - beta) / (gamma + self.eps)
        x = x * self.stdev + self.mean
        return x

    def transform(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # reuse last stats & affine (for targets)
        assert self.mean is not None and self.stdev is not None, "Call mode='norm' first."
        assert self.last_gamma is not None and self.last_beta is not None, "Affine not set. Call mode='norm' first."
        
        x_transformed = (x - self.mean) / self.stdev
        x_transformed = x_transformed * self.last_gamma + self.last_beta

        if attention_mask is not None:
            mask_expanded = attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
            # Where mask is True, use transformed values. Where False, use original values.
            return torch.where(mask_expanded.bool(), x_transformed, x)

        return x_transformed

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self._denorm(x)
