# temporal/modules/losses/timeflow_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from temporal.modules.losses.losses import BaseTemporalLoss
from temporal.registry.core import register_module

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply shift & scale in an AdaLN fashion."""
    return x * (1 + scale) + shift


class TimeFlowTimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into a vector for TimeFlowLoss."""
    def __init__(self, emb_dim: int, freq_emb_dim: int = 256):
        super().__init__()
        self.freq_emb_dim = freq_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_emb_dim, emb_dim, bias=True),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim, bias=True),
        )

    @staticmethod
    def _build_sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) scalar timesteps (can be fractional)
        sinus = self._build_sinusoidal_embedding(t, self.freq_emb_dim)
        return self.mlp(sinus)  # → (B, emb_dim)


class TimeFlowResBlock(nn.Module):
    """A single residual block with AdaLN modulation for TimeFlowMLPAdaLN."""
    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        # produces shift, scale, gate
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.mod(cond).chunk(3, dim=-1)
        h = _modulate(self.in_ln(x), shift, scale)
        h = self.mlp(h)
        return x + gate * h


class TimeFlowFinalLayer(nn.Module):
    """Final projection layer with AdaLN for TimeFlowMLPAdaLN."""
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        # produces shift, scale
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mod(cond).chunk(2, dim=-1)
        h = _modulate(self.norm(x), shift, scale)
        return self.linear(h)


class TimeFlowMLPAdaLN(nn.Module):
    """
    The MLP backbone for TimeFlowLoss.
    Projects inputs → model_channels, applies num_blocks ResBlocks, then final projection.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        cond_channels: int,
        num_blocks: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.time_embed = TimeFlowTimestepEmbedder(model_channels)
        self.cond_proj  = nn.Linear(cond_channels, model_channels)

        self.blocks = nn.ModuleList([
            TimeFlowResBlock(model_channels)
            for _ in range(num_blocks)
        ])
        self.final = TimeFlowFinalLayer(model_channels, out_channels)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

        for block in self.blocks:
            nn.init.zeros_(block.mod[-1].weight)
            nn.init.zeros_(block.mod[-1].bias)
        nn.init.zeros_(self.final.mod[-1].weight)
        nn.init.zeros_(self.final.mod[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        te = self.time_embed(t)
        ce = self.cond_proj(cond)
        ctx = te + ce
        for block in self.blocks:
            h = block(h, ctx)
        return self.final(h, ctx)

@register_module("loss", "timeflow")
class TimeFlowLoss(BaseTemporalLoss):
    """
    The TimeFlow loss module for temporal forecasting.
    Given target sequences and a conditioning vector z, computes a diffusion-style MSE loss.
    This loss is stateful and contains its own neural network.
    """
    def __init__(
        self,
        target_channels: int,
        cond_channels: int,
        num_blocks: int,
        model_channels: int,
        num_sampling_steps: int = 10,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.net = TimeFlowMLPAdaLN(
            in_channels=target_channels,
            model_channels=model_channels,
            out_channels=target_channels,
            cond_channels=cond_channels,
            num_blocks=num_blocks,
        )
        self.num_sampling_steps = num_sampling_steps

    def forward(
        self,
        preds: torch.Tensor, # Expected to be the conditioning vector `z`
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # preds are the conditioning vector z, targets are the ground truth y
        cond = preds
        
        # 1) sample mixing coefficient t ∈ [0,1]
        noise = torch.randn_like(targets)
        t = torch.rand(targets.size(0), device=targets.device)

        # 2) interpolate target & noise
        noised = t[:, None] * targets + (1 - t[:, None]) * noise

        # 3) predict target from noised + cond
        pred_denoised = self.net(noised, t * 1000, cond)

        # 4) weighted MSE over channels
        weights = 1.0 / torch.arange(
            1, targets.size(-1) + 1,
            device=targets.device, dtype=torch.float32
        )
        err = (pred_denoised - targets) ** 2
        
        # This loss assumes channel-last, so weights are applied to the last dim.
        err = weights * err
        loss = err.sum(dim=-1)

        # 5) apply sequence mask (if any) and reduction
        return self._apply_reduction(loss, loss_mask)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generates samples via simple Euler discretization.
        """
        self.eval()
        B = cond.size(0)
        cond_expanded = cond.repeat_interleave(num_samples, dim=0)
        
        x = torch.randn(B * num_samples, self.net.input_proj.in_features, device=cond.device)
        dt = 1.0 / self.num_sampling_steps

        for i in range(self.num_sampling_steps):
            t_val = i / self.num_sampling_steps
            t = torch.full((B * num_samples,), t_val, device=cond.device)
            pred = self.net(x, t * 1000, cond_expanded)
            # ODE update: x_{t+dt} = x_t + (flow - x_t) * dt
            # Here, `pred` is the predicted target (x_1), so flow is `pred - x_t`.
            # But the original code had `(pred - x) * dt`, which assumes `pred` is the flow `v_t`.
            # Let's stick to the original paper's simple Euler update for flow `pred - x_t`
            v_t = pred - x
            x = x + v_t * dt

        self.train()
        # reshape to (B, num_samples, C)
        return x.view(B, num_samples, -1)

