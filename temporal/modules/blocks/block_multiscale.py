import torch
import torch.nn as nn
from einops import rearrange
from temporal.registry.core import register_module
from typing import Optional, Tuple


@register_module("block", "multiscale")
class MultiScaleBlock(nn.Module):
    """
    Applies attention at two scales:
      - Fine: full resolution
      - Coarse: downsampled (e.g., 4x), then upsampled

    Combines both outputs via additive, concat+proj, or gating fusion.
    """

    def __init__(
        self,
        fine_attn: nn.Module,
        coarse_attn: nn.Module,
        downsample_factor: int = 4,
        fusion_method: str = "add",  # options: "add", "concat", "gate"
        hidden_size: int = 128,
        **kwargs
    ):
        super().__init__()
        self.fine_attn = fine_attn
        self.coarse_attn = coarse_attn
        self.downsample_factor = downsample_factor
        self.fusion_method = fusion_method

        if fusion_method == "concat":
            self.fuse_proj = nn.Linear(2 * hidden_size, hidden_size)
        elif fusion_method == "gate":
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )

    def forward(
        self,
        hidden_states: torch.Tensor,                  # [B, L, D]
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        B, L, D = hidden_states.shape
        S = self.downsample_factor

        # === Step 1: Fine-scale attention ===
        fine_out, fine_attn, _ = self.fine_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs
        )

        # === Step 2: Downsample for coarse-scale ===
        pad_len = (S - (L % S)) % S
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))  # pad time dim
        L_padded = hidden_states.shape[1]
        hidden_grouped = rearrange(hidden_states, 'b (k s) d -> b k s d', s=S)
        coarse_input = hidden_grouped.mean(dim=2)  # [B, k, D]

        # === Step 3: Coarse attention ===
        coarse_out, coarse_attn, _ = self.coarse_attn(
            hidden_states=coarse_input,
            attention_mask=None,
            output_attentions=output_attentions,
            **kwargs
        )

        # === Step 4: Upsample coarse output ===
        coarse_upsampled = coarse_out.repeat_interleave(S, dim=1)  # [B, L_padded, D]
        if pad_len > 0:
            coarse_upsampled = coarse_upsampled[:, :L, :]  # crop to original length

        # === Step 5: Fuse outputs ===
        if self.fusion_method == "add":
            fused = fine_out + coarse_upsampled
        elif self.fusion_method == "concat":
            fused = self.fuse_proj(torch.cat([fine_out, coarse_upsampled], dim=-1))
        elif self.fusion_method == "gate":
            fusion_input = torch.cat([fine_out, coarse_upsampled], dim=-1)
            gate = self.gate(fusion_input)
            fused = gate * fine_out + (1 - gate) * coarse_upsampled
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return fused, (fine_attn, coarse_attn) if output_attentions else None

