import torch
import torch.nn as nn
from einops import rearrange
from temporal.registry.core import register_module


@register_module("block", "multiscale")
class MultiScaleTimeAttention(nn.Module):
    """
    Multi-scale time series block.

    Combines:
    - Fine-scale attention
    - Downsampled coarse-scale attention
    - Upsample + fusion

    Arguments:
        attention: fine-scale attention module (e.g., TimeSeriesAttention)
        coarse_attention: coarse-scale attention module
        downsample_factor: number of timesteps to group for coarse level
    """

    def __init__(
        self,
        attention: nn.Module,
        coarse_attention: nn.Module,
        downsample_factor: int = 4,
    ):
        super().__init__()
        self.fine_attn = attention
        self.coarse_attn = coarse_attention
        self.downsample_factor = downsample_factor

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through fine and coarse branches.

        Args:
            hidden_states: [B, L, D]

        Returns:
            output: [B, L, D]
        """
        B, L, D = hidden_states.shape
        factor = self.downsample_factor

        # [1] Fine-scale attention
        fine_out, _, _ = self.fine_attn(hidden_states=hidden_states, **kwargs)  # [B, L, D]

        # [2] Downsample to coarse scale
        pad_len = (factor - L % factor) % factor
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))  # pad time dim
        grouped = rearrange(hidden_states, "b (k s) d -> b k s d", s=factor)
        coarse_in = grouped.mean(dim=2)  # [B, L//factor, D]

        # [3] Coarse attention
        coarse_out, _, _ = self.coarse_attn(hidden_states=coarse_in, **kwargs)  # [B, L//factor, D]

        # [4] Upsample
        upsampled = torch.repeat_interleave(coarse_out, repeats=factor, dim=1)  # [B, L_padded, D]
        if pad_len > 0:
            upsampled = upsampled[:, :-pad_len]  # remove padding

        # [5] Combine
        return fine_out + upsampled
