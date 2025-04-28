import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.utils.utils import expand_mask  # Corrected import path


@register_module("block", "effitime")
class EffiTimeBlockHybridConvFirst(nn.Module):
    """
    EffiTime hybrid block combining:
    - Depthwise + dilated convolution
    - Pointwise conv
    - Global channel & temporal attention (dual SE)
    - Registry-injected attention module
    - LayerNorm + feedback modulation
    """

    def __init__(
        self,
        attention: nn.Module,
        embed_dim: int,
        kernel_size: int = 7,
        dilation: int = 2,
        reduction_ratio: int = 4,
        **kwargs
    ):
        super().__init__()
        self.attn = attention
        self.d_model = embed_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.reduction_ratio = reduction_ratio

        # TLDC
        self.dw_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=kernel_size // 2, groups=embed_dim)
        effective_kernel = (kernel_size - 1) * dilation + 1
        correct_padding = (effective_kernel - 1) // 2
        self.dwd_conv = nn.Conv1d(
            self.d_model, self.d_model,
            kernel_size=kernel_size,
            padding=correct_padding,
            dilation=dilation,
            groups=self.d_model
        )

        self.pw_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

        # SE
        se_dim = max(1, embed_dim // reduction_ratio)
        self.temporal_fc1 = nn.Linear(embed_dim, se_dim)
        self.temporal_fc2 = nn.Linear(se_dim, embed_dim)
        self.channel_fc1 = nn.Linear(embed_dim, se_dim)
        self.channel_fc2 = nn.Linear(se_dim, embed_dim)

        self.norm_attn = nn.LayerNorm(embed_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ):
        B, L, D = hidden_states.shape
        residual = hidden_states

        # [1] TLDC
        x = hidden_states.transpose(1, 2)
        x_local = self.dw_conv(x)
        x_dilated = self.dwd_conv(x_local)
        x_tldc = x_local + x_dilated

        # [2] Pointwise
        x_pw = self.pw_conv(x_tldc)

        # [3] Dual SE
        t_pool = F.adaptive_avg_pool1d(x_pw, 1).squeeze(-1)
        t_attn = self.sigmoid(self.temporal_fc2(self.relu(self.temporal_fc1(t_pool)))).unsqueeze(-1)
        c_pool = F.adaptive_avg_pool1d(x_pw, 1).squeeze(-1)  # [B, D]
        c_attn = self.sigmoid(self.channel_fc2(self.relu(self.channel_fc1(c_pool)))).unsqueeze(-1)
        x_mod = self.sigmoid(x_pw * t_attn * c_attn).transpose(1, 2)

        # [4] Mask
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = expand_mask(attention_mask, tgt_len=L, dtype=hidden_states.dtype)

        # [5] Attention
        attn_out, attn_probs, present_kv = self.attn(
            hidden_states=x_mod,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions
        )

        # [6] Feedback modulation
        out = self.norm_attn(residual + attn_out)

        return out, attn_probs, present_kv
