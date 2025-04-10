import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention
from temporal.modules.utils import expand_mask


class EffiTimeBlockHybridConvFirst(nn.Module):
    """
    A hybrid time-series block combining:
    - TLDC (DW + DW-D convolution)
    - Pointwise conv
    - GTVA (global temporal & channel attention)
    - BaseMultiHeadAttention on top of enhanced features
    - Feedback modulation (element-wise) with LayerNorm
    Inspired by: https://arxiv.org/pdf/2411.04669
    """

    def __init__(self, config, attn_module: BaseMultiHeadAttention):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.kernel_size = config.kernel_size
        self.dilation = config.dilation
        self.reduction_ratio = config.reduction_ratio

        self.attn = attn_module

        # TLDC: DW + DW-D convs
        self.dw_conv = nn.Conv1d(
            self.d_model, self.d_model,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.d_model
        )
        self.dwd_conv = nn.Conv1d(
            self.d_model, self.d_model,
            kernel_size=(self.kernel_size + 1) // self.dilation,
            padding=self.dilation,
            dilation=self.dilation,
            groups=self.d_model
        )

        # Pointwise conv (simplified IVGC)
        self.pw_conv = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)

        # GTVA (dual SE)
        se_dim = max(1, self.d_model // self.reduction_ratio)
        self.temporal_fc1 = nn.Linear(self.d_model, se_dim)
        self.temporal_fc2 = nn.Linear(se_dim, self.d_model)

        self.channel_fc1 = nn.Linear(self.d_model, se_dim)
        self.channel_fc2 = nn.Linear(se_dim, self.d_model)

        # LayerNorms
        self.norm_conv = nn.LayerNorm(self.d_model)
        self.norm_attn = nn.LayerNorm(self.d_model)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        hidden_states: torch.Tensor,                  # [B, L, D]
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] or [B, 1, L, S]
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        B, L, D = hidden_states.shape
        residual = hidden_states

        # === [1] TLDC ===
        x = hidden_states.transpose(1, 2)       # (B, D, L)
        x_local = self.dw_conv(x)
        x_dilated = self.dwd_conv(x_local)
        x_tldc = x_local + x_dilated            # (B, D, L)

        # === [2] Pointwise Conv ===
        x_pw = self.pw_conv(x_tldc)             # (B, D, L)

        # === [3] GTVA ===
        t_pool = F.adaptive_avg_pool1d(x_pw, 1).squeeze(-1)  # (B, D)
        t_attn = self.sigmoid(self.temporal_fc2(self.relu(self.temporal_fc1(t_pool))))  # (B, D)
        t_attn = t_attn.unsqueeze(-1)  # (B, D, 1)

        c_pool = F.adaptive_avg_pool1d(x_pw.transpose(1, 2), 1).squeeze(-1)  # (B, L)
        c_attn = self.sigmoid(self.channel_fc2(self.relu(self.channel_fc1(c_pool))))  # (B, D)
        c_attn = c_attn.unsqueeze(-1)  # (B, D, 1)

        x_mod = self.sigmoid(x_pw * t_attn * c_attn)  # (B, D, L)
        x_mod = x_mod.transpose(1, 2)                 # (B, L, D)

        # === [4] Expand attention mask if needed ===
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = expand_mask(attention_mask, tgt_len=L, dtype=hidden_states.dtype)

        # === [5] Multi-head Attention ===
        attn_out, attn_probs, present_kv = self.attn(
            hidden_states=x_mod,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions
        )

        # === [6] Feedback Fusion ===
        out = self.norm_attn(residual * attn_out)

        return out, attn_probs, present_kv
