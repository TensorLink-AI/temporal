import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.utils.utils import expand_mask  # Corrected import path

@register_module("block", "effitime")
class EffiTimeBlockHybridConvFirst(nn.Module):
    """
    An efficient time-series block combining convolutions and attention.
    """

    def __init__(
        self,
        attention: nn.Module,
        embed_dim: int,
        seq_len: int,
        kernel_size: int = 7,
        dilation: int = 2,
        reduction_ratio: int = 4,
        **kwargs,
    ):
        """
        Initializes the EffiTimeBlockHybridConvFirst module.
        """
        super().__init__()
        self.attn = attention
        self.d_model = embed_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.reduction_ratio = reduction_ratio

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

        # Correctly initialize SE blocks based on their respective input dimensions
        temporal_se_hidden_dim = max(1, embed_dim // reduction_ratio)
        self.temporal_fc1 = nn.Linear(embed_dim, temporal_se_hidden_dim)
        self.temporal_fc2 = nn.Linear(temporal_se_hidden_dim, embed_dim)

        channel_se_hidden_dim = max(1, seq_len // reduction_ratio)
        self.channel_fc1 = nn.Linear(seq_len, channel_se_hidden_dim)
        self.channel_fc2 = nn.Linear(channel_se_hidden_dim, seq_len)

        self.norm_attn = nn.LayerNorm(embed_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs the forward pass of the EffiTime block.
        """
        B, L, D = hidden_states.shape
        residual = hidden_states
        is_cross_attention = encoder_hidden_states is not None
        
        x_conv = hidden_states.transpose(1, 2)
        x_local = self.dw_conv(x_conv)
        x_dilated = self.dwd_conv(x_local)
        x_tldc = x_local + x_dilated
        x_pointwise = self.pw_conv(x_tldc)

        temporal_pooled = F.adaptive_avg_pool1d(x_pointwise, 1).squeeze(-1)
        temporal_attention = self.sigmoid(self.temporal_fc2(self.relu(self.temporal_fc1(temporal_pooled)))).unsqueeze(-1)

        channel_pooled = F.adaptive_avg_pool1d(x_pointwise.transpose(1, 2), 1).squeeze(-1)
        channel_attention = self.sigmoid(self.channel_fc2(self.relu(self.channel_fc1(channel_pooled)))).unsqueeze(-1)

        modulated_output = self.sigmoid(x_pointwise * temporal_attention * channel_attention.transpose(1,2)).transpose(1, 2)

        attn_mask = encoder_attention_mask if is_cross_attention else attention_mask
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = expand_mask(attn_mask, tgt_len=L, dtype=modulated_output.dtype)

        attention_output, attention_probs, present_key_value = self.attn(
            hidden_states=modulated_output,
            attention_mask=attn_mask,
            key_value_states=encoder_hidden_states,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            head_mask=head_mask,
        )

        final_output = self.norm_attn(residual + attention_output)

        return final_output, attention_probs, present_key_value
