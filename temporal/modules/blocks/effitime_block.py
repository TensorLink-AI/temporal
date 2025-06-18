import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.utils.utils import expand_mask

@register_module("block", "effitime")
class EffiTimeBlockHybridConvFirst(nn.Module):
    """
    An efficient time-series block combining convolutions and attention.
    This version removes the sequence-length-dependent channel SE block
    for robust handling of variable sequence lengths.
    """

    def __init__(
        self,
        attention: nn.Module,
        embed_dim: int,
        kernel_size: int = 7,
        dilation: int = 2,
        reduction_ratio: int = 4,
        **kwargs,
    ):
        """
        Initializes the EffiTimeBlockHybridConvFirst module.

        Args:
            attention (nn.Module): An instantiated attention module.
            embed_dim (int): The embedding dimension of the input and output.
            kernel_size (int): The kernel size for the depthwise convolutions.
            dilation (int): The dilation factor for the second depthwise convolution.
            reduction_ratio (int): The reduction ratio for the SE block's hidden layer.
            **kwargs: Additional keyword arguments.
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

        # Squeeze-and-Excitation block for channel-wise attention
        se_hidden_dim = max(1, embed_dim // reduction_ratio)
        self.se_fc1 = nn.Linear(embed_dim, se_hidden_dim)
        self.se_fc2 = nn.Linear(se_hidden_dim, embed_dim)

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

        # Squeeze-and-Excitation
        pooled = F.adaptive_avg_pool1d(x_pointwise, 1).squeeze(-1)
        se_attention = self.sigmoid(self.se_fc2(self.relu(self.se_fc1(pooled)))).unsqueeze(-1)

        # Apply SE and reshape back for attention
        modulated_output = self.sigmoid(x_pointwise * se_attention).transpose(1, 2)

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
