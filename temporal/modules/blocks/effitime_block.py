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

    This block implements the EffiTime architecture, which is designed for
    efficient and effective time-series forecasting. It integrates several
    components in a specific sequence:
    1.  **Temporal Local Dilated Convolution (TLDC):** A combination of
        depthwise and dilated depthwise convolutions to capture local
        temporal patterns at multiple scales.
    2.  **Pointwise Convolution:** A 1x1 convolution to mix channel information.
    3.  **Dual Squeeze-and-Excitation (SE):** Separate SE blocks for temporal
        and channel-wise attention to recalibrate feature maps.
    4.  **Injected Attention Module:** A configurable attention mechanism,
        injected via the registry, to capture global dependencies.
    5.  **Feedback Modulation:** A residual connection and LayerNorm for stable
        training and effective feature combination.

    Attributes:
        attn (nn.Module): The injected attention module.
        d_model (int): The embedding dimension.
        dw_conv (nn.Conv1d): The depthwise convolution layer.
        dwd_conv (nn.Conv1d): The dilated depthwise convolution layer.
        pw_conv (nn.Conv1d): The pointwise convolution layer.
        temporal_fc1 (nn.Linear): First linear layer for temporal SE.
        temporal_fc2 (nn.Linear): Second linear layer for temporal SE.
        channel_fc1 (nn.Linear): First linear layer for channel SE.
        channel_fc2 (nn.Linear): Second linear layer for channel SE.
        norm_attn (nn.LayerNorm): The final layer normalization.
    """

    def __init__(
        self,
        attention: nn.Module,
        embed_dim: int,
        kernel_size: int = 7,     # Kernel size for convolutions
        dilation: int = 2,        # Dilation for the second DW conv
        reduction_ratio: int = 4, # Reduction ratio for SE blocks
        **kwargs,                 # To absorb any other args from builder
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

        # Temporal Local Dilated Convolution (TLDC)
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

        self.pw_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1) # Pointwise

        # Squeeze-and-Excitation (SE) blocks
        se_hidden_dim = max(1, embed_dim // reduction_ratio)
        self.temporal_fc1 = nn.Linear(embed_dim, se_hidden_dim)
        self.temporal_fc2 = nn.Linear(se_hidden_dim, embed_dim)
        self.channel_fc1 = nn.Linear(embed_dim, se_hidden_dim)
        self.channel_fc2 = nn.Linear(se_hidden_dim, embed_dim)

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

        Args:
            hidden_states (torch.Tensor): The input tensor of shape `[B, L, D]`.
            attention_mask (Optional[torch.Tensor]): An optional mask for the
                attention module.
            encoder_hidden_states (Optional[torch.Tensor]): Hidden states from an
                encoder, used for cross-attention.
            encoder_attention_mask (Optional[torch.Tensor]): Mask for the encoder's
                hidden states.
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Cached
                key-value states for autoregressive decoding.
            output_attentions (bool): Whether to return attention probabilities.
            use_cache (bool): Whether to use caching for the key and value states.
            head_mask (Optional[torch.Tensor]): The mask for attention heads.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                - The output tensor of shape `[B, L, D]`.
                - The attention probabilities from the attention module, if requested.
                - The updated key-value cache, if applicable.
        """
        B, L, D = hidden_states.shape
        residual = hidden_states
        is_cross_attention = encoder_hidden_states is not None
        
        # Reshape for 1D convolutions: [B, L, D] -> [B, D, L]
        x_conv = hidden_states.transpose(1, 2)

        # Step 1: Temporal Local Dilated Convolution (TLDC)
        x_local = self.dw_conv(x_conv)
        x_dilated = self.dwd_conv(x_local)
        x_tldc = x_local + x_dilated

        # Step 2: Pointwise Convolution
        x_pointwise = self.pw_conv(x_tldc)

        # Step 3: Dual Squeeze-and-Excitation (SE)
        temporal_pooled = F.adaptive_avg_pool1d(x_pointwise, 1).squeeze(-1)
        temporal_attention = self.sigmoid(self.temporal_fc2(self.relu(self.temporal_fc1(temporal_pooled)))).unsqueeze(-1)

        channel_pooled = F.adaptive_avg_pool1d(x_pointwise.transpose(1, 2), 1).squeeze(-1) # Pool across time
        channel_attention = self.sigmoid(self.channel_fc2(self.relu(self.channel_fc1(channel_pooled)))).unsqueeze(-1)

        # Apply SE and reshape back for attention: [B, D, L] -> [B, L, D]
        modulated_output = self.sigmoid(x_pointwise * temporal_attention * channel_attention.transpose(1,2)).transpose(1, 2)

        # Step 4: Prepare attention mask
        attn_mask = encoder_attention_mask if is_cross_attention else attention_mask
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = expand_mask(attn_mask, tgt_len=L, dtype=modulated_output.dtype)

        # Step 5: Global Attention
        attention_output, attention_probs, present_key_value = self.attn(
            hidden_states=modulated_output,
            attention_mask=attn_mask,
            key_value_states=encoder_hidden_states,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            head_mask=head_mask,
        )

        # Step 6: Final Feedback Modulation and Normalization
        final_output = self.norm_attn(residual + attention_output)

        return final_output, attention_probs, present_key_value
