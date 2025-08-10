# Modified temporal/modules/encoders/transformer_encoder_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
# Import necessary config types individually
from temporal.configs.transformer_block_config import TransformerBlockConfig
from temporal.configs.attention_config import AttentionConfig
from temporal.configs.feedforward_config import FeedForwardConfig
from temporal.configs.normalization_config import NormalizationConfig
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.registry.core import register_module
from temporal.models.outputs import EncoderLayerOutput

@register_module("block", "default_encoder")
class TimeSeriesTransformerEncoderLayer(nn.Module):
    """A standard Transformer encoder layer for time series.

    This module implements a single layer of a Transformer encoder, which is a
    fundamental building block for sequence-to-sequence models in time series
    forecasting. It consists of two main components:
    1.  A self-attention mechanism to process the input sequence.
    2.  A feed-forward network (FFN).

    Each component is followed by a residual connection and layer normalization.
    The specific implementations of attention, FFN, and normalization are
    dynamically built based on the provided configuration.

    Attributes:

        config (TransformerBlockConfig): The configuration for this specific block.
        self_attn (nn.Module): The self-attention module.
        ffn (nn.Module): The feed-forward network.
        norm1 (nn.Module): Layer normalization after self-attention.
        norm2 (nn.Module): Layer normalization after the FFN.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, config: TransformerBlockConfig, builder: ModuleBuilder, **kwargs):
        """Initializes the TimeSeriesTransformerEncoderLayer.

        Args:

            config (TransformerBlockConfig): The configuration specific to this encoder
                layer, defining the types of attention and FFN to be used.
            builder (ModuleBuilder): A helper class that constructs the sub-modules
                (attention, FFN, normalization) based on the main model configuration.
            **kwargs: Catches unused arguments to ensure backward compatibility.
        """
        super().__init__()
        self.config = config
        main_config: TransformerTimeSeriesConfig = builder.config

        # --- Resolve Attention Config ---
        resolved_attn_config = config.attention_config
        if resolved_attn_config is None:
             # Fallback to a global config if a block-specific one isn't provided
             if hasattr(main_config, 'attention_config_global') and main_config.attention_config_global:
                 resolved_attn_config = main_config.attention_config_global.encoder_attention
             else:
                 raise ValueError("No attention configuration found for encoder layer (neither block-specific nor global).")
        if not isinstance(resolved_attn_config, AttentionConfig):
             raise TypeError(f"Resolved attention configuration is not an AttentionConfig instance, got {type(resolved_attn_config)}")
        self.self_attn = builder.build_attention(resolved_attn_config)
        # --- End Resolve Attention Config ---

        # --- Resolve FFN Config ---
        resolved_ffn_config = config.ffn_config
        if resolved_ffn_config is None:
            # Fallback to a global config if a block-specific one isn't provided
            if hasattr(main_config, 'feedforward_config') and main_config.feedforward_config:
                resolved_ffn_config = main_config.feedforward_config
            else:
                 raise ValueError("No feedforward configuration found for encoder layer (neither block-specific nor global).")
        if not isinstance(resolved_ffn_config, FeedForwardConfig):
             raise TypeError(f"Resolved feedforward configuration is not a FeedForwardConfig instance, got {type(resolved_ffn_config)}")
        self.ffn = builder.build_feedforward(resolved_ffn_config)
        # --- End Resolve FFN Config ---

        # --- Resolve Normalization Config ---
        resolved_norm_config = config.normalization_config
        if resolved_norm_config is None:
            # Fallback to the new global layer_norm_config
            if hasattr(main_config, 'layer_norm_config') and main_config.layer_norm_config:
                resolved_norm_config = main_config.layer_norm_config
            else:
                 raise ValueError("No layer normalization configuration found for encoder layer (neither block-specific nor global).")
        if not isinstance(resolved_norm_config, NormalizationConfig):
             raise TypeError(f"Resolved normalization configuration is not a NormalizationConfig instance, got {type(resolved_norm_config)}")
        self.norm1 = builder.build_normalization(resolved_norm_config)
        self.norm2 = builder.build_normalization(resolved_norm_config)
        # --- End Resolve Normalization Config ---
        
        dropout_prob = getattr(main_config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        x_raw: Optional[torch.Tensor] = None,
    ) -> EncoderLayerOutput:
        """Performs the forward pass of the encoder layer.

        Args:

            hidden_states (torch.Tensor): The input to the layer of shape
                `(batch, seq_len, embed_dim)`.
            attention_mask (Optional[torch.Tensor]): A mask to prevent attention
                to padding tokens, shape `(batch, 1, seq_len, seq_len)`.
            output_attentions (Optional[bool]): Whether to return the attention
                probabilities.
            x_raw (Optional[torch.Tensor]): Raw input for de-stationary attention.

        Returns:
        
             EncoderLayerOutput: An object containing the output hidden states,
                optional attention weights, and optional auxiliary loss.
        """
        residual = hidden_states
        attention_weights = None
        aux_loss = None

        # --- Self-Attention Block ---
        # The attention block itself may return an auxiliary loss (e.g. in MoE)
        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=None,      # Not used in self-attention
            past_key_value=None,        # Not used in encoder
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            use_cache=False,             # Not used in encoder
            x_raw=x_raw,
        )
        attention_output = attention_outputs[0]
        # Capture attention probabilities if requested and returned
        if output_attentions and len(attention_outputs) > 1:
             attention_weights = attention_outputs[1]

        hidden_states = self.norm1(residual + self.dropout(attention_output))
        # --- End Self-Attention Block ---

        # --- Feedforward Block ---
        residual = hidden_states
        # The FFN may return an auxiliary loss (e.g. in MoE)
        ffn_outputs = self.ffn(hidden_states)
        hidden_states = self.norm2(residual + self.dropout(ffn_outputs[0]))
        if len(ffn_outputs) > 1 and ffn_outputs[1] is not None:
            aux_loss = ffn_outputs[1]
        # --- End Feedforward Block ---

        return EncoderLayerOutput(
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            aux_loss=aux_loss
        )
