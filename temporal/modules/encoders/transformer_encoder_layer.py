import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
# Import the specific block config type hint
from temporal.configs.transformer_config import TransformerBlockConfig, AttentionConfig, FeedForwardConfig
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.registry.core import register_module

@register_module("block", "default_encoder")
class TimeSeriesTransformerEncoderLayer(nn.Module):
    # Assume 'config' passed is the specific TransformerBlockConfig for this layer
    # Builder provides access to main config and helper methods
    def __init__(self, config: TransformerBlockConfig, builder: ModuleBuilder):
        super().__init__()
        self.config = config # Stores the block config

        # Ensure config.attention_config is an AttentionConfig object
        if not isinstance(config.attention_config, AttentionConfig):
             raise TypeError(f"Expected attention_config to be AttentionConfig, got {type(config.attention_config)}")

        # Build self-attention using the block's specific attention config
        self.self_attn = builder.build_attention(config.attention_config)

        # Ensure config.ffn_config is a FeedForwardConfig object
        if not isinstance(config.ffn_config, FeedForwardConfig):
            raise TypeError(f"Expected ffn_config to be FeedForwardConfig, got {type(config.ffn_config)}")

        # Build FFN using the block's specific FFN config
        self.ffn = builder.build_feedforward(config.ffn_config)

        # Build normalization - assume builder handles config lookup (e.g., from main config)
        self.norm1 = builder.build_normalization()
        self.norm2 = builder.build_normalization()

        # Get dropout probability from the main config via the builder
        # Assuming builder has 'main_config' attribute holding TransformerTimeSeriesConfig
        dropout_prob = 0.1 # Default dropout
        if hasattr(builder, 'main_config') and hasattr(builder.main_config, 'hidden_dropout_prob'):
            dropout_prob = builder.main_config.hidden_dropout_prob
        elif hasattr(config, 'dropout') and config.dropout is not None: # Fallback to block config dropout if exists
             dropout_prob = config.dropout


        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # === Self-Attention ===
        residual = hidden_states
        # Original forward logic uses dropout after residual connection, let's keep it
        attn_output_tensor, attn_probs, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # Apply dropout *after* attention, before adding residual and norm
        hidden_states = self.norm1(residual + self.dropout(attn_output_tensor))

        # === Feedforward ===
        residual = hidden_states
        ffn_output = self.ffn(hidden_states)
        # Apply dropout *after* FFN, before adding residual and norm
        hidden_states = self.norm2(residual + self.dropout(ffn_output))

        # The original returned (hidden_states, attn_probs) or (hidden_states,)
        # Let's match that exactly.
        if output_attentions:
            return (hidden_states, attn_probs)
        else:
             return (hidden_states,)
