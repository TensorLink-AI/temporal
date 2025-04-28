import torch
import torch.nn as nn
from typing import Optional, Tuple
# Import the specific block config type hint
from temporal.configs.transformer_config import TransformerBlockConfig, AttentionConfig, FeedForwardConfig
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.registry.core import register_module

@register_module("block", "default_decoder")
class TimeSeriesTransformerDecoderLayer(nn.Module):
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
        # Build cross-attention using the *same* block's attention config
        self.cross_attn = builder.build_attention(config.attention_config) # Assuming same config for cross-attn

        # Ensure config.ffn_config is a FeedForwardConfig object
        if not isinstance(config.ffn_config, FeedForwardConfig):
            raise TypeError(f"Expected ffn_config to be FeedForwardConfig, got {type(config.ffn_config)}")

        # Build FFN using the block's specific FFN config (corrected from feedforward_config)
        self.ffn = builder.build_feedforward(config.ffn_config)

        # Build normalization - assume builder handles config lookup
        self.norm1 = builder.build_normalization()
        self.norm2 = builder.build_normalization()
        self.norm3 = builder.build_normalization()

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
        hidden_states: torch.Tensor,                            # [B, T_dec, D]
        encoder_hidden_states: Optional[torch.Tensor] = None,   # [B, T_enc, D]
        attention_mask: Optional[torch.Tensor] = None,          # [B, 1, T_dec, T_dec]
        encoder_attention_mask: Optional[torch.Tensor] = None,  # [B, 1, T_dec, T_enc]
        past_key_value: Optional[Tuple[Tuple, Tuple]] = None,   # ((self_k, self_v), (cross_k, cross_v))
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[Tuple, Tuple]]]: # Added Optional to return type

        self_kv = past_key_value[0] if past_key_value is not None else None
        cross_kv = past_key_value[1] if past_key_value is not None else None

        # === Self Attention ===
        residual = hidden_states
        # Original code calls self_attn with potentially past_key_value=None which is fine
        self_attn_out, self_attn_probs, present_self_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=self_kv,
            output_attentions=output_attentions
        )
        hidden_states = self.norm1(residual + self.dropout(self_attn_out))

        # Initialize cross attention outputs
        cross_attn_probs = None
        present_cross_kv = None # Ensure it's defined even if cross-attn doesn't run

        # === Cross Attention ===
        # Cross attention should only happen if encoder_hidden_states are provided
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_out, cross_attn_probs, present_cross_kv = self.cross_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_kv,
                output_attentions=output_attentions
            )
            hidden_states = self.norm2(residual + self.dropout(cross_attn_out))

        # === Feedforward ===
        residual = hidden_states
        ffn_out = self.ffn(hidden_states)
        hidden_states = self.norm3(residual + self.dropout(ffn_out))

        # Only return present KV state if it was computed
        present = None
        if present_self_kv is not None or present_cross_kv is not None:
             present = (present_self_kv, present_cross_kv)

        return hidden_states, self_attn_probs, cross_attn_probs, present
