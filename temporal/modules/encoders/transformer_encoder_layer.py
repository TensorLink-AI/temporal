# Modified temporal/modules/encoders/transformer_encoder_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple # Added Tuple for return type consistency
# Import necessary config types
from temporal.configs.transformer_config import TransformerBlockConfig, AttentionConfig, FeedForwardConfig, TransformerTimeSeriesConfig
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.registry.core import register_module

@register_module("block", "default_encoder")
class TimeSeriesTransformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerBlockConfig, builder: ModuleBuilder):
        super().__init__()
        self.config = config # Stores the block config
        main_config: TransformerTimeSeriesConfig = builder.config # Get main config from builder

        # --- Resolve Attention Config ---
        # Priority: 1. Block-specific config, 2. Global encoder config
        resolved_attn_config = config.attention_config # Try block specific first

        if resolved_attn_config is None:
             # Fallback to the global config's encoder attention
             if hasattr(main_config, 'attention_config_global') and main_config.attention_config_global:
                 resolved_attn_config = main_config.attention_config_global.encoder_attention
             else:
                 raise ValueError("No attention configuration found for encoder layer (neither block-specific nor global).")

        # Ensure we have a valid AttentionConfig object now
        if not isinstance(resolved_attn_config, AttentionConfig):
             raise TypeError(f"Resolved attention configuration is not an AttentionConfig instance, got {type(resolved_attn_config)}")

        # Build self-attention using the resolved config (builder handles embed_dim)
        self.self_attn = builder.build_attention(resolved_attn_config)
        # --- End Resolve Attention Config ---


        # --- Resolve FFN Config ---
        # Priority: 1. Block-specific config, 2. Global config
        resolved_ffn_config = config.ffn_config
        if resolved_ffn_config is None:
            # Fallback to the main config's feedforward config
            if hasattr(main_config, 'feedforward_config') and main_config.feedforward_config:
                resolved_ffn_config = main_config.feedforward_config
            else:
                 raise ValueError("No feedforward configuration found for encoder layer (neither block-specific nor global).")

        # Ensure we have a valid FeedForwardConfig object
        if not isinstance(resolved_ffn_config, FeedForwardConfig):
             raise TypeError(f"Resolved feedforward configuration is not a FeedForwardConfig instance, got {type(resolved_ffn_config)}")

        # Build FFN using the resolved config
        self.ffn = builder.build_feedforward(resolved_ffn_config)
        # --- End Resolve FFN Config ---


        # Build normalization - builder uses main_config.norm_config by default
        self.norm1 = builder.build_normalization()
        self.norm2 = builder.build_normalization()

        # Get dropout probability from the main config
        dropout_prob = getattr(main_config, 'hidden_dropout_prob', 0.1) # Default 0.1

        self.dropout = nn.Dropout(dropout_prob)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: # Ensure return type is a tuple
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        attn_probs = None # Initialize

        # --- Self-Attention ---
        # Assuming the attention module returns (output, weights, past_kv)
        # For encoder, past_kv is usually None or not used.
        attn_outputs = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        if output_attentions:
             attn_probs = attn_outputs[1]

        hidden_states = self.norm1(residual + self.dropout(attn_output))
        # --- End Self-Attention ---


        # --- Feedforward ---
        residual = hidden_states
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm2(residual + self.dropout(ffn_output))
        # --- End Feedforward ---

        # The layer should return a tuple: (hidden_states, attention_probs (optional))
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (attn_probs,)

        return outputs
