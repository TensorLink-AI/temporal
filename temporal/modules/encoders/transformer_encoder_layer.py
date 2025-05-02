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
        resolved_attn_config = config.attention_config 
        if resolved_attn_config is None:
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
            if hasattr(main_config, 'feedforward_config') and main_config.feedforward_config:
                resolved_ffn_config = main_config.feedforward_config
            else:
                 raise ValueError("No feedforward configuration found for encoder layer (neither block-specific nor global).")
        if not isinstance(resolved_ffn_config, FeedForwardConfig):
             raise TypeError(f"Resolved feedforward configuration is not a FeedForwardConfig instance, got {type(resolved_ffn_config)}")
        self.ffn = builder.build_feedforward(resolved_ffn_config)
        # --- End Resolve FFN Config ---

        self.norm1 = builder.build_normalization()
        self.norm2 = builder.build_normalization()
        dropout_prob = getattr(main_config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        # Encoder layer doesn't typically use past_kv or encoder context
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: # Return signature matches expected tuple
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                 Note: BaseMultiHeadAttention handles expansion if input is [B, S]
            output_attentions (`bool`, *optional*):
                Whether or not to return the attention probabilities.
        """
        residual = hidden_states
        attn_probs = None # Initialize

        # --- Self-Attention ---
        # Call attention layer with the expected signature: (hidden_states, key_value=None, past_kv=None, mask, ...)
        # Encoder self-attention: hidden_states acts as Q, K, V. key_value_states is None.
        attn_outputs = self.self_attn(
            hidden_states=hidden_states, # Pass the input as hidden_states
            key_value_states=None,      # Explicitly None for self-attention
            past_key_value=None,        # Encoder doesn't use caching
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            use_cache=False             # Explicitly False for encoder
        )
        attn_output = attn_outputs[0]
        if output_attentions:
             attn_probs = attn_outputs[1] # Assuming attn_probs is the second element

        hidden_states = self.norm1(residual + self.dropout(attn_output))
        # --- End Self-Attention ---

        # --- Feedforward ---
        residual = hidden_states
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm2(residual + self.dropout(ffn_output))
        # --- End Feedforward ---

        # The layer returns a tuple: (hidden_states, attention_probs (optional))
        outputs = (hidden_states,)
        if output_attentions and attn_probs is not None:
            outputs = outputs + (attn_probs,)
        elif output_attentions:
            # If output_attentions is True but attn_probs is None (e.g., FlashAttention)
            # Add None placeholder to maintain tuple structure if expected by caller
             outputs = outputs + (None,)
        
        # Return only hidden_states if not output_attentions
        return outputs[0] if not output_attentions else outputs
