# Modified temporal/modules/encoders/transformer_encoder_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
# Import necessary config types
from temporal.configs.transformer_config import TransformerBlockConfig, AttentionConfig, FeedForwardConfig, TransformerTimeSeriesConfig
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.registry.core import register_module

@register_module("block", "default_encoder")
class TimeSeriesTransformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerBlockConfig, builder: ModuleBuilder):
        super().__init__()
        self.config = config 
        main_config: TransformerTimeSeriesConfig = builder.config 

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: # Return signature adjusted
        """
        Args:
            hidden_states (`torch.FloatTensor`): input shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): mask shape `(batch, 1, tgt_len, src_len)`
            output_attentions (`bool`, *optional*): Whether to return attention probabilities.
        Returns:
             Tuple: (hidden_states, attn_probs)
                    attn_probs is None if output_attentions is False or not returned by attn layer.
        """
        residual = hidden_states
        attn_probs = None 

        # --- Self-Attention ---
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=None,      
            past_key_value=None,        
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            use_cache=False             
        )
        attn_output = attn_outputs[0]
        # Capture attn_probs only if output_attentions is True AND the attn module returned them
        if output_attentions and len(attn_outputs) > 1:
             attn_probs = attn_outputs[1] 

        hidden_states = self.norm1(residual + self.dropout(attn_output))
        # --- End Self-Attention ---

        # --- Feedforward ---
        residual = hidden_states
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm2(residual + self.dropout(ffn_output))
        # --- End Feedforward ---

        # Always return a tuple (hidden_states, attn_probs or None)
        return (hidden_states, attn_probs)
