# Modified temporal/modules/decoders/base_decoder_layer.py
import torch
import torch.nn as nn
from typing import Optional, Tuple
# Import the specific block config type hint
from temporal.configs.transformer_config import TransformerBlockConfig, AttentionConfig, FeedForwardConfig, TransformerTimeSeriesConfig # Added main config
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.registry.core import register_module

@register_module("block", "default_decoder")
class TimeSeriesTransformerDecoderLayer(nn.Module):
    def __init__(self, config: TransformerBlockConfig, builder: ModuleBuilder):
        super().__init__()
        self.config = config # Stores the block config
        main_config: TransformerTimeSeriesConfig = builder.config # Get main config from builder

        # --- Resolve Self-Attention Config ---
        # Priority: 1. Block-specific, 2. Global decoder self-attention
        resolved_self_attn_config = config.attention_config
        if resolved_self_attn_config is None:
             if hasattr(main_config, 'attention_config_global') and main_config.attention_config_global:
                 resolved_self_attn_config = main_config.attention_config_global.decoder_attention
             else:
                 raise ValueError("No self-attention configuration found for decoder layer (neither block-specific nor global).")
        if not isinstance(resolved_self_attn_config, AttentionConfig):
             raise TypeError(f"Resolved self-attention config is not an AttentionConfig instance, got {type(resolved_self_attn_config)}")
        # Builder handles embed_dim injection using main_config.hidden_size if needed
        self.self_attn = builder.build_attention(resolved_self_attn_config)
        # --- End Resolve Self-Attention Config ---


        # --- Resolve Cross-Attention Config ---
        # Priority: 1. Block-specific (if defined), 2. Global decoder cross-attention
        resolved_cross_attn_config = config.attention_config # Try block config first (assuming it applies to both if present)
        if resolved_cross_attn_config is None:
             if hasattr(main_config, 'attention_config_global') and main_config.attention_config_global:
                 # Fallback to the specific cross-attention config from global
                 resolved_cross_attn_config = main_config.attention_config_global.decoder_cross_attention
             else:
                 raise ValueError("No cross-attention configuration found for decoder layer (neither block-specific nor global fallback).")
        if not isinstance(resolved_cross_attn_config, AttentionConfig):
             raise TypeError(f"Resolved cross-attention config is not an AttentionConfig instance, got {type(resolved_cross_attn_config)}")
        # Builder handles embed_dim injection using main_config.hidden_size if needed
        self.cross_attn = builder.build_attention(resolved_cross_attn_config)
        # --- End Resolve Cross-Attention Config ---


        # --- Resolve FFN Config ---
        # Priority: 1. Block-specific, 2. Global
        resolved_ffn_config = config.ffn_config
        if resolved_ffn_config is None:
            if hasattr(main_config, 'feedforward_config') and main_config.feedforward_config:
                resolved_ffn_config = main_config.feedforward_config
            else:
                 raise ValueError("No feedforward configuration found for decoder layer (neither block-specific nor global).")
        if not isinstance(resolved_ffn_config, FeedForwardConfig):
             raise TypeError(f"Resolved feedforward configuration is not a FeedForwardConfig instance, got {type(resolved_ffn_config)}")
        self.ffn = builder.build_feedforward(resolved_ffn_config)
        # --- End Resolve FFN Config ---

        # Build normalization - builder uses main_config.norm_config by default
        self.norm1 = builder.build_normalization()
        self.norm2 = builder.build_normalization()
        self.norm3 = builder.build_normalization()

        # Get dropout probability from the main config
        dropout_prob = getattr(main_config, 'hidden_dropout_prob', 0.1) # Default 0.1 documented in config
        self.dropout = nn.Dropout(dropout_prob)


    def forward(
        self,
        hidden_states: torch.Tensor,                            # [B, T_dec, D]
        encoder_hidden_states: Optional[torch.Tensor] = None,   # [B, T_enc, D]
        attention_mask: Optional[torch.Tensor] = None,          # [B, 1, T_dec, T_dec] (causal mask)
        encoder_attention_mask: Optional[torch.Tensor] = None,  # [B, 1, T_dec, T_enc] (padding mask)
        past_key_value: Optional[Tuple[Optional[Tuple], Optional[Tuple]]] = None, # Allow inner tuples to be None
        output_attentions: bool = False,
        use_cache: bool = False, # Standard HF argument
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[Tuple, Tuple]]]:

        residual = hidden_states

        # Initialize outputs
        self_attn_probs = None
        cross_attn_probs = None
        present_self_kv = None
        present_cross_kv = None

        # --- Self Attention ---
        self_attn_past_key_value = past_key_value[0] if past_key_value is not None else None

        self_attn_outputs = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attention_mask=attention_mask,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            # Pass use_cache if the underlying attention module supports it
            # use_cache=use_cache # Assuming BaseMultiHeadAttention handles this implicitly or via kwargs
        )
        self_attn_out = self_attn_outputs[0]
        if output_attentions:
            self_attn_probs = self_attn_outputs[1]
        # Get present_kv only if use_cache is True and the attn module returns it
        if use_cache and len(self_attn_outputs) > 2:
             present_self_kv = self_attn_outputs[2]

        hidden_states = self.norm1(residual + self.dropout(self_attn_out))
        # --- End Self Attention ---


        # --- Cross Attention ---
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[1] if past_key_value is not None else None

            cross_attn_outputs = self.cross_attn(
                query=hidden_states,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                 # Pass use_cache if the underlying attention module supports it
                 # use_cache=use_cache
            )
            cross_attn_out = cross_attn_outputs[0]
            if output_attentions:
                cross_attn_probs = cross_attn_outputs[1]
            # Get present_kv only if use_cache is True and the attn module returns it
            if use_cache and len(cross_attn_outputs) > 2:
                 present_cross_kv = cross_attn_outputs[2]

            hidden_states = self.norm2(residual + self.dropout(cross_attn_out))
        # --- End Cross Attention ---

        # --- Feedforward ---
        residual = hidden_states
        ffn_out = self.ffn(hidden_states)
        hidden_states = self.norm3(residual + self.dropout(ffn_out))
        # --- End Feedforward ---

        # Assemble present key/value tuple only if caching is enabled
        present_key_value = (present_self_kv, present_cross_kv) if use_cache else None

        # Return order matches type hint: (hidden_states, self_attn_probs, cross_attn_probs, present_key_value)
        # Note: Standard HF BaseModelOutput might expect a different order.
        return hidden_states, self_attn_probs, cross_attn_probs, present_key_value
