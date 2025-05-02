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
        resolved_self_attn_config = config.attention_config
        if resolved_self_attn_config is None:
             # DEPRECATED: Fallback to global config is less flexible than per-block config
             # Consider removing global attention_blocks from TransformerTimeSeriesConfig later
             if hasattr(main_config, 'attention_blocks') and main_config.attention_blocks:
                 resolved_self_attn_config = main_config.attention_blocks.decoder_attention
             else:
                 raise ValueError("No self-attention configuration found for decoder layer.")
        if not isinstance(resolved_self_attn_config, AttentionConfig):
             raise TypeError(f"Resolved self-attention config is not AttentionConfig: {type(resolved_self_attn_config)}")
        self.self_attn = builder.build_attention(resolved_self_attn_config, is_decoder=True, is_cross_attention=False)
        # --- End Resolve Self-Attention Config ---

        # --- Resolve Cross-Attention Config ---
        # Use the same resolved config as self-attention by default if block-specific
        # Or fallback to specific global cross-attention config
        resolved_cross_attn_config = config.attention_config # Assume block config applies to both if provided
        if resolved_cross_attn_config is None:
             if hasattr(main_config, 'attention_blocks') and main_config.attention_blocks:
                 resolved_cross_attn_config = main_config.attention_blocks.decoder_cross_attention
             else:
                 raise ValueError("No cross-attention configuration found for decoder layer.")
        if not isinstance(resolved_cross_attn_config, AttentionConfig):
             raise TypeError(f"Resolved cross-attention config is not AttentionConfig: {type(resolved_cross_attn_config)}")
        self.cross_attn = builder.build_attention(resolved_cross_attn_config, is_decoder=True, is_cross_attention=True)
        # --- End Resolve Cross-Attention Config ---

        # --- Resolve FFN Config ---
        resolved_ffn_config = config.ffn_config
        if resolved_ffn_config is None:
            if hasattr(main_config, 'feedforward_config') and main_config.feedforward_config:
                resolved_ffn_config = main_config.feedforward_config
            else:
                 raise ValueError("No feedforward configuration found for decoder layer.")
        if not isinstance(resolved_ffn_config, FeedForwardConfig):
             raise TypeError(f"Resolved FFN config is not FeedForwardConfig: {type(resolved_ffn_config)}")
        self.ffn = builder.build_feedforward(resolved_ffn_config)
        # --- End Resolve FFN Config ---

        self.norm1 = builder.build_normalization()
        self.norm2 = builder.build_normalization()
        self.norm3 = builder.build_normalization()
        dropout_prob = getattr(main_config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,                            # [B, T_dec, D]
        encoder_hidden_states: Optional[torch.Tensor] = None,   # [B, T_enc, D]
        attention_mask: Optional[torch.Tensor] = None,          # [B, 1, T_dec, T_dec] (causal mask for self-attn)
        encoder_attention_mask: Optional[torch.Tensor] = None,  # [B, 1, T_dec, T_enc] (padding mask for cross-attn)
        past_key_value: Optional[Tuple[Optional[Tuple], Optional[Tuple]]] = None, # ((past_self_k, past_self_v), (past_cross_k, past_cross_v))
        output_attentions: bool = False,
        use_cache: bool = False, # Standard HF argument
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[Tuple, Tuple]]]:

        residual = hidden_states
        self_attn_probs = None
        cross_attn_probs = None
        present_self_kv = None
        present_cross_kv = None

        # --- Self Attention ---
        self_attn_past_key_value = past_key_value[0] if past_key_value is not None else None

        # Correct call using BaseMultiHeadAttention signature
        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=None, # Self-attention uses hidden_states for K/V
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            use_cache=use_cache 
        )
        self_attn_out = self_attn_outputs[0]
        if output_attentions:
            self_attn_probs = self_attn_outputs[1]
        if use_cache:
             present_self_kv = self_attn_outputs[2] # Expects (output, probs, present_kv)

        hidden_states = self.norm1(residual + self.dropout(self_attn_out))
        # --- End Self Attention ---

        # --- Cross Attention ---
        # Optional: Only perform cross-attention if encoder context exists
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[1] if past_key_value is not None else None

            # Correct call using BaseMultiHeadAttention signature
            cross_attn_outputs = self.cross_attn(
                hidden_states=hidden_states,           # Query is the current decoder state
                key_value_states=encoder_hidden_states, # K/V are from the encoder
                past_key_value=cross_attn_past_key_value,
                attention_mask=encoder_attention_mask, # Use encoder mask here
                output_attentions=output_attentions,
                use_cache=use_cache 
            )
            cross_attn_out = cross_attn_outputs[0]
            if output_attentions:
                cross_attn_probs = cross_attn_outputs[1]
            if use_cache:
                 present_cross_kv = cross_attn_outputs[2]

            hidden_states = self.norm2(residual + self.dropout(cross_attn_out))
        # --- End Cross Attention ---

        # --- Feedforward ---
        residual = hidden_states
        ffn_out = self.ffn(hidden_states)
        hidden_states = self.norm3(residual + self.dropout(ffn_out))
        # --- End Feedforward ---

        present_key_value = (present_self_kv, present_cross_kv) if use_cache else None

        # Return order: hidden_states, self_attn_probs, cross_attn_probs, present_key_value
        return hidden_states, self_attn_probs, cross_attn_probs, present_key_value
