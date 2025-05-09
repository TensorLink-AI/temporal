# Modified temporal/modules/decoders/base_decoder_layer.py
import torch
import torch.nn as nn
from typing import Optional, Tuple
# Import the specific block config type hint
from temporal.configs.transformer_config import TransformerBlockConfig, AttentionConfig, FeedForwardConfig, TransformerTimeSeriesConfig # Added main config
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.registry.core import register_module
import copy # Import copy for deepcopy

@register_module("block", "default_decoder")
class TimeSeriesTransformerDecoderLayer(nn.Module):
    def __init__(self, config: TransformerBlockConfig, builder: ModuleBuilder):
        super().__init__()
        self.config = config # Stores the block config
        main_config: TransformerTimeSeriesConfig = builder.config # Get main config from builder
        self.is_encoder_decoder = main_config.architecture.layout == "encoder-decoder"

        # --- Resolve Self-Attention Config ---
        # Self-attention always uses config.attention_config
        resolved_self_attn_config = config.attention_config
        if resolved_self_attn_config is None:
             # DEPRECATED fallback logic for self-attention (should ideally not be hit if config is well-defined)
             if hasattr(main_config, 'attention_blocks') and main_config.attention_blocks and hasattr(main_config.attention_blocks, 'decoder_attention'):
                 print("Warning: Decoder self-attention using DEPRECATED fallback to main_config.attention_blocks.decoder_attention.")
                 resolved_self_attn_config = main_config.attention_blocks.decoder_attention
             else:
                 # If still None, default to a base AttentionConfig or raise error
                 print("Warning: No self-attention config found in TransformerBlockConfig, using default AttentionConfig for decoder self-attention.")
                 resolved_self_attn_config = AttentionConfig() # Default if truly missing
        
        if not isinstance(resolved_self_attn_config, AttentionConfig):
             raise TypeError(f"Resolved self-attention config is not AttentionConfig: {type(resolved_self_attn_config)}")
        
        # Prepare build config for self-attention (always is_decoder=True, is_cross_attention=False)
        self_attn_build_config = copy.deepcopy(resolved_self_attn_config)
        self_attn_build_config.kwargs = self_attn_build_config.kwargs or {}
        self_attn_build_config.kwargs['is_decoder'] = True
        self_attn_build_config.kwargs['is_cross_attention'] = False
        self.self_attn = builder.build_attention(self_attn_build_config)
        # --- End Resolve Self-Attention Config ---

        # --- Conditionally Resolve Cross-Attention Config ---
        self.cross_attn = None
        if self.is_encoder_decoder:
            # 1. Prefer specific cross_attention_config from the block config
            resolved_cross_attn_config = config.cross_attention_config
            
            # 2. If not provided, fall back to the block's general attention_config
            if resolved_cross_attn_config is None:
                resolved_cross_attn_config = config.attention_config

            # 3. DEPRECATED fallback if still no suitable config (should ideally not be hit)
            if resolved_cross_attn_config is None:
                if hasattr(main_config, 'attention_blocks') and main_config.attention_blocks and hasattr(main_config.attention_blocks, 'decoder_cross_attention'):
                    print("Warning: Decoder cross-attention using DEPRECATED fallback to main_config.attention_blocks.decoder_cross_attention.")
                    resolved_cross_attn_config = main_config.attention_blocks.decoder_cross_attention
                else:
                    # If truly no config, default to a base AttentionConfig or raise error
                    print("Warning: No cross-attention config found, using default AttentionConfig for decoder cross-attention.")
                    resolved_cross_attn_config = AttentionConfig()
            
            if not isinstance(resolved_cross_attn_config, AttentionConfig):
                 raise TypeError(f"Resolved cross-attention config is not AttentionConfig: {type(resolved_cross_attn_config)}")
            
            # Prepare build config for cross-attention (always is_decoder=True, is_cross_attention=True)
            cross_attn_build_config = copy.deepcopy(resolved_cross_attn_config)
            cross_attn_build_config.kwargs = cross_attn_build_config.kwargs or {}
            cross_attn_build_config.kwargs['is_decoder'] = True
            cross_attn_build_config.kwargs['is_cross_attention'] = True
            self.cross_attn = builder.build_attention(cross_attn_build_config)
        # --- End Resolve Cross-Attention Config ---

        # --- Resolve FFN Config ---
        resolved_ffn_config = config.ffn_config
        if resolved_ffn_config is None:
            # DEPRECATED fallback for FFN
            if hasattr(main_config, 'feedforward_config') and main_config.feedforward_config:
                print("Warning: Decoder FFN using DEPRECATED fallback to main_config.feedforward_config.")
                resolved_ffn_config = main_config.feedforward_config
            else:
                 print("Warning: No FFN config found in TransformerBlockConfig, using default FeedForwardConfig for decoder.")
                 resolved_ffn_config = FeedForwardConfig()
        if not isinstance(resolved_ffn_config, FeedForwardConfig):
             raise TypeError(f"Resolved FFN config is not FeedForwardConfig: {type(resolved_ffn_config)}")
        self.ffn = builder.build_feedforward(resolved_ffn_config)
        # --- End Resolve FFN Config ---

        self.norm1 = builder.build_normalization()
        self.norm2 = builder.build_normalization() if self.is_encoder_decoder and self.cross_attn is not None else None 
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
        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=None, # Self-attention does not use key_value_states
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            use_cache=use_cache 
        )
        self_attn_out = self_attn_outputs[0]
        if output_attentions: self_attn_probs = self_attn_outputs[1]
        if use_cache: present_self_kv = self_attn_outputs[2] if len(self_attn_outputs) > 2 else None 
        hidden_states = self.norm1(residual + self.dropout(self_attn_out))
        # --- End Self Attention ---

        # --- Cross Attention ---
        if self.is_encoder_decoder and self.cross_attn is not None and encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[1] if past_key_value is not None else None
            cross_attn_outputs = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                past_key_value=cross_attn_past_key_value,
                attention_mask=encoder_attention_mask, 
                output_attentions=output_attentions,
                use_cache=use_cache 
            )
            cross_attn_out = cross_attn_outputs[0]
            if output_attentions: cross_attn_probs = cross_attn_outputs[1]
            if use_cache: present_cross_kv = cross_attn_outputs[2] if len(cross_attn_outputs) > 2 else None
            
            if self.norm2 is not None:
                hidden_states = self.norm2(residual + self.dropout(cross_attn_out))
            else: 
                hidden_states = residual + self.dropout(cross_attn_out)
        # --- End Cross Attention ---

        # --- Feedforward ---
        residual = hidden_states
        ffn_out = self.ffn(hidden_states)
        hidden_states = self.norm3(residual + self.dropout(ffn_out))
        # --- End Feedforward ---

        present_key_value = (present_self_kv, present_cross_kv) if use_cache else None

        return hidden_states, self_attn_probs, cross_attn_probs, present_key_value
