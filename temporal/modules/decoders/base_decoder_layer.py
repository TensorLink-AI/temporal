# Modified temporal/modules/decoders/base_decoder_layer.py
import torch
import torch.nn as nn
from typing import Optional, Tuple
# Import the specific block config type hint
from temporal.configs.transformer_block_config import TransformerBlockConfig
from temporal.configs.attention_config import AttentionConfig
from temporal.configs.feedforward_config import FeedForwardConfig
from temporal.configs.transformer_model_config import TransformerTimeSeriesConfig # Added main config
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.registry.core import register_module
from temporal.models.outputs import DecoderLayerOutput
import copy # Import copy for deepcopy

@register_module("block", "default_decoder")
class TimeSeriesTransformerDecoderLayer(nn.Module):
    """
    A standard Transformer decoder layer for time series.

    This module implements a single layer of a Transformer decoder, which is
    a fundamental building block for sequence-to-sequence models in time series
    forecasting. It consists of three main components:
    1.  A masked self-attention mechanism to process the decoder's own input sequence.
    2.  An optional cross-attention mechanism to attend to the output of an encoder.
    3.  A feed-forward network (FFN).

    Each component is followed by a residual connection and layer normalization.
    The specific implementations of attention, FFN, and normalization are
    dynamically built based on the provided configuration.

    Attributes:
        config (TransformerBlockConfig): The configuration for this specific block.
        is_encoder_decoder (bool): Flag indicating if this layer is part of an
            encoder-decoder architecture, which determines if cross-attention is built.
        self_attn (nn.Module): The self-attention module.
        cross_attn (Optional[nn.Module]): The cross-attention module.
        ffn (nn.Module): The feed-forward network.
        norm1 (nn.Module): Layer normalization after self-attention.
        norm2 (Optional[nn.Module]): Layer normalization after cross-attention.
        norm3 (nn.Module): Layer normalization after the FFN.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, config: TransformerBlockConfig, builder: ModuleBuilder):
        """
        Initializes the TimeSeriesTransformerDecoderLayer.

        Args:
            config (TransformerBlockConfig): The configuration specific to this decoder
                layer, defining the types of attention and FFN to be used.
            builder (ModuleBuilder): A helper class that constructs the sub-modules
                (attention, FFN, normalization) based on the main model configuration.
        """
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
        # self_attn_build_config.kwargs = self_attn_build_config.kwargs or {} # REMOVED: kwargs is now guaranteed by BaseConfig
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
            # cross_attn_build_config.kwargs = cross_attn_build_config.kwargs or {} # REMOVED: kwargs is now guaranteed by BaseConfig
            cross_attn_build_config.kwargs['is_decoder'] = True
            cross_attn_build_config.kwargs['is_cross_attention'] = True
            self.cross_attn = builder.build_attention(cross_attn_build_config)
        # --- End Cross Attention ---

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

        self.norm1 = builder.build_normalization(config.norm_config)
        self.norm2 = builder.build_normalization(config.norm_config) if self.is_encoder_decoder and self.cross_attn is not None else None 
        self.norm3 = builder.build_normalization(config.norm_config)
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
    ) -> DecoderLayerOutput:
        """
        Performs the forward pass of the decoder layer.

        Args:
            hidden_states (torch.Tensor): The input to the layer, shape `[B, T_dec, D]`.
            encoder_hidden_states (Optional[torch.Tensor]): The sequence from the
                encoder's output, shape `[B, T_enc, D]`. Required for cross-attention.
            attention_mask (Optional[torch.Tensor]): The causal mask for self-attention,
                shape `[B, 1, T_dec, T_dec]`.
            encoder_attention_mask (Optional[torch.Tensor]): The padding mask for
                cross-attention, shape `[B, 1, T_dec, T_enc]`.
            past_key_value (Optional[Tuple[Optional[Tuple], Optional[Tuple]]]): A tuple
                containing cached key-value states for self-attention and cross-attention,
                used for efficient autoregressive decoding.
            output_attentions (bool): Whether to return the attention weights.
            use_cache (bool): If True, the layer will return the updated key-value
                states for future decoding steps.

        Returns:
            DecoderLayerOutput: An object containing the output hidden states,
                optional attention weights, optional auxiliary loss, and optional
                past key value.
        """
        residual = hidden_states
        self_attention_weights = None
        cross_attention_weights = None
        present_self_key_value = None
        present_cross_key_value = None
        aux_loss = None

        # --- Self Attention ---
        self_attn_past_key_value = past_key_value[0] if past_key_value is not None else None
        self_attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=None, # Self-attention does not use key_value_states
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            use_cache=use_cache
        )
        self_attention_output = self_attention_outputs[0]
        if output_attentions:
            self_attention_weights = self_attention_outputs[1]
        if use_cache:
            present_self_key_value = self_attention_outputs[2] if len(self_attention_outputs) > 2 else None
        hidden_states = self.norm1(residual + self.dropout(self_attention_output))
        # --- End Self Attention ---

        # --- Cross Attention ---
        if self.is_encoder_decoder and self.cross_attn is not None and encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[1] if past_key_value is not None else None
            cross_attention_outputs = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                past_key_value=cross_attn_past_key_value,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            cross_attention_output = cross_attention_outputs[0]
            if output_attentions:
                cross_attention_weights = cross_attention_outputs[1]
            if use_cache:
                present_cross_key_value = cross_attention_outputs[2] if len(cross_attention_outputs) > 2 else None
            
            if self.norm2 is not None:
                hidden_states = self.norm2(residual + self.dropout(cross_attention_output))
            else:
                hidden_states = residual + self.dropout(cross_attention_output)
        # --- End Cross Attention ---

        # --- Feedforward ---
        residual = hidden_states
        ffn_outputs = self.ffn(hidden_states)
        hidden_states = self.norm3(residual + self.dropout(ffn_outputs[0]))
        if len(ffn_outputs) > 1 and ffn_outputs[1] is not None:
            aux_loss = ffn_outputs[1]

        # --- End Feedforward ---

        present_key_value = (present_self_key_value, present_cross_key_value) if use_cache else None

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            self_attention_weights=self_attention_weights,
            cross_attention_weights=cross_attention_weights,
            past_key_value=present_key_value,
            aux_loss=aux_loss
        )
