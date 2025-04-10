import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from temporal.models.builder import ModuleBuilder

from temporal.registry.core import register_module

@register_module("block", "default_decoder")
class TimeSeriesTransformerDecoderLayer(nn.Module):
    """
    A single decoder layer for the time series transformer, built via registry-backed submodules:
      - self-attention
      - cross-attention
      - feedforward network
      - normalization (pre/post)
      - residual connections

    All modules are instantiated using the provided ModuleBuilder and config.
    """

    def __init__(self, config, builder: ModuleBuilder):
        super().__init__()
        self.config = config

        # Build submodules using the registry-based builder
        self.self_attn = builder.build_attention(config.attention_blocks.decoder_attention)
        self.cross_attn = builder.build_attention(config.attention_blocks.decoder_cross_attention)
        self.ffn = builder.build_feedforward()

        self.norm1 = builder.build_normalization()
        self.norm2 = builder.build_normalization()
        self.norm3 = builder.build_normalization()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.functional.__dict__.get(config.hidden_act, nn.functional.gelu)

    def forward(
        self,
        hidden_states: torch.Tensor,                            # [B, T_dec, D]
        encoder_hidden_states: Optional[torch.Tensor] = None,   # [B, T_enc, D]
        attention_mask: Optional[torch.Tensor] = None,          # [B, 1, T_dec, T_dec]
        encoder_attention_mask: Optional[torch.Tensor] = None,  # [B, 1, T_dec, T_enc]
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        # === Self-Attention ===
        residual = hidden_states
        attn_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=None,  # Could integrate caching logic here
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self.norm1(residual + self.dropout(attn_output))

        # === Cross-Attention ===
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_output = self.cross_attn(
                hidden_states,
                context=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = self.norm2(residual + self.dropout(cross_output))

        # === Feedforward ===
        residual = hidden_states
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm3(residual + self.dropout(ffn_output))

        return hidden_states
