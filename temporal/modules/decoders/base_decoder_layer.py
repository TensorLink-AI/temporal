import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from temporal.models.builder import ModuleBuilder

from temporal.registry.core import register_module

@register_module("block", "default_decoder")
class TimeSeriesTransformerDecoderLayer(nn.Module):
    def __init__(self, config, builder: ModuleBuilder):
        super().__init__()
        self.config = config

        self.self_attn = builder.build_attention(config.attention_blocks.decoder_attention)
        self.cross_attn = builder.build_attention(config.attention_blocks.decoder_cross_attention)
        self.ffn = builder.build_feedforward(config.feedforward_config)

        self.norm1 = builder.build_normalization()
        self.norm2 = builder.build_normalization()
        self.norm3 = builder.build_normalization()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,                            # [B, T_dec, D]
        encoder_hidden_states: Optional[torch.Tensor] = None,   # [B, T_enc, D]
        attention_mask: Optional[torch.Tensor] = None,          # [B, 1, T_dec, T_dec]
        encoder_attention_mask: Optional[torch.Tensor] = None,  # [B, 1, T_dec, T_enc]
        past_key_value: Optional[Tuple[Tuple, Tuple]] = None,   # ((self_k, self_v), (cross_k, cross_v))
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Tuple[Tuple, Tuple]]:

        self_kv = past_key_value[0] if past_key_value else None
        cross_kv = past_key_value[1] if past_key_value else None

        # === Self Attention ===
        residual = hidden_states
        self_attn_out, self_attn_probs, present_self_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=self_kv,
            output_attentions=output_attentions
        )
        hidden_states = self.norm1(residual + self.dropout(self_attn_out))

        # === Cross Attention ===
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
        else:
            cross_attn_probs = None
            present_cross_kv = None

        # === Feedforward ===
        residual = hidden_states
        ffn_out = self.ffn(hidden_states)
        hidden_states = self.norm3(residual + self.dropout(ffn_out))

        present = (present_self_kv, present_cross_kv)
        return hidden_states, self_attn_probs, cross_attn_probs, present