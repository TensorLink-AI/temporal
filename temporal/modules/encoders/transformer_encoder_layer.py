import torch
import torch.nn as nn
import torch.nn.functional as F
from temporal.models.builder import ModuleBuilder

from temporal.registry.core import register_module

@register_module("block", "default_encoder")
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from temporal.models.builder import ModuleBuilder


class TimeSeriesTransformerEncoderLayer(nn.Module):
    def __init__(self, config, builder: ModuleBuilder):
        super().__init__()
        self.config = config

        self.self_attn = builder.build_attention(config.attention_blocks.encoder_attention)
        self.ffn = builder.build_feedforward()
        self.norm1 = builder.build_normalization()
        self.norm2 = builder.build_normalization()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # === Self-Attention ===
        residual = hidden_states
        attn_output_tensor, attn_probs, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.norm1(residual + self.dropout(attn_output_tensor))

        # === Feedforward ===
        residual = hidden_states
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm2(residual + self.dropout(ffn_output))

        return (hidden_states, attn_probs) if output_attentions else (hidden_states,)
