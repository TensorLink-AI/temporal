
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from temporal.modules.embedding import TimeSeriesValueEmbedding,TimeSeriesSinusoidalPositionalEmbedding
from temporal.modules.attention import TimeSeriesAttention

class BaseLayer(nn.Module):
    """Base layer for all transformer components."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Each layer must implement its own forward method.")



class BaseDecoder(nn.Module):
    """Base class for all decoders to inherit from."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        raise NotImplementedError("Each decoder must implement its own forward method.")


class TimeSeriesTransformerDecoderLayer(BaseLayer):
    """Decoder layer for Time Series Transformer, inheriting from BaseLayer."""

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.hidden_size

        # Self-attention layer
        # Make an abstract layer to define attention
        self.self_attn = TimeSeriesAttention(config)


        # Cross-attention layer
        # Make an abstract layer to define attention

        self.encoder_attn = TimeSeriesAttention(config)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        training_mode: bool = True,  # ✅ NEW FLAG
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        residual = hidden_states

        # ✅ Disable `past_key_values` during training (AR loss with sliding window)
        if training_mode:
            past_key_values = None
            use_cache = False  # ✅ Ensure caching is disabled in training

        # Self-Attention (with caching for inference)
        self_attn_past_key_value = past_key_values[:2] if past_key_values else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
        )

        hidden_states = F.dropout(hidden_states, p=self.config.hidden_dropout_prob, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention (only if encoder_hidden_states exist)
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_values[-2:] if past_key_values else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                head_mask=cross_attn_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = F.dropout(hidden_states, p=self.config.hidden_dropout_prob, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # ✅ Ensure cross-attn cache is valid
            if present_key_value is not None and cross_attn_present_key_value is not None:
                present_key_value = present_key_value + cross_attn_present_key_value

        # Feed-Forward
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.config.hidden_dropout_prob, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.config.hidden_dropout_prob, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # ✅ Use cache only for inference
        if use_cache and not training_mode and present_key_value is not None:
            outputs += (present_key_value,)

        return outputs


class TimeSeriesTransformerDecoder(BaseDecoder):
    """Decoder for Time Series Transformer with feature embeddings."""

    def __init__(self, config):
        super().__init__(config)

        self.layers = nn.ModuleList([
            TimeSeriesTransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

        self.value_embedding = TimeSeriesValueEmbedding(feature_size=config.feature_size, d_model=config.hidden_size)
        self.embed_positions = TimeSeriesSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.hidden_size
        )
        
        self.output_projection = nn.Linear(config.hidden_size, 1)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = True,
        training_mode: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_cache = []

        # Apply feature embedding
        hidden_states = self.value_embedding(inputs_embeds)
        embed_pos = self.embed_positions(inputs_embeds.size())
        hidden_states = self.layernorm(hidden_states + embed_pos)

        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values else None

            layer_outputs = layer(
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_value,
                use_cache=use_cache,
                training_mode=training_mode,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache and not training_mode and layer_outputs[-1] is not None:
                next_cache.append(layer_outputs[-1])

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache if use_cache and not training_mode else None,
        )