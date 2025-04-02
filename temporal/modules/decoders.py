import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from temporal.modules.embedding import TimeSeriesValueEmbedding, TimeSeriesSinusoidalPositionalEmbedding
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

        # Self-attention
        self.self_attn = TimeSeriesAttention(config)

        # Cross-attention
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
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        training_mode: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Return a tuple of (hidden_states, attention_weights, present_key_value).
        If cross-attn is used, the second element includes cross-attn weights too.
        """

        residual = hidden_states

        # Disable caching in training mode
        if training_mode:
            past_key_values = None
            use_cache = False

        # 1) Self-Attention
        self_attn_past_key_value = past_key_values[:2] if past_key_values else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=None,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
        )

        hidden_states = F.dropout(hidden_states, p=self.config.hidden_dropout_prob, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 2) Cross-Attention
        cross_attn_weights = None
        cross_attn_present_key_value = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_values[-2:] if past_key_values else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                past_key_value=cross_attn_past_key_value,
                attention_mask=encoder_attention_mask,
                head_mask=None,
                output_attentions=output_attentions,
            )

            hidden_states = F.dropout(hidden_states, p=self.config.hidden_dropout_prob, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            if present_key_value is not None and cross_attn_present_key_value is not None:
                # Combine cross and self keys
                present_key_value = present_key_value + cross_attn_present_key_value

        # 3) Feed-forward
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.config.hidden_dropout_prob, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.config.hidden_dropout_prob, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # Return 3-tuple: (hidden_states, [self_attn_weights, cross_attn_weights], present_key_value)
        if output_attentions:
            # Combine self + cross
            attn_info = (self_attn_weights, cross_attn_weights)
        else:
            attn_info = None

        if use_cache and not training_mode and present_key_value is not None:
            return (hidden_states, attn_info, present_key_value)
        else:
            return (hidden_states, attn_info, None)


class TimeSeriesTransformerDecoder(BaseDecoder):
    """Decoder for Time Series Transformer with value & positional embeddings."""

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            TimeSeriesTransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.value_embedding = TimeSeriesValueEmbedding(
            feature_size=config.feature_size, d_model=config.hidden_size
        )
        self.embed_positions = TimeSeriesSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.hidden_size
        )
        self.output_projection = nn.Linear(config.hidden_size, 1)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,    # <--- new param, consistent with encoder
        training_mode: bool = True,  # pass in training_mode if needed
    ) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]:
        """
        If return_dict=True, returns BaseModelOutputWithPastAndCrossAttentions.
        Otherwise, returns a tuple (last_hidden_state, [self/cross attn weights], next_cache).
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = []

        # 1) Value + Positional Embeddings
        hidden_states = self.value_embedding(inputs_embeds)  # => [B, S, hidden_size]
        bsz, seq_len = inputs_embeds.shape[:2]
        pos_emb = self.embed_positions((bsz, seq_len))       # => [B, S, hidden_size]
        hidden_states = self.layernorm(hidden_states + pos_emb)

        # 2) Pass through layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # handle past cache
            layer_past = past_key_values[i] if (past_key_values and i < len(past_key_values)) else None

            layer_outputs = layer(
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
                training_mode=training_mode,
            )

            # layer_outputs => (hidden_states, attn_info, present_key_value)
            hidden_states = layer_outputs[0]
            attn_info = layer_outputs[1]
            present_key_value = layer_outputs[2]

            if output_attentions and attn_info is not None:
                # attn_info is (self_attn_weights, cross_attn_weights)
                all_attentions += (attn_info,)

            if use_cache and not training_mode and present_key_value is not None:
                next_cache.append(present_key_value)

        # If returning hidden states, add final
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 3) Build outputs
        if not return_dict:
            # Return as a tuple
            outputs = (hidden_states, next_cache)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attentions,)
            return outputs

        # 4) Return as BaseModelOutputWithPastAndCrossAttentions
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache if (use_cache and not training_mode) else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
