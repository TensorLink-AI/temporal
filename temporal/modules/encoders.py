import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
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


class BaseEncoder(nn.Module):
    """Base class for all encoders to inherit from."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Each encoder must implement its own forward method.")


class TimeSeriesTransformerEncoderLayer(BaseLayer):
    """Encoder layer for Time Series Transformer, inheriting from BaseLayer."""

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.hidden_size

        # Self-attention layer
        self.self_attn = TimeSeriesAttention(config)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = F.dropout(hidden_states, p=self.config.hidden_dropout_prob, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.config.hidden_dropout_prob, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.config.hidden_dropout_prob, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return (hidden_states, attn_weights) if output_attentions else (hidden_states,)


class TimeSeriesTransformerEncoder(BaseEncoder):
    """
    Transformer encoder for time series, utilizing feature embeddings and positional encoding.
    """

    def __init__(self, config):
        super().__init__(config)

        self.dropout = config.hidden_dropout_prob
        self.layerdrop = config.encoder_layerdrop

        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        # Feature embedding for input values
        # Make an abstract call for this to use different embedders
        self.value_embedding = TimeSeriesValueEmbedding(feature_size=config.feature_size, d_model=config.hidden_size)
        # self.value_embedding = TimeSeriesValueEmbedding(feature_size=config.feature_size, d_model=config.hidden_size)

        # Sinusoidal positional encoding
        # Make an abstract call for this to use different embedders

        self.embed_positions = TimeSeriesSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.hidden_size
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TimeSeriesTransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

        self.layernorm_embedding = nn.LayerNorm(config.hidden_size)



    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,  # ✅ Future-proofing
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        hidden_states = self.value_embedding(inputs_embeds)
        # new
        bsz, seq_len = hidden_states.shape[:2]
        pos = self.embed_positions((bsz, seq_len))
        hidden_states = hidden_states + pos
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # Expand attention mask if provided
       # if attention_mask is not None:
       #     attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)  # ✅ Fix expansion function

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # ✅ Implement LayerDrop (skip layer with some probability)
            if self.training and torch.rand([]) < self.layerdrop:
                continue  # Skip this layer with probability `self.layerdrop`

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[idx] if head_mask is not None else None,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,  # ✅ No need for caching in encoder
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )



def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    """
    Expands a 2D or 3D attention mask (of shape [batch_size, seq_len]) 
    into a 4D mask for multi-head attention.

    Args:
        mask: shape [batch_size, seq_len] (or [batch_size, 1, seq_len, seq_len])
        dtype: usually mask.dtype or hidden_states.dtype
        tgt_len: optionally the target length (decoder length) if needed

    Returns:
        mask of shape [batch_size, 1, tgt_len, seq_len]
        with 0.0 for "keep" and -inf for "masked" positions
    """

    if mask.dim() == 2:
        # shape = [batch_size, seq_len]
        batch_size, src_len = mask.shape
        tgt_len = tgt_len if tgt_len is not None else src_len
        # (batch_size, 1, tgt_len, src_len)
        expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len)
    elif mask.dim() == 3:
        # shape = [batch_size, 1, seq_len, seq_len] -> assume it's already 4D
        return mask
    else:
        raise ValueError(f"Unsupported mask dim: {mask.dim()}")

    # Convert from 1.0/0.0 mask to 0.0/-inf
    expanded_mask = expanded_mask.to(dtype=dtype)
    inverted_mask = (1.0 - expanded_mask) * -1e9
    return inverted_mask