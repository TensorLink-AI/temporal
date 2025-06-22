
import torch.nn as nn
from temporal.models.mixin.adaptive_patching import AdaptivePatching, PatchMerging
from temporal.modules.encoders.transformer_encoder_layer import TransformerEncoderLayer
from temporal.modules.decoders.base_decoder_layer import TimeSeriesTransformerDecoderLayer
from typing import Optional, Tuple
import torch


class HierarchicalTransformerEncoderBlock(nn.Module):
    """
    A wrapper around a standard TransformerEncoderLayer that applies adaptive
    patching and merging.
    """

    def __init__(self, encoder_layer: TransformerEncoderLayer, expansion_factor: int):
        super().__init__()

        if not isinstance(encoder_layer, TransformerEncoderLayer):
            raise TypeError(
                f"encoder_layer must be of type TransformerEncoderLayer, but got {type(encoder_layer)}"
            )

        self.encoder_layer = encoder_layer
        self.expansion_factor = expansion_factor

        embed_dim = encoder_layer.self_attn.embed_dim
        if embed_dim % expansion_factor != 0:
            raise ValueError(
                f"The embedding dimension ({embed_dim}) of the encoder_layer is not divisible by the expansion_factor ({expansion_factor})"
            )

        self.patcher = AdaptivePatching(expansion_factor)
        self.merger = PatchMerging(expansion_factor)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        """
        Args:
            hidden_states: [B, T, D]
        Returns:
            A tuple containing the output hidden states and optionally the attention weights.
        """
        x_patched = self.patcher(hidden_states)

        if attention_mask is not None:
            B, N, T, T_kv = attention_mask.shape
            attention_mask = attention_mask.repeat_interleave(self.expansion_factor, dim=2)
            attention_mask = attention_mask.repeat_interleave(self.expansion_factor, dim=3)

        layer_outputs = self.encoder_layer(
            hidden_states=x_patched,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        merged_output = self.merger(layer_outputs[0])

        if output_attentions:
            return (merged_output, layer_outputs[1])
        else:
            return (merged_output,)


class HierarchicalTransformerDecoderBlock(nn.Module):
    """
    A wrapper around a TimeSeriesTransformerDecoderLayer that applies adaptive
    patching and merging to the self-attention part of the decoder layer.
    """

    def __init__(
        self, decoder_layer: TimeSeriesTransformerDecoderLayer, expansion_factor: int
    ):
        super().__init__()

        if not isinstance(decoder_layer, TimeSeriesTransformerDecoderLayer):
            raise TypeError(
                f"decoder_layer must be of type TimeSeriesTransformerDecoderLayer, but got {type(decoder_layer)}"
            )

        self.decoder_layer = decoder_layer
        self.expansion_factor = expansion_factor

        embed_dim = decoder_layer.self_attn.embed_dim
        if embed_dim % expansion_factor != 0:
            raise ValueError(
                f"The embedding dimension ({embed_dim}) of the decoder_layer is not divisible by the expansion_factor ({expansion_factor})"
            )

        self.patcher = AdaptivePatching(expansion_factor)
        self.merger = PatchMerging(expansion_factor)

    def forward(
        self,
        hidden_states: nn.Module,
        encoder_hidden_states: Optional[nn.Module] = None,
        attention_mask: Optional[nn.Module] = None,
        encoder_attention_mask: Optional[nn.Module] = None,
        past_key_value: Optional[
            Tuple[Optional[Tuple], Optional[Tuple]]
        ] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Tuple[Tuple, Tuple]],
    ]:

        if use_cache:
            raise NotImplementedError(
                "KV caching is not yet supported in HierarchicalTransformerDecoderBlock."
            )

        x_patched = self.patcher(hidden_states)

        if attention_mask is not None:
            B, N, T, T_kv = attention_mask.shape
            attention_mask = attention_mask.repeat_interleave(self.expansion_factor, dim=2)
            attention_mask = attention_mask.repeat_interleave(self.expansion_factor, dim=3)

        layer_outputs = self.decoder_layer(
            hidden_states=x_patched,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        merged_output = self.merger(layer_outputs[0])

        final_outputs = (merged_output,) + layer_outputs[1:]

        return final_outputs
