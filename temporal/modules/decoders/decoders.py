# modules/decoders/decoders.py
from __future__ import annotations

"""
TimeSeriesTransformerDecoder
===========================

A thin orchestration wrapper that builds a stack of decoder blocks and handles:
- Per-layer KV cache wiring (correctly slices `past_key_values[idx]`).
- Optional layerdrop during training.
- Collection of hidden states and (self/cross) attentions.
- Tuple-style cache accumulation compatible with HF-style decoders.
- Optional auxiliary losses aggregated across layers.

Return behavior (backward compatible with your existing code):
- If `return_dict=True` (default): returns `BaseModelOutputWithPastAndCrossAttentions`
  with `last_hidden_state`, `past_key_values`, `hidden_states`, `attentions`,
  and `cross_attentions`. (Also attaches `aux_loss` attribute if produced.)
- If `return_dict=False`: returns a tuple in the order your code previously used:
  (hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions, next_cache)

Notes on KV cache:
- `past_key_values` is expected to be a tuple/list of length == num_layers, where
  each item is the cache tuple for that layer.
- During training with layerdrop, we append `None` to the next cache to preserve
  positional alignment.

Dependencies:
- `DecoderLayerOutput` must provide:
  - hidden_states (Tensor)
  - past_key_value (tuple or None)
  - self_attention_weights (Tensor or None)
  - cross_attention_weights (Tensor or None)
  - aux_loss (Tensor or None)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.block_builder import BlockBuilder
from temporal.configs.transformer_block_config import TransformerBlockConfig
from temporal.models.outputs import DecoderLayerOutput


class TimeSeriesTransformerDecoder(nn.Module):
    """
    A flexible Transformer decoder built from a list of block configurations.

    This module serves as the main decoder component in a Transformer-based
    time series model. It dynamically constructs a stack of decoder layers
    based on a list of `TransformerBlockConfig` objects.

    The decoder is responsible for:
      - Sequentially processing an already-embedded sequence through its layers.
      - Handling the Key-Value (KV) cache for efficient autoregressive generation.

    The calling model (e.g. `TransformerTemporalModel`) is responsible for input
    preprocessing (embeddings, normalization, etc.).

    Args:
        config: Main configuration object (must expose `decoder_layerdrop` if used).
        builder: ModuleBuilder to construct submodules.
        block_configs: Per-layer configs that `BlockBuilder` consumes.

    Attributes:
        layerdrop (float): Probability to drop a layer (training only).
        layers (nn.ModuleList): Stack of constructed decoder layers.
    """

    def __init__(
        self,
        config,
        builder: ModuleBuilder,
        block_configs: List[TransformerBlockConfig],
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.layerdrop = float(getattr(config, "decoder_layerdrop", 0.0))

        block_builder = BlockBuilder(config, builder)
        self.layers = nn.ModuleList(
            [block_builder.build_block(block_cfg) for block_cfg in block_configs]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        return_dict: bool = True,
        x_raw: Optional[torch.Tensor] = None,
    ) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]:
        """
        Args:
            hidden_states: [batch, seq, hidden]
            encoder_hidden_states: Optional cross-attn memory [batch, src_seq, hidden]
            attention_mask: Decoder self-attention mask
            encoder_attention_mask: Cross-attention mask
            past_key_values: Tuple of length `num_layers`, each the per-layer cache
            output_attentions: If True, collect attention weights
            output_hidden_states: If True, collect per-layer hidden states
            use_cache: If True, return per-layer caches
            return_dict: If True, return a dataclass output; else a tuple (legacy order)
            x_raw: Optional raw input (for blocks that need it)

        Returns:
            BaseModelOutputWithPastAndCrossAttentions OR tuple per `return_dict`.
        """
        # Collectors
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None

        # Cache to accumulate (tuple for HF compatibility)
        next_decoder_cache = () if use_cache else None

        total_aux_loss = None
        num_layers = len(self.layers)

        # Safety: allow shorter past_key_values; gracefully ignore extras
        has_past = past_key_values is not None
        past_len = len(past_key_values) if has_past else 0

        for idx, layer in enumerate(self.layers):
            # LayerDrop (training only): skip executing the layer with prob=layerdrop
            if self.training and self.layerdrop > 0.0 and torch.rand(()) < self.layerdrop:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                if use_cache:
                    # Preserve alignment in cache structure
                    next_decoder_cache += (None,)
                continue

            # Slice the correct past cache for this layer
            layer_past = past_key_values[idx] if (has_past and idx < past_len) else None

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs: DecoderLayerOutput = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=layer_past,
                output_attentions=output_attentions,
                use_cache=use_cache,
                x_raw=x_raw,
            )

            # Unpack
            hidden_states = layer_outputs.hidden_states

            if use_cache:
                next_decoder_cache += (layer_outputs.past_key_value,)

            if output_attentions:
                if layer_outputs.self_attention_weights is not None:
                    all_self_attns += (layer_outputs.self_attention_weights,)
                if layer_outputs.cross_attention_weights is not None:
                    all_cross_attns += (layer_outputs.cross_attention_weights,)

            if layer_outputs.aux_loss is not None:
                total_aux_loss = (
                    layer_outputs.aux_loss
                    if total_aux_loss is None
                    else total_aux_loss + layer_outputs.aux_loss
                )

        # Final hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache  # already a tuple (or None)

        # Legacy tuple path: keep your existing order to avoid breaking callers
        if not return_dict:
            # (hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions, next_cache)
            return tuple(
                v
                for v in (
                    hidden_states,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attns,
                    next_cache,
                )
                if v is not None
            )

        # Structured dataclass output (HF-compatible field names)
        output = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attns,
        )
        # Attach aux_loss if available (downstream can check hasattr(output, "aux_loss"))
        if total_aux_loss is not None:
            output.aux_loss = total_aux_loss
        return output
