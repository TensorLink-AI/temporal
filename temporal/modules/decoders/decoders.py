import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from typing import Optional, Tuple, List, Union

# Import ModuleBuilder from the new helper file
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.block_builder import BlockBuilder
from temporal.configs.transformer_config import TransformerBlockConfig
from temporal.models.outputs import DecoderLayerOutput

class TimeSeriesTransformerDecoder(nn.Module):
    """
    A flexible Transformer decoder built from a list of block configurations.

    This module serves as the main decoder component in a Transformer-based
    time series model. It dynamically constructs a stack of decoder layers
    based on a list of `TransformerBlockConfig` objects.

    The decoder is responsible for:
    - Adding positional information to already embedded features.
    - Sequentially processing the embedded sequence through its layers.
    - Handling the Key-Value (KV) cache for efficient autoregressive generation.

    Attributes:
        config: The main configuration object for the model.
        value_embedding (nn.Module): The module for embedding input features.
            Note: This is not used in the `forward` pass, but is held here to be
            used by the parent model.
        dropout (nn.Dropout): Dropout layer applied after embeddings.
        layernorm_embedding (nn.Module): Layer normalization applied to the embeddings.
        positional_embedding (nn.Module): The module for adding positional information.
        layers (nn.ModuleList): The stack of decoder layers.
    """

    def __init__(
        self,
        config,
        builder: ModuleBuilder,
        block_configs: List[TransformerBlockConfig],
    ):
        """
        Initializes the TimeSeriesTransformerDecoder.

        Args:
            config: The main model configuration object.
            builder (ModuleBuilder): A helper class that constructs the various
                sub-modules (embeddings, normalization, etc.) based on the config.
            block_configs (List[TransformerBlockConfig]): A list of configurations,
                where each configuration defines a single decoder layer in the stack.
        """
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerdrop = getattr(config, "decoder_layerdrop", 0.0)

        self.layernorm_embedding = builder.build_normalization()
        self.value_embedding = builder.build_value_embedding()
        self.positional_embedding = builder.build_positional_embedding()

        block_builder = BlockBuilder(config, builder)
        self.layers = nn.ModuleList([
            block_builder.build_block(block_cfg) for block_cfg in block_configs
        ])

    def _get_past_key_values_length(self, past_key_values: Optional[List[Tuple]]) -> int:
        """
        Safely determines the sequence length from a Key-Value cache.

        Args:
            past_key_values (Optional[List[Tuple]]): The KV cache, which is a
                list of tuples, one for each layer.

        Returns:
            int: The length of the cached sequences, or 0 if the cache is empty.
        """
        if past_key_values is None or not past_key_values:
            return 0
        try:
            # Assuming KV cache format [Layer][Self/Cross][Key/Value] -> Tensor
            # Shape: [B, Heads, SeqLen, HeadDim] or [B, SeqLen, HiddenDim]
            first_layer_self_k = past_key_values[0][0][0]
            if first_layer_self_k.dim() == 4:
                 # Standard multi-head attention format
                 past_length = first_layer_self_k.shape[2]
            elif first_layer_self_k.dim() == 3:
                 # Might be a different attention format
                 past_length = first_layer_self_k.shape[1]
            else:
                 print(f"Warning: Unexpected KV cache tensor dimension: {first_layer_self_k.dim()}.")
                 past_length = 0
            return int(past_length)

        except (IndexError, AttributeError, TypeError) as e:
            print(f"Warning: Could not determine past_key_values_length from cache: {e}. Returning 0.")
            return 0

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, D] embedded features
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # Decoder self-attention mask
        encoder_attention_mask: Optional[torch.Tensor] = None, # Cross-attention mask
        past_key_values: Optional[List[Tuple[Tuple, Tuple]]] = None,  # List of (self_kv, cross_kv)
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]:
        """
        Performs the forward pass of the Transformer decoder.

        Args:
            hidden_states (torch.Tensor): The embedded input features for the decoder,
                shape `[B, T, D]`. The calling model is responsible for performing
                value embedding (e.g., patching) before passing to this module.
            encoder_hidden_states (Optional[torch.Tensor]): The output from the
                encoder, used for cross-attention. Shape `[B, T_enc, D]`.
            attention_mask (Optional[torch.Tensor]): The causal self-attention mask
                for the decoder. This mask must match the sequence dimension of `hidden_states`.
            encoder_attention_mask (Optional[torch.Tensor]): The padding mask for
                the encoder hidden states.
            past_key_values (Optional[List[Tuple[Tuple, Tuple]]]): The KV cache
                from previous decoding steps.
            output_attentions (bool): Whether to return attention weights.
            output_hidden_states (bool): Whether to return all hidden states.
            use_cache (bool): Whether to use and return the KV cache.
            return_dict (bool): Whether to return a structured model output.

        Returns:
            Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]: The decoder's
            output, either as a structured object or a tuple.
        """

        # Determine past sequence length for positional embeddings if using cache
        past_key_values_length = self._get_past_key_values_length(past_key_values)

        value_embeds = hidden_states
        batch_size, current_seq_len, _ = value_embeds.shape

        # Generate positional encoding using the new signature
        try:
            pos_embed = self.positional_embedding(
                batch_size=batch_size,
                seq_len=current_seq_len,
                past_key_values_length=past_key_values_length
            )

            # Ensure pos_embed shape is broadcastable: [1, T, D] or [B, T, D]
            if pos_embed.shape[0] != batch_size and pos_embed.shape[0] != 1:
                 raise ValueError(f"Positional embedding returned unexpected batch dim: {pos_embed.shape[0]}. Expected 1 or {batch_size}")
            if pos_embed.shape[1] != current_seq_len:
                 raise ValueError(f"Positional embedding returned unexpected seq len dim: {pos_embed.shape[1]}. Expected {current__seq_len}")

        except TypeError as e:
             if "positional_embedding()" in str(e) or "forward()" in str(e):
                 raise TypeError(
                     f"The positional embedding layer ({type(self.positional_embedding).__name__}) "
                     f"does not support the signature `forward(self, batch_size, seq_len, past_key_values_length)`. "
                     f"Check its implementation or the builder logic."
                 ) from e
             else:
                 raise e # Re-raise other TypeErrors
        except Exception as e:
            print(f"Error during positional embedding call: {e}")
            raise e

        # Sum + norm + dropout
        # Broadcasting handles cases where pos_embed is [1, T, D]
        hidden_states = value_embeds + pos_embed
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # --- Rest of the decoder logic remains largely the same ---
        all_hidden_states_collector = () if output_hidden_states else None
        all_self_attentions_collector = () if output_attentions else None
        all_cross_attentions_collector = () if output_attentions else None
        next_decoder_cache = [] if use_cache else None
        total_aux_loss = None

        for idx, layer in enumerate(self.layers):
            # Support for layer dropping during training
            if self.training and torch.rand([]).item() < self.layerdrop:
                if use_cache: next_decoder_cache.append(None)
                continue

            layer_past_key_value = past_key_values[idx] if past_key_values is not None else None

            if output_hidden_states:
                all_hidden_states_collector += (hidden_states,)

            layer_outputs: DecoderLayerOutput = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=layer_past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs.hidden_states

            if use_cache:
                 next_decoder_cache.append(layer_outputs.past_key_value)

            if output_attentions:
                 if layer_outputs.self_attention_weights is not None:
                     all_self_attentions_collector += (layer_outputs.self_attention_weights,)
                 if layer_outputs.cross_attention_weights is not None:
                     all_cross_attentions_collector += (layer_outputs.cross_attention_weights,)

            if layer_outputs.aux_loss is not None:
                if total_aux_loss is None:
                    total_aux_loss = layer_outputs.aux_loss
                else:
                    total_aux_loss += layer_outputs.aux_loss

        if output_hidden_states:
            all_hidden_states_collector += (hidden_states,)

        next_cache = tuple(next_decoder_cache) if use_cache else None

        if not return_dict:
            return tuple(v for v in [
                hidden_states, all_hidden_states_collector, all_self_attentions_collector, all_cross_attentions_collector, next_cache
            ] if v is not None)

        output = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states_collector,
            attentions=all_self_attentions_collector,
            cross_attentions=all_cross_attentions_collector,
            past_key_values=next_cache,
        )
        if total_aux_loss is not None:
            output.aux_loss = total_aux_loss
        return output
