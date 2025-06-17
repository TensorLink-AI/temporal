import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from typing import Optional, Tuple, List, Union

# Import ModuleBuilder from the new helper file
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.block_builder import BlockBuilder
from temporal.configs.transformer_config import TransformerBlockConfig


class TimeSeriesTransformerDecoder(nn.Module):
    """A flexible Transformer decoder built from a list of block configurations.

    This module serves as the main decoder component in a Transformer-based
    time series model. It dynamically constructs a stack of decoder layers
    based on a list of `TransformerBlockConfig` objects.

    The decoder is responsible for:
    - Embedding the input time series features.
    - Adding positional information.
    - Sequentially processing the embedded sequence through its layers.
    - Handling the Key-Value (KV) cache for efficient autoregressive generation.

    Attributes:

        config: The main configuration object for the model.
        dropout (nn.Dropout): Dropout layer applied after embeddings.
        layernorm_embedding (nn.Module): Layer normalization applied to the embeddings.
        value_embedding (nn.Module): The module for embedding input features.
        positional_embedding (nn.Module): The module for adding positional information.
        layers (nn.ModuleList): The stack of decoder layers.
    """

    def __init__(
        self,
        config,
        builder: ModuleBuilder,
        block_configs: List[TransformerBlockConfig],
    ):
        """Initializes the TimeSeriesTransformerDecoder.

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
        # Positional embedding layer should now accept (batch_size, seq_len, past_len)
        self.positional_embedding = builder.build_positional_embedding()

        block_builder = BlockBuilder(config, builder)
        self.layers = nn.ModuleList([
            block_builder.build_block(block_cfg) for block_cfg in block_configs
        ])

    def _get_past_key_values_length(self, past_key_values: Optional[List[Tuple]]) -> int:
        """Safely determines the sequence length from a Key-Value cache.

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
        input_ids: torch.Tensor,  # [B, T, F] raw features
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # Decoder self-attention mask
        encoder_attention_mask: Optional[torch.Tensor] = None, # Cross-attention mask
        past_key_values: Optional[List[Tuple[Tuple, Tuple]]] = None,  # List of (self_kv, cross_kv)
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]:
        """Performs the forward pass of the Transformer decoder.

        Args:

            input_ids (torch.Tensor): The raw input features for the decoder,
                shape `[B, T, F]`.
            encoder_hidden_states (Optional[torch.Tensor]): The output from the
                encoder, used for cross-attention. Shape `[B, T_enc, D]`.
            attention_mask (Optional[torch.Tensor]): The causal self-attention mask
                for the decoder.
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

        batch_size, current_seq_len, _ = input_ids.shape

        # === Embedding ===
        value_embeds = self.value_embedding(input_ids)  # [B, T, D]

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
                 raise ValueError(f"Positional embedding returned unexpected seq len dim: {pos_embed.shape[1]}. Expected {current_seq_len}")

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

        for idx, layer in enumerate(self.layers):
            # Support for layer dropping during training
            if self.training and torch.rand([]).item() < self.layerdrop:
                if use_cache: next_decoder_cache.append(None)
                continue

            layer_past_key_value = past_key_values[idx] if past_key_values is not None else None

            if output_hidden_states:
                all_hidden_states_collector += (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=layer_past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                 present_key_value = layer_outputs[-1] if len(layer_outputs) > 1 else None
                 if not isinstance(present_key_value, tuple) and present_key_value is not None:
                      if hasattr(present_key_value, 'past_key_value'):
                           present_key_value = present_key_value.past_key_value
                      else:
                           print(f"Warning: Layer {idx} output structure unexpected when use_cache=True.")
                           present_key_value = None
                 next_decoder_cache.append(present_key_value)

            if output_attentions:
                 self_attn_weights = layer_outputs[1] if len(layer_outputs) > 1 and layer_outputs[1] is not None else None
                 cross_attn_weights = layer_outputs[2] if len(layer_outputs) > 2 and layer_outputs[2] is not None else None
                 if self_attn_weights is not None: all_self_attentions_collector += (self_attn_weights,)
                 if cross_attn_weights is not None and encoder_hidden_states is not None: all_cross_attentions_collector += (cross_attn_weights,)

        if output_hidden_states:
            all_hidden_states_collector += (hidden_states,)

        next_cache = tuple(next_decoder_cache) if use_cache else None

        if not return_dict:
            return tuple(v for v in [
                hidden_states, all_hidden_states_collector, all_self_attentions_collector, all_cross_attentions_collector, next_cache
            ] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states_collector,
            attentions=all_self_attentions_collector,
            cross_attentions=all_cross_attentions_collector,
            past_key_values=next_cache,
        )
