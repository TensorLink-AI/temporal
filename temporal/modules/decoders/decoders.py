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
    """
    Flexible Transformer decoder using block configs.

    Supports:
    - Self + Cross attention
    - Caching for autoregressive decoding
    - Optional return of attention weights
    """

    def __init__(
        self,
        config,
        builder: ModuleBuilder,
        block_configs: List[TransformerBlockConfig],
    ):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerdrop = getattr(config, "decoder_layerdrop", 0.0)

        self.layernorm_embedding = builder.build_normalization()
        self.value_embedding = builder.build_value_embedding()
        # Ensure the positional embedding layer used supports the required signature
        self.positional_embedding = builder.build_positional_embedding()

        block_builder = BlockBuilder(config, builder)
        self.layers = nn.ModuleList([
            block_builder.build_block(block_cfg) for block_cfg in block_configs
        ])

    def _get_past_key_values_length(self, past_key_values: Optional[List[Tuple]]) -> int:
        """ Helper to get sequence length from KV cache. """
        if past_key_values is None or not past_key_values:
            return 0
        # Cache format: List[layer_idx] -> Tuple[self_attn_kv, cross_attn_kv]
        # self_attn_kv: Tuple[key, value], shapes like [B, Heads, SeqLen, HeadDim]
        # We need the SeqLen dimension from the key or value tensor of the first layer's self-attention cache
        try:
            # Check self-attention key tensor shape (index 0, 0)
            # Shape is typically [batch_size, num_heads, sequence_length, head_dim]
            # Or sometimes [batch_size, sequence_length, num_heads * head_dim] if heads are merged
            # Assuming the standard [B, Heads, SeqLen, HeadDim] or similar where SeqLen is dim 2
            first_layer_self_k = past_key_values[0][0][0]
            if first_layer_self_k.dim() == 4: # Standard case
                 return first_layer_self_k.shape[2]
            elif first_layer_self_k.dim() == 3: # Might be [B, SeqLen, HiddenDim]
                 return first_layer_self_k.shape[1]
            else: # Fallback if shape is unexpected
                 print(f"Warning: Unexpected KV cache tensor dimension: {first_layer_self_k.dim()}. Assuming seq_len is 0.")
                 return 0

        except (IndexError, AttributeError) as e:
            print(f"Warning: Could not determine past_key_values_length from cache structure: {e}. Returning 0.")
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

        # Determine past sequence length for positional embeddings if using cache
        past_key_values_length = self._get_past_key_values_length(past_key_values)

        # === Embedding ===
        # Embed raw float input features
        value_embeds = self.value_embedding(input_ids)  # [B, T, D]

        # Generate sinusoidal position encoding based on input shape and past length
        # This requires self.positional_embedding to have the signature:
        # forward(self, input_shape: torch.Size, past_key_values_length: int = 0)
        try:
            pos_embed = self.positional_embedding(
                input_ids.shape, # Pass the shape [B, T, F]
                past_key_values_length=past_key_values_length
            ) # [B, T, D]
        except TypeError as e:
             # Add more informative error if the embedding layer doesn't support the call signature
             if "positional_embedding() takes" in str(e) or "forward() takes" in str(e):
                 raise TypeError(
                     f"The configured positional embedding layer ({type(self.positional_embedding).__name__}) "
                     f"does not seem to support the required signature "
                     f"`forward(self, input_shape, past_key_values_length)`. Check its implementation."
                 ) from e
             else:
                 raise e # Re-raise other TypeErrors

        # Sum + norm + dropout
        hidden_states = value_embeds + pos_embed
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)


        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None
        next_decoder_cache = [] if use_cache else None # Renamed for clarity

        # Adjust attention mask for caching if needed
        # `attention_mask` passed in is for the *current* input_ids relative to the full sequence.
        # It might need adjustment or combination with cached length info depending on the specific block implementation.
        # For now, assume blocks handle slicing or absolute positions correctly based on past_kv.

        for idx, layer in enumerate(self.layers):
            if self.training and torch.rand([]).item() < self.layerdrop:
                if use_cache: # Need to append None to keep cache structure consistent
                    next_decoder_cache.append(None)
                continue

            layer_past_key_value = past_key_values[idx] if past_key_values is not None else None

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Pass necessary inputs to the decoder layer
            # Ensure the layer signature matches what's being passed
            layer_outputs = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask, # Pass the potentially adjusted mask for self-attention
                encoder_attention_mask=encoder_attention_mask, # Pass the mask for cross-attention
                past_key_value=layer_past_key_value, # Pass the cache for this layer
                output_attentions=output_attentions,
                use_cache=use_cache, # Inform layer if cache is needed
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                 # layer_outputs should ideally return the new cache state if use_cache=True
                 # Assuming layer_outputs structure is (hidden_state, self_attn_weights, cross_attn_weights, present_key_value)
                 # Check if the layer actually returned a cache tuple
                 present_key_value = layer_outputs[-1] if len(layer_outputs) > 1 else None
                 if not isinstance(present_key_value, tuple) and present_key_value is not None:
                      # Attempt to find cache in a potential dict output if layer uses return_dict=True internally
                      if hasattr(present_key_value, 'past_key_value'):
                           present_key_value = present_key_value.past_key_value
                      else:
                           print(f"Warning: Layer {idx} output structure unexpected or did not return cache when use_cache=True.")
                           present_key_value = None # Assign None if cache is not found/returned correctly
                 next_decoder_cache.append(present_key_value)


            if output_attentions:
                 # Assuming structure (hidden, self_attn(optional), cross_attn(optional), cache(optional))
                 # Adjust indices based on actual layer output structure
                 self_attn_weights = layer_outputs[1] if len(layer_outputs) > 1 and layer_outputs[1] is not None else None
                 cross_attn_weights = layer_outputs[2] if len(layer_outputs) > 2 and layer_outputs[2] is not None else None
                 if self_attn_weights is not None:
                     all_self_attns += (self_attn_weights,)
                 # Check if cross attention exists and was output
                 if cross_attn_weights is not None and encoder_hidden_states is not None:
                     all_cross_attns += (cross_attn_weights,)


        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Convert list of tuples to tuple of tuples if using cache
        next_cache = tuple(next_decoder_cache) if use_cache else None

        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                all_hidden_states,
                all_self_attns,
                all_cross_attns,
                next_cache, # Use the potentially converted cache
            ] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attns,
            past_key_values=next_cache, # Use the potentially converted cache
        )

