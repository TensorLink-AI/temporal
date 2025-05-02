import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, List

# Import ModuleBuilder from the new helper file
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.block_builder import BlockBuilder
from temporal.configs.transformer_config import TransformerBlockConfig


class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(self, config, builder: ModuleBuilder, block_configs: List[TransformerBlockConfig]):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerdrop = getattr(config, "encoder_layerdrop", 0.0)

        self.layernorm_embedding = builder.build_normalization()
        self.value_embedding = builder.build_value_embedding()
        # Ensure positional embedding supports explicit (batch, seq_len) call
        self.positional_embedding = builder.build_positional_embedding()
        block_builder = BlockBuilder(config, builder)
        self.layers = nn.ModuleList([
            block_builder.build_block(cfg) for cfg in block_configs
        ])

    def _get_tensor_dim_as_int(self, tensor: torch.Tensor, dim: int) -> int:
        """ Safely extracts a dimension size, handling potential tensor dim values. """
        try:
            dim_size = tensor.shape[dim]
            if torch.is_tensor(dim_size):
                 if dim_size.numel() == 1:
                     return int(dim_size.item())
                 else:
                     print(f"Warning (Encoder): Dimension {dim} size is a tensor with {dim_size.numel()} elements. Using first.")
                     return int(dim_size[0].item())
            return int(dim_size)
        except (IndexError, TypeError) as e:
            print(f"Warning (Encoder): Failed to get dimension {dim} size as int: {e}. Returning 0.")
            return 0

    def forward(
        self,
        input_values: torch.FloatTensor,  # [B, L, F]
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutput:

        # === Embedding ===
        value_embeds = self.value_embedding(input_values)  # [B, L, D]

        # Explicitly get batch size and sequence length as integers
        batch_size = self._get_tensor_dim_as_int(input_values, 0)
        seq_len = self._get_tensor_dim_as_int(input_values, 1)

        # Generate positional encoding using the explicit signature
        # Encoder processes the whole sequence at once (past_key_values_length=0)
        try:
            pos_embed = self.positional_embedding(
                batch_size=batch_size,
                seq_len=seq_len,
                past_key_values_length=0 
            ) # Expected shape: [1, L, D]

            # Ensure pos_embed shape is broadcastable: [1, L, D]
            # The positional embedding should return [1, L, D] for broadcasting
            if pos_embed.shape[0] != 1:
                 raise ValueError(f"Positional embedding returned unexpected batch dim: {pos_embed.shape[0]}. Expected 1.")
            if pos_embed.shape[1] != seq_len:
                 raise ValueError(f"Positional embedding returned unexpected seq len dim: {pos_embed.shape[1]}. Expected {seq_len}.")

        except TypeError as e:
             if "positional_embedding()" in str(e) or "forward()" in str(e):
                 raise TypeError(
                     f"The positional embedding layer ({type(self.positional_embedding).__name__}) in Encoder "
                     f"does not support the signature `forward(self, batch_size, seq_len, past_key_values_length)`. "
                     f"Check its implementation or the builder logic." 
                 ) from e
             else:
                 raise e # Re-raise other TypeErrors
        except Exception as e:
            print(f"Error during positional embedding call in Encoder: {e}")
            raise e

        # Sum + norm + dropout
        # Broadcasting handles cases where pos_embed is [1, L, D]
        hidden_states = value_embeds + pos_embed
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # --- Transformer Layers --- 
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if self.training and torch.rand([]).item() < self.layerdrop:
                continue

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Encoder blocks typically only need hidden_states and attention_mask
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                     all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
             outputs = (hidden_states,) + (all_hidden_states,) + (all_attentions,)
             return tuple(output for output in outputs if output is not None)
             
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
