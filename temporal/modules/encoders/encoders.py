import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, List, Tuple, Union

# Import ModuleBuilder from the new helper file
from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.block_builder import BlockBuilder
from temporal.configs.transformer_config import TransformerBlockConfig
from temporal.models.outputs import EncoderLayerOutput

class TimeSeriesTransformerEncoder(nn.Module):
    """
    A flexible Transformer encoder built from a list of block configurations.

    This module serves as the main encoder component in a Transformer-based
    time series model. It dynamically constructs a stack of encoder layers
    based on a list of `TransformerBlockConfig` objects.

    The encoder is responsible for:
    - Adding positional information to already embedded features.
    - Sequentially processing the embedded sequence through its layers to
      create a rich contextual representation.

    Attributes:
        config: The main configuration object for the model.
        dropout (nn.Dropout): Dropout layer applied after embeddings.
        layernorm_embedding (nn.Module): Layer normalization applied to the embeddings.
        value_embedding (nn.Module): The module for embedding input features.
            Note: This is not used in the `forward` pass, but is held here to be
            used by the parent model.
        positional_embedding (nn.Module): The module for adding positional information.
        layers (nn.ModuleList): The stack of encoder layers.
    """
    def __init__(self, config, builder: ModuleBuilder, block_configs: List[TransformerBlockConfig]):
        """
        Initializes the TimeSeriesTransformerEncoder.

        Args:
            config: The main model configuration object.
            builder (ModuleBuilder): A helper class that constructs the various
                sub-modules (embeddings, normalization, etc.) based on the config.
            block_configs (List[TransformerBlockConfig]): A list of configurations,
                where each configuration defines a single encoder layer in the stack.
        """
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerdrop = getattr(config, "encoder_layerdrop", 0.0)

        self.layernorm_embedding = builder.build_normalization()
        self.value_embedding = builder.build_value_embedding()
        self.positional_embedding = builder.build_positional_embedding()
        block_builder = BlockBuilder(config, builder)
        self.layers = nn.ModuleList([
            block_builder.build_block(cfg) for cfg in block_configs
        ])

    def _get_tensor_dim_as_int(self, tensor: torch.Tensor, dim: int) -> int:
        """
        Safely extracts a tensor dimension size as an integer.

        Args:
            tensor (torch.Tensor): The tensor to inspect.
            dim (int): The dimension index.

        Returns:
            int: The size of the specified dimension.
        """
        try:
            dim_size = tensor.shape[dim]
            return int(dim_size)
        except (IndexError, TypeError) as e:
            print(f"Warning (Encoder): Failed to get dimension {dim} size as int: {e}. Returning 0.")
            return 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,  # [B, L, D] embedded features
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[BaseModelOutput, Tuple]:
        """
        Performs the forward pass of the Transformer encoder.

        Args:
            hidden_states (torch.FloatTensor): The embedded input features for the encoder,
                shape `[B, L, D]`. The calling model is responsible for performing
                value embedding (e.g., patching) before passing to this module.
            attention_mask (Optional[torch.Tensor]): A mask to prevent attention
                to padding tokens. This mask must match the sequence dimension of `hidden_states`.
            output_attentions (bool): Whether to return attention weights.
            output_hidden_states (bool): Whether to return all hidden states.
            return_dict (bool): Whether to return a structured model output.

        Returns:
            Union[BaseModelOutput, Tuple]: The encoder's output, either as a
            structured object or a tuple.
        """
        # === Embedding Layer ===
        value_embeds = hidden_states
        batch_size = self._get_tensor_dim_as_int(value_embeds, 0)
        seq_len = self._get_tensor_dim_as_int(value_embeds, 1)

        try:
            pos_embed = self.positional_embedding(
                batch_size=batch_size,
                seq_len=seq_len,
                past_key_values_length=0  # Encoder does not use KV cache
            )
            if pos_embed.shape[1] != seq_len:
                raise ValueError(f"Positional embedding returned incorrect sequence length: got {pos_embed.shape[1]}, expected {seq_len}.")
        except Exception as e:
            print(f"Error during positional embedding call in Encoder: {e}")
            raise e

        # Combine value and positional embeddings
        hidden_states = value_embeds + pos_embed
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # === End Embedding Layer ===

        # --- Transformer Layers ---
        all_hidden_states_collector = () if output_hidden_states else None
        all_attentions_collector = () if output_attentions else None
        total_aux_loss = None

        for layer in self.layers:
            # Support for layer dropping during training
            if self.training and torch.rand([]).item() < self.layerdrop:
                continue

            if output_hidden_states:
                all_hidden_states_collector += (hidden_states,)

            # Each layer now returns an EncoderLayerOutput object
            layer_outputs: EncoderLayerOutput = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            # Unpack layer results
            hidden_states = layer_outputs.hidden_states
            if layer_outputs.aux_loss is not None:
                if total_aux_loss is None:
                    total_aux_loss = layer_outputs.aux_loss
                else:
                    total_aux_loss += layer_outputs.aux_loss
            
            if output_attentions:
                attention_probs = layer_outputs.attention_weights
                if attention_probs is not None:
                    all_attentions_collector += (attention_probs,)

        if output_hidden_states:
            all_hidden_states_collector += (hidden_states,)

        if not return_dict:
             outputs = (hidden_states,)
             if output_hidden_states:
                 outputs += (all_hidden_states_collector,)
             if output_attentions:
                 outputs += (all_attentions_collector,)
             return tuple(output for output in outputs if output is not None)
        
        output = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states_collector,
            attentions=all_attentions_collector,
        )
        if total_aux_loss is not None:
            output.aux_loss = total_aux_loss
        return output
