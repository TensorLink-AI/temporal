import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, List, Tuple, Union

from temporal.models.module_builder_helper import ModuleBuilder
from temporal.models.block_builder import BlockBuilder
from temporal.configs.transformer_block_config import TransformerBlockConfig # Corrected import
from temporal.models.outputs import EncoderLayerOutput

class TimeSeriesTransformerEncoder(nn.Module):
    """
    A flexible Transformer encoder built from a list of block configurations.

    This module serves as the main encoder component in a Transformer-based
    time series model. It dynamically constructs a stack of encoder layers
    based on a list of `TransformerBlockConfig` objects.

    The encoder is responsible for:
    - Sequentially processing an already-embedded sequence through its layers to
      create a rich contextual representation.
    
    The calling model, such as `TransformerTemporalModel`, is responsible for
    all preprocessing, including value and positional embeddings and normalization,
    via the `InputPreprocessor`.

    Attributes:
        config: The main configuration object for the model.
        layers (nn.ModuleList): The stack of encoder layers.
    """
    def __init__(self, config, builder: ModuleBuilder, block_configs: List[TransformerBlockConfig]):
        """
        Initializes the TimeSeriesTransformerEncoder.

        Args:
            config: The main model configuration object.
            builder (ModuleBuilder): A helper class that constructs the various
                sub-modules for the layers based on the config.
            block_configs (List[TransformerBlockConfig]): A list of configurations,
                where each configuration defines a single encoder layer in the stack.
        """
        super().__init__()
        self.config = config
        self.layerdrop = getattr(config, "encoder_layerdrop", 0.0)

        block_builder = BlockBuilder(config, builder)
        self.layers = nn.ModuleList([
            block_builder.build_block(cfg) for cfg in block_configs
        ])

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
            hidden_states (torch.FloatTensor): The preprocessed input features for the
                encoder, shape `[B, L, D]`. The calling model is responsible for all
                embedding and normalization.
            attention_mask (Optional[torch.Tensor]): A mask to prevent attention
                to padding tokens.
            output_attentions (bool): Whether to return attention weights.
            output_hidden_states (bool): Whether to return all hidden states.
            return_dict (bool): Whether to return a structured model output.

        Returns:
            Union[BaseModelOutput, Tuple]: The encoder's output, either as a
            structured object or a tuple.
        """
        all_hidden_states_collector = () if output_hidden_states else None
        all_attentions_collector = () if output_attentions else None
        total_aux_loss = None

        for layer in self.layers:
            if self.training and torch.rand([]).item() < self.layerdrop:
                continue

            if output_hidden_states:
                all_hidden_states_collector += (hidden_states,)

            layer_outputs: EncoderLayerOutput = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

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
