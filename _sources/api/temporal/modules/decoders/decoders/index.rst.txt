temporal.modules.decoders.decoders
==================================

.. py:module:: temporal.modules.decoders.decoders


Classes
-------

.. autoapisummary::

   temporal.modules.decoders.decoders.TimeSeriesTransformerDecoder


Module Contents
---------------

.. py:class:: TimeSeriesTransformerDecoder(config, builder: temporal.models.module_builder_helper.ModuleBuilder, block_configs: List[temporal.configs.transformer_block_config.TransformerBlockConfig], **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   A flexible Transformer decoder built from a list of block configurations.

   This module serves as the main decoder component in a Transformer-based
   time series model. It dynamically constructs a stack of decoder layers
   based on a list of `TransformerBlockConfig` objects.

   The decoder is responsible for:
   - Sequentially processing an already-embedded sequence through its layers.
   - Handling the Key-Value (KV) cache for efficient autoregressive generation.

   The calling model, such as `TransformerTemporalModel`, is responsible for
   all preprocessing, including value and positional embeddings and normalization,
   via the `InputPreprocessor`.

   .. attribute:: config

      The main configuration object for the model.

   .. attribute:: layers

      The stack of decoder layers.

      :type: nn.ModuleList


   .. py:attribute:: config


   .. py:attribute:: layerdrop


   .. py:attribute:: layers


   .. py:method:: forward(hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.Tensor] = None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, output_attentions: bool = False, output_hidden_states: bool = False, use_cache: bool = False, return_dict: bool = True, x_raw: Optional[torch.Tensor] = None) -> Union[transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions, Tuple]


