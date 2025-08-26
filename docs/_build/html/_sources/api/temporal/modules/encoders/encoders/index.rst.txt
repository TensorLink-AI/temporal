temporal.modules.encoders.encoders
==================================

.. py:module:: temporal.modules.encoders.encoders


Classes
-------

.. autoapisummary::

   temporal.modules.encoders.encoders.TimeSeriesTransformerEncoder


Module Contents
---------------

.. py:class:: TimeSeriesTransformerEncoder(config, builder: temporal.models.module_builder_helper.ModuleBuilder, block_configs: List[temporal.configs.transformer_block_config.TransformerBlockConfig], **kwargs)

   Bases: :py:obj:`torch.nn.Module`


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

   .. attribute:: config

      The main configuration object for the model.

   .. attribute:: layers

      The stack of encoder layers.

      :type: nn.ModuleList


   .. py:attribute:: config


   .. py:attribute:: layerdrop


   .. py:attribute:: layers


   .. py:method:: forward(hidden_states: torch.FloatTensor, attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False, output_hidden_states: bool = False, return_dict: bool = True, x_raw: Optional[torch.Tensor] = None) -> Union[transformers.modeling_outputs.BaseModelOutput, Tuple]

      Performs the forward pass of the Transformer encoder.

      :param hidden_states: The preprocessed input features for the
                            encoder, shape `[B, L, D]`. The calling model is responsible for all
                            embedding and normalization.
      :type hidden_states: torch.FloatTensor
      :param attention_mask: A mask to prevent attention
                             to padding tokens.
      :type attention_mask: Optional[torch.Tensor]
      :param output_attentions: Whether to return attention weights.
      :type output_attentions: bool
      :param output_hidden_states: Whether to return all hidden states.
      :type output_hidden_states: bool
      :param return_dict: Whether to return a structured model output.
      :type return_dict: bool
      :param x_raw: Raw input for de-stationary attention.
      :type x_raw: Optional[torch.Tensor]

      :returns: The encoder's output, either as a
                structured object or a tuple.
      :rtype: Union[BaseModelOutput, Tuple]



