temporal.modules.encoders.transformer_encoder_layer
===================================================

.. py:module:: temporal.modules.encoders.transformer_encoder_layer


Classes
-------

.. autoapisummary::

   temporal.modules.encoders.transformer_encoder_layer.TimeSeriesTransformerEncoderLayer


Module Contents
---------------

.. py:class:: TimeSeriesTransformerEncoderLayer(config: temporal.configs.transformer_block_config.TransformerBlockConfig, builder: temporal.models.module_builder_helper.ModuleBuilder, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   A standard Transformer encoder layer for time series.

   This module implements a single layer of a Transformer encoder, which is a
   fundamental building block for sequence-to-sequence models in time series
   forecasting. It consists of two main components:
   1.  A self-attention mechanism to process the input sequence.
   2.  A feed-forward network (FFN).

   Each component is followed by a residual connection and layer normalization.
   The specific implementations of attention, FFN, and normalization are
   dynamically built based on the provided configuration.

   .. attribute:: config

      The configuration for this specific block.

      :type: TransformerBlockConfig

   .. attribute:: self_attn

      The self-attention module.

      :type: nn.Module

   .. attribute:: ffn

      The feed-forward network.

      :type: nn.Module

   .. attribute:: norm1

      Layer normalization after self-attention.

      :type: nn.Module

   .. attribute:: norm2

      Layer normalization after the FFN.

      :type: nn.Module

   .. attribute:: dropout

      Dropout layer.

      :type: nn.Dropout


   .. py:attribute:: config


   .. py:attribute:: self_attn


   .. py:attribute:: ffn


   .. py:attribute:: norm1


   .. py:attribute:: norm2


   .. py:attribute:: dropout


   .. py:method:: forward(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, output_attentions: Optional[bool] = False, x_raw: Optional[torch.Tensor] = None) -> temporal.models.outputs.EncoderLayerOutput

      Performs the forward pass of the encoder layer.

      :param hidden_states: The input to the layer of shape
                            `(batch, seq_len, embed_dim)`.
      :type hidden_states: torch.Tensor
      :param attention_mask: A mask to prevent attention
                             to padding tokens, shape `(batch, 1, seq_len, seq_len)`.
      :type attention_mask: Optional[torch.Tensor]
      :param output_attentions: Whether to return the attention
                                probabilities.
      :type output_attentions: Optional[bool]
      :param x_raw: Raw input for de-stationary attention.
      :type x_raw: Optional[torch.Tensor]

      :returns:

                An object containing the output hidden states,
                   optional attention weights, and optional auxiliary loss.
      :rtype: EncoderLayerOutput



