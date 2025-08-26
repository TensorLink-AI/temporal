temporal.modules.decoders.base_decoder_layer
============================================

.. py:module:: temporal.modules.decoders.base_decoder_layer


Classes
-------

.. autoapisummary::

   temporal.modules.decoders.base_decoder_layer.TimeSeriesTransformerDecoderLayer


Module Contents
---------------

.. py:class:: TimeSeriesTransformerDecoderLayer(config: temporal.configs.transformer_block_config.TransformerBlockConfig, builder: temporal.models.module_builder_helper.ModuleBuilder, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   A standard Transformer decoder layer for time series.

   This module implements a single layer of a Transformer decoder, which is
   a fundamental building block for sequence-to-sequence models in time series
   forecasting. It consists of three main components:
   1.  A masked self-attention mechanism to process the decoder's own input sequence.
   2.  An optional cross-attention mechanism to attend to the output of an encoder.
   3.  A feed-forward network (FFN).

   Each component is followed by a residual connection and layer normalization.
   The specific implementations of attention, FFN, and normalization are
   dynamically built based on the provided configuration.

   .. attribute:: config

      The configuration for this specific block.

      :type: TransformerBlockConfig

   .. attribute:: is_encoder_decoder

      Flag indicating if this layer is part of an
      encoder-decoder architecture, which determines if cross-attention is built.

      :type: bool

   .. attribute:: self_attn

      The self-attention module.

      :type: nn.Module

   .. attribute:: cross_attn

      The cross-attention module.

      :type: Optional[nn.Module]

   .. attribute:: ffn

      The feed-forward network.

      :type: nn.Module

   .. attribute:: norm1

      Layer normalization after self-attention.

      :type: nn.Module

   .. attribute:: norm2

      Layer normalization after cross-attention.

      :type: Optional[nn.Module]

   .. attribute:: norm3

      Layer normalization after the FFN.

      :type: nn.Module

   .. attribute:: dropout

      Dropout layer.

      :type: nn.Dropout


   .. py:attribute:: config


   .. py:attribute:: is_encoder_decoder


   .. py:attribute:: self_attn


   .. py:attribute:: cross_attn
      :value: None



   .. py:attribute:: ffn


   .. py:attribute:: norm1


   .. py:attribute:: norm2


   .. py:attribute:: norm3


   .. py:attribute:: dropout


   .. py:method:: forward(hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None, output_attentions: bool = False, use_cache: bool = False, x_raw: Optional[torch.Tensor] = None) -> temporal.models.outputs.DecoderLayerOutput


