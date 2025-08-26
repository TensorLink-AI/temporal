temporal.models.block_builder
=============================

.. py:module:: temporal.models.block_builder


Classes
-------

.. autoapisummary::

   temporal.models.block_builder.BlockBuilder


Module Contents
---------------

.. py:class:: BlockBuilder(config, module_builder)

   Builds various sub-modules required by encoder and decoder blocks.
   This class is intended to be used internally by the TimeSeriesTransformerEncoder
   and TimeSeriesTransformerDecoder modules.


   .. py:attribute:: config


   .. py:attribute:: module_builder


   .. py:method:: build_attention(attention_config: temporal.configs.attention_config.AttentionConfig) -> torch.nn.Module

      Builds an attention module based on the provided AttentionConfig.
      This method is called from within the TransformerEncoderLayer and TransformerDecoderLayer.



   .. py:method:: build_feedforward(ffn_config: temporal.configs.feedforward_config.FeedForwardConfig) -> torch.nn.Module

      Builds a feed-forward network based on the provided FeedForwardConfig.



   .. py:method:: build_normalization(norm_config: temporal.configs.normalization_config.NormalizationConfig) -> torch.nn.Module

      Builds a normalization layer based on the provided NormalizationConfig.



   .. py:method:: build_block(block_config: temporal.configs.transformer_block_config.TransformerBlockConfig) -> torch.nn.Module

      Builds a transformer block (e.g., encoder layer, decoder layer, or special block)
      based on the provided TransformerBlockConfig.

      This method delegates to the main module_builder's _build method,
      which handles the polymorphic instantiation and argument injection.



