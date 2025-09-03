temporal.models.module_builder_helper
=====================================

.. py:module:: temporal.models.module_builder_helper


Classes
-------

.. autoapisummary::

   temporal.models.module_builder_helper.ModuleBuilder


Module Contents
---------------

.. py:class:: ModuleBuilder(config: temporal.configs.transformer_model_config.TransformerTimeSeriesConfig)

   A helper class to build all primitive modules of a time series model.

   This class reads from a main configuration object and uses the module
   registry to instantiate the various components needed for the model,
   such as attention layers, embeddings, normalization layers, and loss functions.
   It encapsulates the logic for preparing arguments and handling dependencies
   between different configuration sections.

   .. attribute:: config

      The main `TransformerTimeSeriesConfig` object.


   .. py:attribute:: config


   .. py:method:: build_attention(cfg: temporal.configs.attention_config.AttentionConfig) -> torch.nn.Module

      Builds an attention module from an `AttentionConfig`.



   .. py:method:: build_feedforward(cfg: temporal.configs.feedforward_config.FeedForwardConfig) -> torch.nn.Module

      Builds a feed-forward network from a `FeedForwardConfig`.



   .. py:method:: build_value_embedding(cfg: temporal.configs.embedding_config.EmbeddingConfig) -> torch.nn.Module

      Builds the primary value embedding module.



   .. py:method:: build_positional_embedding(cfg: temporal.configs.embedding_config.EmbeddingConfig) -> torch.nn.Module

      Builds the positional embedding module.



   .. py:method:: build_normalization(cfg: temporal.configs.normalization_config.NormalizationConfig) -> torch.nn.Module

      Builds a normalization layer from a `NormalizationConfig`.



   .. py:method:: build_head_aggregator(cfg: temporal.configs.head_aggregation_config.HeadAggregationConfig) -> torch.nn.Module

      Builds a head aggregator module.



   .. py:method:: build_loss(cfg: temporal.configs.loss_config.LossConfig) -> torch.nn.Module

      Builds the loss function from a `LossConfig`.



