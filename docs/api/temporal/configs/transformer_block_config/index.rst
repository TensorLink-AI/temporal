temporal.configs.transformer_block_config
=========================================

.. py:module:: temporal.configs.transformer_block_config


Classes
-------

.. autoapisummary::

   temporal.configs.transformer_block_config.TransformerBlockConfig
   temporal.configs.transformer_block_config.EncoderBlockConfig
   temporal.configs.transformer_block_config.DecoderBlockConfig
   temporal.configs.transformer_block_config.AdaptivePatchTransformerBlockConfig


Functions
---------

.. autoapisummary::

   temporal.configs.transformer_block_config.transformer_block_config_from_dict


Module Contents
---------------

.. py:class:: TransformerBlockConfig

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`


   Base configuration for a generic transformer block.
   Specific block types should inherit from this class.


   .. py:attribute:: attention_config
      :type:  temporal.configs.attention_config.AttentionConfig


   .. py:attribute:: ffn_config
      :type:  temporal.configs.feedforward_config.FeedForwardConfig


   .. py:attribute:: normalization_config
      :type:  temporal.configs.normalization_config.NormalizationConfig


   .. py:attribute:: kwargs
      :type:  Dict[str, Any]


   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: EncoderBlockConfig

   Bases: :py:obj:`TransformerBlockConfig`


   Configuration for a transformer encoder block.


   .. py:attribute:: type
      :type:  str
      :value: 'default_encoder'



.. py:class:: DecoderBlockConfig

   Bases: :py:obj:`TransformerBlockConfig`


   Configuration for a transformer decoder block.


   .. py:attribute:: type
      :type:  str
      :value: 'default_decoder'



   .. py:attribute:: cross_attention_config
      :type:  Optional[temporal.configs.attention_config.AttentionConfig]
      :value: None



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: AdaptivePatchTransformerBlockConfig

   Bases: :py:obj:`TransformerBlockConfig`


   Configuration for an adaptive patch transformer block.


   .. py:attribute:: expansion_factor
      :type:  int


   .. py:attribute:: wrapped_block_type
      :type:  str


   .. py:attribute:: order
      :type:  str
      :value: 'split_first'



   .. py:attribute:: type
      :type:  str
      :value: 'adaptive_patch_transformer'



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:function:: transformer_block_config_from_dict(data: Dict[str, Any]) -> TransformerBlockConfig

