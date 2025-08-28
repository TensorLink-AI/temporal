temporal.configs.attention_config
=================================

.. py:module:: temporal.configs.attention_config


Classes
-------

.. autoapisummary::

   temporal.configs.attention_config.AttentionPatternConfig
   temporal.configs.attention_config.DestationaryProjectorConfig
   temporal.configs.attention_config.AttentionConfig
   temporal.configs.attention_config.FullAttentionConfig
   temporal.configs.attention_config.PatternedAttentionConfig
   temporal.configs.attention_config.FlashAttentionConfig
   temporal.configs.attention_config.LSEAttentionConfig
   temporal.configs.attention_config.DiffWistAttentionConfig
   temporal.configs.attention_config.HybridAttentionConfig


Functions
---------

.. autoapisummary::

   temporal.configs.attention_config.attention_config_from_dict


Module Contents
---------------

.. py:class:: AttentionPatternConfig

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`


   Base class for all immutable configuration dataclasses.
   Provides common serialization/deserialization methods.


   .. py:attribute:: type
      :type:  Literal['global', 'local', 'sliding', 'dilated']
      :value: 'global'



   .. py:attribute:: window_size
      :type:  int
      :value: 0



   .. py:attribute:: stride
      :type:  Optional[int]
      :value: None



   .. py:attribute:: dilation
      :type:  int
      :value: 1



   .. py:attribute:: global_indices
      :type:  Optional[List[int]]
      :value: []



.. py:class:: DestationaryProjectorConfig

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`


   Base class for all immutable configuration dataclasses.
   Provides common serialization/deserialization methods.


   .. py:attribute:: type
      :type:  str
      :value: 'destationary_projector'



   .. py:attribute:: hidden_dims
      :type:  List[int]
      :value: [64, 128]



   .. py:attribute:: hidden_layers
      :type:  int
      :value: 2



   .. py:attribute:: kernel_size
      :type:  int
      :value: 3



.. py:class:: AttentionConfig

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`


   Base configuration for a transformer attention mechanism.
   Specific attention types should inherit from this class.


   .. py:attribute:: num_heads
      :type:  int
      :value: 4



   .. py:attribute:: dropout
      :type:  float
      :value: 0.1



   .. py:attribute:: bias
      :type:  bool
      :value: True



   .. py:attribute:: qk_layernorm
      :type:  bool
      :value: False



   .. py:attribute:: destationary_projector
      :type:  Optional[DestationaryProjectorConfig]
      :value: None



   .. py:attribute:: kwargs
      :type:  Dict[str, Any]


   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: FullAttentionConfig

   Bases: :py:obj:`AttentionConfig`


   Configuration for standard multi-head attention.


   .. py:attribute:: type
      :type:  str
      :value: 'full'



   .. py:attribute:: use_rope
      :type:  bool
      :value: False



   .. py:attribute:: use_alibi
      :type:  bool
      :value: False



   .. py:attribute:: rope_base
      :type:  int
      :value: 10000



   .. py:attribute:: max_position_embeddings
      :type:  int
      :value: 4096



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: PatternedAttentionConfig

   Bases: :py:obj:`AttentionConfig`


   Configuration for attention with a fixed pattern (local, sliding, etc.),
   with support for RoPE and ALiBi.


   .. py:attribute:: type
      :type:  str
      :value: 'patterned'



   .. py:attribute:: pattern
      :type:  AttentionPatternConfig


   .. py:attribute:: use_rope
      :type:  bool
      :value: False



   .. py:attribute:: use_alibi
      :type:  bool
      :value: False



   .. py:attribute:: rope_base
      :type:  int
      :value: 10000



   .. py:attribute:: max_position_embeddings
      :type:  int
      :value: 4096



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: FlashAttentionConfig

   Bases: :py:obj:`AttentionConfig`


   Configuration for Flash Attention.


   .. py:attribute:: type
      :type:  str
      :value: 'flash'



   .. py:attribute:: softmax_scale
      :type:  Optional[float]
      :value: None



   .. py:attribute:: causal
      :type:  bool
      :value: False



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: LSEAttentionConfig

   Bases: :py:obj:`AttentionConfig`


   Configuration for LSE Attention.


   .. py:attribute:: type
      :type:  str
      :value: 'lse'



   .. py:attribute:: use_rope
      :type:  bool
      :value: False



   .. py:attribute:: use_alibi
      :type:  bool
      :value: False



   .. py:attribute:: rope_base
      :type:  int
      :value: 10000



   .. py:attribute:: max_position_embeddings
      :type:  int
      :value: 4096



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: DiffWistAttentionConfig

   Bases: :py:obj:`AttentionConfig`


   Configuration for Differential-Wist (DiffWist) attention.


   .. py:attribute:: type
      :type:  str
      :value: 'diffwist'



   .. py:attribute:: depth
      :type:  int
      :value: 1



   .. py:attribute:: num_kv_heads
      :type:  Optional[int]
      :value: None



   .. py:attribute:: use_rope
      :type:  bool
      :value: True



   .. py:attribute:: rope_base
      :type:  int
      :value: 10000



   .. py:attribute:: max_position_embeddings
      :type:  int
      :value: 4096



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: HybridAttentionConfig

   Bases: :py:obj:`AttentionConfig`


   Configuration for Hybrid Attention, allowing multiple attention kernel types.


   .. py:attribute:: type
      :type:  str
      :value: 'hybrid'



   .. py:attribute:: head_splits
      :type:  List[int]
      :value: []



   .. py:attribute:: head_types
      :type:  List[str]
      :value: []



   .. py:attribute:: head_agg
      :type:  str
      :value: 'concat'



   .. py:attribute:: head_agg_kwargs
      :type:  Optional[Dict[str, Any]]


   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:function:: attention_config_from_dict(data: Dict[str, Any]) -> AttentionConfig

