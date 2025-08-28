temporal.configs.embedding_config
=================================

.. py:module:: temporal.configs.embedding_config


Classes
-------

.. autoapisummary::

   temporal.configs.embedding_config.EmbeddingConfig
   temporal.configs.embedding_config.TimeSeriesValueEmbeddingConfig
   temporal.configs.embedding_config.FlexibleValueEmbeddingConfig
   temporal.configs.embedding_config.SinusoidalPositionalEmbeddingConfig
   temporal.configs.embedding_config.TimeSeriesPatchEmbeddingConfig
   temporal.configs.embedding_config.TimeSeriesGlobalEmbeddingConfig
   temporal.configs.embedding_config.RotaryPositionalEmbeddingConfig
   temporal.configs.embedding_config.LearnedAbsolutePositionalEmbeddingConfig
   temporal.configs.embedding_config.ShawRelativePositionalBiasConfig
   temporal.configs.embedding_config.FourierFeatureEmbeddingConfig
   temporal.configs.embedding_config.Time2VecEmbeddingConfig
   temporal.configs.embedding_config.ALiBiPositionalBiasConfig
   temporal.configs.embedding_config.BucketedRelativeBiasConfig
   temporal.configs.embedding_config.ConvolutionalPositionalEmbeddingConfig
   temporal.configs.embedding_config.TimeDeltaEmbeddingConfig
   temporal.configs.embedding_config.StackedPositionalEmbeddingConfig
   temporal.configs.embedding_config.NoneEmbeddingConfig
   temporal.configs.embedding_config.S4PositionalEmbeddingConfig
   temporal.configs.embedding_config.WaveletPositionalEmbeddingConfig


Functions
---------

.. autoapisummary::

   temporal.configs.embedding_config.embedding_config_from_dict


Module Contents
---------------

.. py:class:: EmbeddingConfig

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`


   Base configuration for embedding layers.
   Specific embedding types should inherit from this class.


   .. py:attribute:: dropout
      :type:  float
      :value: 0.1



   .. py:attribute:: embedding_dim
      :type:  Optional[int]
      :value: None



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: TimeSeriesValueEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for a simple linear value embedding.


   .. py:attribute:: type
      :type:  str
      :value: 'value'



   .. py:attribute:: feature_size
      :type:  int
      :value: 1



   .. py:attribute:: use_value_norm
      :type:  bool
      :value: False



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: FlexibleValueEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for a flexible value embedding.


   .. py:attribute:: input_dims
      :type:  Union[int, Sequence[int]]


   .. py:attribute:: type
      :type:  str
      :value: 'flexible_value'



   .. py:attribute:: proj_kwargs
      :type:  Dict[str, Any]


   .. py:attribute:: use_layer_norm
      :type:  bool
      :value: False



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: SinusoidalPositionalEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for fixed sinusoidal positional embeddings.


   .. py:attribute:: type
      :type:  str
      :value: 'sinusoidal'



   .. py:attribute:: max_seq_len
      :type:  int
      :value: 2048



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: TimeSeriesPatchEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for patch embeddings.


   .. py:attribute:: patch_size
      :type:  int


   .. py:attribute:: feature_size
      :type:  int


   .. py:attribute:: type
      :type:  str
      :value: 'patch'



   .. py:attribute:: output_patch_size
      :type:  Optional[int]
      :value: None



   .. py:attribute:: stride
      :type:  Optional[int]
      :value: None



   .. py:attribute:: pad_value
      :type:  float
      :value: 0.0



   .. py:attribute:: use_mlp
      :type:  bool
      :value: False



   .. py:attribute:: mlp_hidden_size
      :type:  Optional[int]
      :value: None



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: TimeSeriesGlobalEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for global embedding.


   .. py:attribute:: seq_len
      :type:  int


   .. py:attribute:: feature_size
      :type:  int


   .. py:attribute:: type
      :type:  str
      :value: 'global'



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: RotaryPositionalEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for Rotary Positional Embedding.


   .. py:attribute:: type
      :type:  str
      :value: 'rotary'



   .. py:attribute:: max_seq_len
      :type:  int
      :value: 2048



   .. py:attribute:: base
      :type:  int
      :value: 10000



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: LearnedAbsolutePositionalEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for learned absolute positional embeddings.


   .. py:attribute:: type
      :type:  str
      :value: 'learned_abs'



   .. py:attribute:: max_seq_len
      :type:  int
      :value: 2048



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: ShawRelativePositionalBiasConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for Shaw relative positional bias.


   .. py:attribute:: num_heads
      :type:  int


   .. py:attribute:: type
      :type:  str
      :value: 'relative_shaw'



   .. py:attribute:: max_distance
      :type:  int
      :value: 128



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: FourierFeatureEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for Fourier Feature Embedding.


   .. py:attribute:: type
      :type:  str
      :value: 'fourier'



   .. py:attribute:: num_features
      :type:  int
      :value: 16



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: Time2VecEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for Time2Vec embedding.


   .. py:attribute:: type
      :type:  str
      :value: 'time2vec'



   .. py:attribute:: use_cos
      :type:  bool
      :value: True



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: ALiBiPositionalBiasConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for ALiBi Positional Bias.


   .. py:attribute:: num_heads
      :type:  int


   .. py:attribute:: type
      :type:  str
      :value: 'alibi'



   .. py:attribute:: max_seq_len
      :type:  int
      :value: 2048



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: BucketedRelativeBiasConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for Bucketed Relative Bias.


   .. py:attribute:: num_heads
      :type:  int


   .. py:attribute:: type
      :type:  str
      :value: 'bucketed'



   .. py:attribute:: num_buckets
      :type:  int
      :value: 32



   .. py:attribute:: max_distance
      :type:  int
      :value: 128



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: ConvolutionalPositionalEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for Convolutional Positional Embedding.


   .. py:attribute:: type
      :type:  str
      :value: 'conv_pos'



   .. py:attribute:: kernel_size
      :type:  int
      :value: 3



   .. py:attribute:: max_seq_len
      :type:  int
      :value: 2048



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: TimeDeltaEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for TimeDelta Embedding.


   .. py:attribute:: type
      :type:  str
      :value: 'timedelta'



   .. py:attribute:: hidden_dim
      :type:  int
      :value: 64



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: StackedPositionalEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for a Stacked Positional Embedding.


   .. py:attribute:: type
      :type:  str
      :value: 'stacked_embedding'



   .. py:attribute:: embedding_configs
      :type:  List[EmbeddingConfig]
      :value: []



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: NoneEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for a placeholder None Embedding.


   .. py:attribute:: type
      :type:  str
      :value: 'none'



.. py:class:: S4PositionalEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for S4 Positional Embedding.


   .. py:attribute:: type
      :type:  str
      :value: 's4'



   .. py:attribute:: kernel_size
      :type:  int
      :value: 512



   .. py:attribute:: max_seq_len
      :type:  int
      :value: 4096



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: WaveletPositionalEmbeddingConfig

   Bases: :py:obj:`EmbeddingConfig`


   Configuration for Wavelet Positional Embedding.


   .. py:attribute:: type
      :type:  str
      :value: 'wavelet'



   .. py:attribute:: wavelet
      :type:  str
      :value: 'db4'



   .. py:attribute:: level
      :type:  int
      :value: 3



   .. py:attribute:: max_seq_len
      :type:  int
      :value: 2048



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:function:: embedding_config_from_dict(data: Dict[str, Any], **kwargs) -> EmbeddingConfig

