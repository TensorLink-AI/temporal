temporal.configs.normalization_config
=====================================

.. py:module:: temporal.configs.normalization_config


Classes
-------

.. autoapisummary::

   temporal.configs.normalization_config.NormalizationConfig
   temporal.configs.normalization_config.RevINConfig
   temporal.configs.normalization_config.DynamicRevINConfig
   temporal.configs.normalization_config.RevIN2dConfig


Functions
---------

.. autoapisummary::

   temporal.configs.normalization_config.normalization_config_from_dict


Module Contents
---------------

.. py:class:: NormalizationConfig

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`


   Configuration for normalization layers.


   .. py:attribute:: type
      :type:  str
      :value: 'layer'



   .. py:attribute:: eps
      :type:  float
      :value: 1e-05



   .. py:attribute:: elementwise_affine
      :type:  bool
      :value: True



   .. py:attribute:: kwargs
      :type:  Dict[str, Any]


   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: RevINConfig

   Bases: :py:obj:`NormalizationConfig`


   Configuration for Reversible Instance Normalization (RevIN).


   .. py:attribute:: type
      :type:  str
      :value: 'revin'



   .. py:attribute:: num_features
      :type:  int


   .. py:attribute:: affine
      :type:  bool
      :value: True



   .. py:attribute:: subtract_last
      :type:  bool
      :value: False



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: DynamicRevINConfig

   Bases: :py:obj:`NormalizationConfig`


   Configuration for Dynamic Reversible Instance Normalization (DynamicRevIN).


   .. py:attribute:: type
      :type:  str
      :value: 'dynamic_revin'



   .. py:attribute:: num_features
      :type:  int


   .. py:attribute:: affine_mode
      :type:  Union[str, Dict[str, Any]]


   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: RevIN2dConfig

   Bases: :py:obj:`NormalizationConfig`


   Configuration for 2D Reversible Instance Normalization (RevIN2d).


   .. py:attribute:: type
      :type:  str
      :value: 'revin2d'



   .. py:attribute:: num_features
      :type:  int


   .. py:attribute:: affine
      :type:  bool
      :value: True



   .. py:attribute:: subtract_last
      :type:  bool
      :value: False



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:function:: normalization_config_from_dict(data: Dict[str, Any]) -> NormalizationConfig

