temporal.configs.quantizer_config
=================================

.. py:module:: temporal.configs.quantizer_config


Classes
-------

.. autoapisummary::

   temporal.configs.quantizer_config.QuantizerConfig


Functions
---------

.. autoapisummary::

   temporal.configs.quantizer_config.quantizer_config_from_dict


Module Contents
---------------

.. py:class:: QuantizerConfig

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`


   Configuration for time series quantization.


   .. py:attribute:: type
      :type:  str
      :value: 'mean_std_bins'



   .. py:attribute:: vocab_size
      :type:  int
      :value: 4096



   .. py:attribute:: num_features
      :type:  int
      :value: 1



   .. py:attribute:: kwargs
      :type:  Dict[str, Any]


   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:function:: quantizer_config_from_dict(data: Dict[str, Any]) -> QuantizerConfig

