temporal.configs.architecture_config
====================================

.. py:module:: temporal.configs.architecture_config


Classes
-------

.. autoapisummary::

   temporal.configs.architecture_config.TransformerArchitectureConfig


Module Contents
---------------

.. py:class:: TransformerArchitectureConfig

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`


   Configuration for transformer architecture specifying layer layout and weight sharing.


   .. py:attribute:: layout
      :type:  str
      :value: 'encoder-decoder'



   .. py:attribute:: num_encoder_layers
      :type:  int
      :value: 4



   .. py:attribute:: num_decoder_layers
      :type:  int
      :value: 2



   .. py:attribute:: share_weights
      :type:  bool
      :value: False



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



