temporal.configs.feedforward_config
===================================

.. py:module:: temporal.configs.feedforward_config


Classes
-------

.. autoapisummary::

   temporal.configs.feedforward_config.FeedForwardConfig
   temporal.configs.feedforward_config.StandardFeedForwardConfig
   temporal.configs.feedforward_config.MoEFeedForwardConfig


Functions
---------

.. autoapisummary::

   temporal.configs.feedforward_config.feedforward_config_from_dict


Module Contents
---------------

.. py:class:: FeedForwardConfig

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`


   Base configuration for the feed-forward network sublayer.
   Specific FFN types should inherit from this class.


   .. py:attribute:: activation
      :type:  str
      :value: 'gelu'



   .. py:attribute:: dropout
      :type:  float
      :value: 0.1



   .. py:attribute:: bias
      :type:  bool
      :value: True



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: StandardFeedForwardConfig

   Bases: :py:obj:`FeedForwardConfig`


   Configuration for a standard feed-forward network.


   .. py:attribute:: type
      :type:  str
      :value: 'standard'



   .. py:attribute:: intermediate_size
      :type:  int


   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: MoEFeedForwardConfig

   Bases: :py:obj:`FeedForwardConfig`


   Configuration for a Mixture-of-Experts (MoE) feed-forward network.


   .. py:attribute:: num_experts
      :type:  int
      :value: 8



   .. py:attribute:: top_k
      :type:  int
      :value: 2



   .. py:attribute:: type
      :type:  str
      :value: 'moe'



   .. py:attribute:: expert_intermediate_size
      :type:  Optional[int]
      :value: None



   .. py:attribute:: load_balancing_coef
      :type:  float
      :value: 0.01



   .. py:attribute:: gate_dropout
      :type:  Optional[float]
      :value: None



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:function:: feedforward_config_from_dict(data: Dict[str, Any]) -> FeedForwardConfig

