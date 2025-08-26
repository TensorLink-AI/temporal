temporal.configs.head_aggregation_config
========================================

.. py:module:: temporal.configs.head_aggregation_config


Classes
-------

.. autoapisummary::

   temporal.configs.head_aggregation_config.HeadAggregationConfig


Functions
---------

.. autoapisummary::

   temporal.configs.head_aggregation_config.head_aggregation_config_from_dict


Module Contents
---------------

.. py:class:: HeadAggregationConfig

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`


   Configuration for combining multiple output heads.


   .. py:attribute:: type
      :type:  str
      :value: 'mean'



   .. py:attribute:: kwargs
      :type:  Dict[str, Any]


   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:function:: head_aggregation_config_from_dict(data: Dict[str, Any]) -> HeadAggregationConfig

