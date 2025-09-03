temporal.configs.loss_config
============================

.. py:module:: temporal.configs.loss_config


Attributes
----------

.. autoapisummary::

   temporal.configs.loss_config.PROBABILISTIC_LOSSES
   temporal.configs.loss_config.PROBABILISTIC_LOSSES


Classes
-------

.. autoapisummary::

   temporal.configs.loss_config.LossConfig
   temporal.configs.loss_config.TimeSeriesLossConfig
   temporal.configs.loss_config.MSELossConfig
   temporal.configs.loss_config.CRPSLossConfig
   temporal.configs.loss_config.CRPSHuberLossConfig
   temporal.configs.loss_config.QuantileLossConfig
   temporal.configs.loss_config.TimeFlowLossConfig
   temporal.configs.loss_config.NLLLossConfig


Functions
---------

.. autoapisummary::

   temporal.configs.loss_config.loss_config_from_dict


Module Contents
---------------

.. py:data:: PROBABILISTIC_LOSSES
   :value: ['quantile', 'mq', 'crps', 'crps_huber']


.. py:class:: LossConfig

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`


   Base configuration for the loss function.


   .. py:attribute:: kwargs
      :type:  Dict[str, Any]


   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:data:: PROBABILISTIC_LOSSES
   :value: ['quantile', 'mq', 'crps', 'crps_huber']


.. py:class:: TimeSeriesLossConfig

   Bases: :py:obj:`LossConfig`


   Base configuration for the loss function.


   .. py:attribute:: type
      :type:  str
      :value: 'timeseries_generic'



   .. py:attribute:: loss_type
      :type:  str
      :value: 'mse'



   .. py:attribute:: quantiles
      :type:  Optional[List[float]]
      :value: None



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: MSELossConfig

   Bases: :py:obj:`LossConfig`


   Configuration for Mean Squared Error loss.


   .. py:attribute:: type
      :type:  str
      :value: 'mse'



.. py:class:: CRPSLossConfig

   Bases: :py:obj:`LossConfig`


   Configuration for Continuous Ranked Probability Score (CRPS) loss.


   .. py:attribute:: type
      :type:  str
      :value: 'crps'



   .. py:attribute:: reduction
      :type:  str
      :value: 'mean'



   .. py:attribute:: estimator
      :type:  str
      :value: 'pinball'



   .. py:attribute:: spread_lambda
      :type:  float
      :value: 0.0



   .. py:attribute:: spread_penalty_type
      :type:  str
      :value: 'log'



   .. py:attribute:: spread_penalty_epsilon
      :type:  float
      :value: 0.0



   .. py:attribute:: spread_target_spread
      :type:  float
      :value: 0.0



   .. py:attribute:: scaling_type
      :type:  str
      :value: 'none'



   .. py:attribute:: scaling_dim
      :type:  int
      :value: 1



   .. py:attribute:: scaling_eps
      :type:  float
      :value: 1e-08



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: CRPSHuberLossConfig

   Bases: :py:obj:`LossConfig`


   Configuration for Continuous Ranked Probability Score (CRPS) Huber loss.


   .. py:attribute:: type
      :type:  str
      :value: 'crps_huber'



   .. py:attribute:: huber_loss_threshold
      :type:  float
      :value: 0.0



   .. py:attribute:: reduction
      :type:  str
      :value: 'mean'



   .. py:attribute:: estimator
      :type:  str
      :value: 'pinball'



   .. py:attribute:: spread_lambda
      :type:  float
      :value: 0.0



   .. py:attribute:: spread_penalty_type
      :type:  str
      :value: 'log'



   .. py:attribute:: spread_penalty_epsilon
      :type:  float
      :value: 0.0



   .. py:attribute:: spread_target_spread
      :type:  float
      :value: 0.0



   .. py:attribute:: scaling_type
      :type:  str
      :value: 'none'



   .. py:attribute:: scaling_dim
      :type:  int
      :value: 1



   .. py:attribute:: scaling_eps
      :type:  float
      :value: 1e-08



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: QuantileLossConfig

   Bases: :py:obj:`LossConfig`


   Configuration for Quantile Loss.


   .. py:attribute:: type
      :type:  str
      :value: 'quantile'



   .. py:attribute:: quantiles
      :type:  List[float]
      :value: []



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: TimeFlowLossConfig

   Bases: :py:obj:`LossConfig`


   Configuration for the TimeFlow loss module.


   .. py:attribute:: type
      :type:  str
      :value: 'timeflow'



   .. py:attribute:: reduction
      :type:  str
      :value: 'mean'



.. py:class:: NLLLossConfig

   Bases: :py:obj:`LossConfig`


   Configuration for Negative Log Likelihood loss.


   .. py:attribute:: type
      :type:  str
      :value: 'nll'



   .. py:attribute:: kwargs
      :type:  Dict[str, Any]


   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:function:: loss_config_from_dict(data: Dict[str, Any]) -> LossConfig

