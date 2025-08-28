temporal.configs.output_head_config
===================================

.. py:module:: temporal.configs.output_head_config


Classes
-------

.. autoapisummary::

   temporal.configs.output_head_config.OutputHeadConfig
   temporal.configs.output_head_config.GaussianOutputHeadConfig
   temporal.configs.output_head_config.DistPredOutputHeadConfig
   temporal.configs.output_head_config.QuantileRegressionOutputHeadConfig
   temporal.configs.output_head_config.MixtureOutputHeadConfig
   temporal.configs.output_head_config.TimeFlowOutputHeadConfig
   temporal.configs.output_head_config.StudentTOutputHeadConfig


Functions
---------

.. autoapisummary::

   temporal.configs.output_head_config.output_head_config_from_dict


Module Contents
---------------

.. py:class:: OutputHeadConfig

   Bases: :py:obj:`temporal.configs.base_config.BaseConfig`


   Configuration for the output head of the transformer model.


   .. py:attribute:: type
      :type:  str
      :value: 'linear'



   .. py:attribute:: output_size
      :type:  Optional[int]
      :value: None



   .. py:attribute:: kwargs
      :type:  Dict[str, Any]


   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: GaussianOutputHeadConfig

   Bases: :py:obj:`OutputHeadConfig`


   Configuration for a Gaussian (Normal) distribution output head.


   .. py:attribute:: type
      :type:  str
      :value: 'gaussian'



   .. py:attribute:: min_log_sigma
      :type:  float
      :value: -7.0



   .. py:attribute:: max_log_sigma
      :type:  float
      :value: 5.0



   .. py:attribute:: sigma_floor
      :type:  float
      :value: 0.0001



   .. py:attribute:: init_log_sigma
      :type:  Optional[float]
      :value: None



.. py:class:: DistPredOutputHeadConfig

   Bases: :py:obj:`OutputHeadConfig`


   Configuration for a distributional prediction head.


   .. py:attribute:: num_outputs
      :type:  int


   .. py:attribute:: feature_size
      :type:  int


   .. py:attribute:: type
      :type:  str
      :value: 'distpred'



   .. py:attribute:: use_tanh
      :type:  bool
      :value: False



   .. py:attribute:: tanh_scale
      :type:  float
      :value: 10.0



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: QuantileRegressionOutputHeadConfig

   Bases: :py:obj:`OutputHeadConfig`


   Configuration for a quantile regression output head.


   .. py:attribute:: num_quantiles
      :type:  int


   .. py:attribute:: feature_size
      :type:  int
      :value: 1



   .. py:attribute:: type
      :type:  str
      :value: 'quantile_regression'



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: MixtureOutputHeadConfig

   Bases: :py:obj:`OutputHeadConfig`


   Configuration for the output head of the transformer model.


   .. py:attribute:: components
      :type:  List[str]
      :value: []



   .. py:attribute:: feature_size
      :type:  int
      :value: 1



   .. py:attribute:: type
      :type:  str
      :value: 'mixture'



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



   .. py:property:: derived_output_size
      :type: int



   .. py:property:: num_components
      :type: int



.. py:class:: TimeFlowOutputHeadConfig

   Bases: :py:obj:`OutputHeadConfig`


   Configuration for a TimeFlow output head.


   .. py:attribute:: target_channels
      :type:  int


   .. py:attribute:: cond_channels
      :type:  int


   .. py:attribute:: num_blocks
      :type:  int


   .. py:attribute:: model_channels
      :type:  int


   .. py:attribute:: num_sampling_steps
      :type:  int
      :value: 10



   .. py:attribute:: type
      :type:  str
      :value: 'timeflow'



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:class:: StudentTOutputHeadConfig

   Bases: :py:obj:`OutputHeadConfig`


   Configuration for a Student's T-Distribution output head.


   .. py:attribute:: feature_size
      :type:  int


   .. py:attribute:: num_outputs
      :type:  int
      :value: 1



   .. py:attribute:: type
      :type:  str
      :value: 'student_t'



   .. py:method:: __post_init__()

      Method for post-initialization validation. Subclasses should override this.



.. py:function:: output_head_config_from_dict(data: Dict[str, Any]) -> OutputHeadConfig

