temporal.models.output_head_builder
===================================

.. py:module:: temporal.models.output_head_builder


Classes
-------

.. autoapisummary::

   temporal.models.output_head_builder.OutputHeadBuilder


Module Contents
---------------

.. py:class:: OutputHeadBuilder(config, builder: temporal.models.module_builder_helper.ModuleBuilder)

   Constructs the output head module for a time series model.

   This builder is responsible for instantiating the correct output head based on
   the model's configuration. It resolves the head's class from the registry
   and prepares the necessary arguments for its constructor, such as the model's
   hidden dimension and the required output dimension, which can vary depending
   on the head type (e.g., for point forecasts vs. quantile forecasts).

   .. attribute:: config

      The main configuration object for the model.

   .. attribute:: builder

      The main module builder helper.


   .. py:attribute:: config


   .. py:attribute:: builder


   .. py:method:: build() -> torch.nn.Module

      Instantiates and returns the configured output head module.

      This method resolves the appropriate output head class from the registry
      and prepares its constructor arguments. It validates that the necessary
      configuration values (e.g., `num_quantiles` for a quantile head) are
      present.

      :returns: An instantiated nn.Module representing the output head.

      :raises ValueError: If the configuration is missing necessary parameters
          for the selected head type.



