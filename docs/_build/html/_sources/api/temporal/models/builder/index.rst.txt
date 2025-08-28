temporal.models.builder
=======================

.. py:module:: temporal.models.builder


Functions
---------

.. autoapisummary::

   temporal.models.builder.build_time_series_transformer


Module Contents
---------------

.. py:function:: build_time_series_transformer(config: temporal.configs.transformer_model_config.TransformerTimeSeriesConfig, generate_strategy: Optional[str] = None) -> temporal.models.base_model.BaseTemporalModel

   Constructs a full transformer-based time series model from a configuration object.

   This function serves as the main entry point for model creation. It takes a
   comprehensive configuration object and orchestrates the entire build process,
   including:
   1.  Validating the configuration (now handled by dataclass __post_init__).
   2.  Initializing the necessary builders (`ModuleBuilder`, `OutputHeadBuilder`).
   3.  Building the encoder and/or decoder stacks based on the specified architecture.
   4.  Building the output head(s).
   5.  Building the primary loss function.
   6.  Resolving the final model class (which defines the `generate` strategy)
       and instantiating it with all the built components.

   :param config: The complete configuration object
                  that defines the model's architecture, layers, and components.
   :type config: TransformerTimeSeriesConfig
   :param generate_strategy: If provided, this string overrides the
                             `model_type` from the config to select a specific generation class
                             from the `generate_registry`.
   :type generate_strategy: Optional[str]

   :returns: An instantiated and fully constructed time series model.
   :rtype: BaseTemporalModel

   :raises ValueError: If the configuration is invalid or missing required sections
       (e.g., `encoder_blocks` for an encoder-based architecture).
   :raises RuntimeError: If the loss function cannot be built from the configuration.


