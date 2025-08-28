temporal.utils.hf_adapter
=========================

.. py:module:: temporal.utils.hf_adapter


Classes
-------

.. autoapisummary::

   temporal.utils.hf_adapter.TimeSeriesTransformerModel


Module Contents
---------------

.. py:class:: TimeSeriesTransformerModel(config: temporal.configs.transformer_model_config.TransformerTimeSeriesConfig)

   Bases: :py:obj:`transformers.PreTrainedModel`


       A Hugging Face-compatible wrapper for the custom TimeSeriesTransformer.

       This class acts as an adapter, allowing a `TimeSeriesTransformer` model to be
       used seamlessly within the Hugging Face ecosystem. By inheriting from
   g    `transformers.PreTrainedModel` and defining the `config_class`, this wrapper
       enables standard Hugging Face functionalities like `.from_pretrained()`,
       `.save_pretrained()`, and integration with the `Trainer` and `pipeline` APIs.

       The core logic is delegated to the underlying `TimeSeriesTransformer` instance,
       which is created during initialization.

       Attributes:
           config_class: Specifies the configuration class to be used with this model.
           base_model_prefix: A name for the core model attribute, used by Hugging Face.
           temporal: The instance of the underlying `TimeSeriesTransformer`.



   .. py:attribute:: config_class


   .. py:attribute:: base_model_prefix
      :value: 'time_series_transformer'



   .. py:attribute:: temporal


   .. py:method:: forward(input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> Any

      Delegates the forward pass to the underlying temporal model.

      :param input_values: The input tensor for the model.
      :type input_values: torch.Tensor
      :param attention_mask: An optional attention mask.
      :type attention_mask: Optional[torch.Tensor]
      :param \*\*kwargs: Any other keyword arguments to be passed to the underlying model.

      :returns: The output from the underlying `TimeSeriesTransformer`'s forward method.
      :rtype: Any



   .. py:method:: generate(input_values: torch.Tensor, prediction_length: Optional[int] = None, **kwargs: Any) -> Any

      Delegates the generation task to the underlying temporal model.

      This allows the model to be used for autoregressive forecasting tasks.

      :param input_values: The initial sequence to start generation from.
      :type input_values: torch.Tensor
      :param prediction_length: The number of time steps to forecast.
                                If None, it defaults to the `prediction_length` in the model's config.
      :type prediction_length: Optional[int]
      :param \*\*kwargs: Any other keyword arguments for the generation process.

      :returns: The generated forecast sequence from the underlying model.
      :rtype: Any



