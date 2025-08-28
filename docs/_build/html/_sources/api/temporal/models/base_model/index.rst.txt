temporal.models.base_model
==========================

.. py:module:: temporal.models.base_model


Classes
-------

.. autoapisummary::

   temporal.models.base_model.BaseTemporalModel


Module Contents
---------------

.. py:class:: BaseTemporalModel(config, encoder: torch.nn.Module = None, decoder: torch.nn.Module = None, output_heads: torch.nn.Module = None, head_aggregator: callable = None, loss_fn: callable = None)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all full time series model architectures.

   This class defines the interface for temporal models, enforcing the
   implementation of core `forward` and `generate` methods. It also holds
   the model's configuration and main components.

   Saving and loading of this model, especially for Hugging Face compatibility,
   should be handled by the `save_hf` and `load_hf` functions in
   `temporal.utils.hf_accessors`, often used in conjunction with the
   `temporal.utils.hf_adapter.TimeSeriesTransformerModel` wrapper.

   .. attribute:: config

      The configuration object containing model hyperparameters.

   .. attribute:: encoder

      The encoder module of the model.

      :type: nn.Module or None

   .. attribute:: decoder

      The decoder module of the model.

      :type: nn.Module or None

   .. attribute:: output_heads

      The module(s) producing the final outputs.

      :type: nn.Module or None

   .. attribute:: head_aggregator

      A function or module to aggregate
      outputs from multiple heads if they exist.

      :type: callable or None

   .. attribute:: loss_fn

      The loss function to be used during training.

      :type: callable or None


   .. py:attribute:: config


   .. py:attribute:: encoder
      :value: None



   .. py:attribute:: decoder
      :value: None



   .. py:attribute:: output_heads
      :value: None



   .. py:attribute:: head_aggregator
      :value: None



   .. py:attribute:: loss_fn
      :value: None



   .. py:method:: forward(*args, **kwargs)
      :abstractmethod:


      Performs a forward pass through the model.

      This method defines the core computation of the model, from inputs to
      final outputs (before loss calculation). Subclasses must override this
      method.

      :raises NotImplementedError: This method must be implemented by a subclass.



   .. py:method:: generate(*args, **kwargs)
      :abstractmethod:


      Generates predictions or forecasts from the model in inference mode.

      This method defines the generation or sampling behavior of the model,
      which might involve autoregressive decoding or other sampling strategies.
      Subclasses must override this method.

      :raises NotImplementedError: This method must be implemented by a subclass.



   .. py:method:: save_pretrained(save_directory)


   .. py:method:: from_pretrained(model_name_or_path, **kwargs)
      :classmethod:



