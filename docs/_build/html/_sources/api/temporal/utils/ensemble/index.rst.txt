temporal.utils.ensemble
=======================

.. py:module:: temporal.utils.ensemble


Classes
-------

.. autoapisummary::

   temporal.utils.ensemble.EnsembleSampler


Module Contents
---------------

.. py:class:: EnsembleSampler(model: torch.nn.Module, dropout_enabled: bool = True)

   A utility for generating Monte Carlo dropout-based ensembles from a model.

   This class provides a convenient context for generating an ensemble of
   predictions by running a model multiple times with dropout enabled during
   inference. It ensures that the model's dropout layers are returned to their
   original evaluation mode afterward, even if an error occurs during generation.

   .. attribute:: model

      The model from which to generate ensembles.

      :type: nn.Module

   .. attribute:: dropout_enabled

      A flag to control whether dropout should be
      activated. If False, the model will be run in its default mode.

      :type: bool


   .. py:attribute:: model


   .. py:attribute:: dropout_enabled
      :value: True



   .. py:method:: generate(ensemble_size: int, **generate_kwargs: Any) -> torch.Tensor

      Generates an ensemble of forecasts by running the model multiple times.

      If dropout is enabled, this method will temporarily switch the model's
      dropout layers to training mode for the duration of the generation process.

      :param ensemble_size: The number of samples to generate for the ensemble.
      :type ensemble_size: int
      :param \*\*generate_kwargs: Keyword arguments to be passed directly to the
                                  `model.generate()` method.
      :type \*\*generate_kwargs: Any

      :returns: A tensor containing the ensemble of predictions, stacked
                along a new dimension. The typical shape is `[B, N, T, Q]`, where `N`
                is the `ensemble_size`.
      :rtype: torch.Tensor



