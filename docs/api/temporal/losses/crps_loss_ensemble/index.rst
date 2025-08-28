temporal.losses.crps_loss_ensemble
==================================

.. py:module:: temporal.losses.crps_loss_ensemble


Functions
---------

.. autoapisummary::

   temporal.losses.crps_loss_ensemble.crps_ensemble
   temporal.losses.crps_loss_ensemble.ensemble


Module Contents
---------------

.. py:function:: crps_ensemble(observations: torch.Tensor, forecasts: torch.Tensor, axis: int = -1, sorted_ensemble: bool = False, estimator: str = 'pwm', reduce: bool = True) -> torch.Tensor

   Estimate the Continuous Ranked Probability Score (CRPS) for a finite ensemble
   of forecasts. This code is adapted from DistPred-Forecast.

   :param observations: The observed values. Shape must be broadcastable to forecasts shape excluding
                        the ensemble dimension `axis`.
   :type observations: torch.Tensor
   :param forecasts: The ensemble predictions. Must have an ensemble dimension specified by `axis`.
   :type forecasts: torch.Tensor
   :param axis: The index of the ensemble dimension in `forecasts`. Defaults to -1 (last axis).
   :type axis: int
   :param sorted_ensemble: If True, indicates `forecasts` is already sorted in ascending order along the
                           ensemble dimension. Some estimators (e.g. 'pwm') assume sorted data, so we
                           automatically sort if needed and if estimator is 'pwm' or 'nrg'.
   :type sorted_ensemble: bool
   :param estimator:
                     Which CRPS estimator to use. Choose from:
                       - "nrg": energy (a.k.a. 'nrg') form
                       - "pwm": probability weighted moments form
                       - "fair": the 'fair CRPS' variant
                     Default is "pwm".
   :type estimator: str, optional
   :param reduce: If True (default), returns the mean CRPS across the entire batch. If False,
                  returns the CRPS for each element/timestep in the batch (shape [...]).
   :type reduce: bool, optional

   :returns: Either a scalar CRPS value (reduce=True) or per-element CRPS values (reduce=False).
   :rtype: torch.Tensor

   :raises ValueError: If estimator is invalid or shapes are incompatible.

   .. rubric:: Example

   >>> # forecasts [B, T, K], observations [B, T]
   >>> crps_ensemble(observations, forecasts, axis=-1)
   >>> # forecasts [B, K, T], observations [B, T]
   >>> crps_ensemble(observations, forecasts, axis=1)


.. py:function:: ensemble(obs: torch.Tensor, fct: torch.Tensor, estimator: str = 'pwm', reduce: bool = True) -> torch.Tensor

   Compute the CRPS for a finite ensemble using the chosen estimator.
   Assumes ensemble dimension is the *last* dimension for both obs and fct.

   :param obs: Observations, shape [..., 1] (ensemble dim last).
   :type obs: torch.Tensor
   :param fct: Forecast ensemble, shape [..., ensemble], sorted if needed.
   :type fct: torch.Tensor
   :param estimator: "nrg", "pwm", or "fair".
   :type estimator: str
   :param reduce: If True, return scalar mean. If False, return tensor with shape [...].
   :type reduce: bool

   :returns:     Either a scalar (mean) or tensor of CRPS values ([...]).
   :rtype: torch.Tensor


