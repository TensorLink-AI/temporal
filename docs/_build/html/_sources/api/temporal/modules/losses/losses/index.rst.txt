temporal.modules.losses.losses
==============================

.. py:module:: temporal.modules.losses.losses


Classes
-------

.. autoapisummary::

   temporal.modules.losses.losses.BaseLoss
   temporal.modules.losses.losses.TimeSeriesLoss
   temporal.modules.losses.losses.CRPSLoss
   temporal.modules.losses.losses.NegativeLogLikelihoodLoss
   temporal.modules.losses.losses.CRPSHuberLoss
   temporal.modules.losses.losses.TimeFlowLoss


Functions
---------

.. autoapisummary::

   temporal.modules.losses.losses.huber_transform


Module Contents
---------------

.. py:class:: BaseLoss(reduction: str = 'mean')

   Bases: :py:obj:`torch.nn.Module`


   An abstract base class for time series loss functions.

   This class provides a common interface for all loss modules, including
   standardized handling of reduction ('mean', 'sum', 'none') and optional
   masking of loss values.

   .. attribute:: reduction

      The type of reduction to apply to the loss.

      :type: str


   .. py:attribute:: reduction
      :value: 'mean'



   .. py:method:: forward(preds: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor
      :abstractmethod:


      The forward pass for the loss calculation. Must be implemented by subclasses.



.. py:class:: TimeSeriesLoss(loss_type: str = 'mse', quantiles: Optional[List[float]] = None, reduction: str = 'mean', **kwargs)

   Bases: :py:obj:`BaseLoss`


   A generic wrapper for various standard time series loss functions.

   This module acts as a factory and wrapper, allowing for the selection of
   common loss functions like MSE, MAE, and Quantile Loss via a configuration
   string. It handles the instantiation of the appropriate underlying loss
   function and applies it during the forward pass.

   .. attribute:: loss_fn

      The underlying instantiated loss function module.


   .. py:attribute:: loss_type
      :value: 'mse'



   .. py:attribute:: quantiles
      :value: None



   .. py:method:: forward(preds: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor

      Calculates the loss for the given predictions and targets.

      :param preds: The model's predictions.
      :type preds: torch.Tensor
      :param targets: The ground truth values.
      :type targets: torch.Tensor
      :param loss_mask: An optional mask to apply to
                        the loss values.
      :type loss_mask: Optional[torch.Tensor]

      :returns: The final computed loss.
      :rtype: torch.Tensor



.. py:class:: CRPSLoss(reduction: str = 'mean', estimator: str = 'pwm', axis: int = -1, scaling_type: str = 'none', scaling_dim: int = 1, scaling_eps: float = 1e-08, spread_lambda: float = 0.0, spread_penalty_type: str = 'symmetric_log', spread_penalty_epsilon: float = 0.001, spread_target_spread: float = 0.0, **kwargs)

   Bases: :py:obj:`BaseLoss`


   Continuous Ranked Probability Score (CRPS) for ensemble/quantile forecasts.

   Accepted `preds` inputs:
     • Tensor: [B,T,S] or [B,T,F,S]  (S = ensemble/paths/quantiles)
     • Dict: one of {"paths","samples","members","quantiles"} -> Tensor
     • ForecastBundle-like: uses .quantiles (Tensor [B,T,F,Q]) if present,
       else .params["paths"] when available.

   Targets:
     • [B,T] (univariate) or [B,T,F] (multivariate). We broadcast F=1 as needed.

   Internally we normalize predictions to [B,T,F,S] with S as the last dim, sort
   along S, and call `crps_ensemble(..., axis=-1, reduce=False)`. The output
   elementwise CRPS is [B,T,F], then masking/scaling/reduction are applied.


   .. py:attribute:: estimator
      :value: 'pwm'



   .. py:attribute:: user_axis
      :value: -1



   .. py:attribute:: scaling_type
      :value: 'none'



   .. py:attribute:: scaling_dim
      :value: 1



   .. py:attribute:: scaling_eps
      :value: 1e-08



   .. py:attribute:: spread_lambda


   .. py:attribute:: spread_penalty_fn
      :value: None



   .. py:method:: forward(preds: Union[torch.Tensor, Dict[str, torch.Tensor], object], targets: torch.Tensor, loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor

      Compute CRPS with optional spread penalty, scaling, and reduction.

      :returns: Reduced loss per your `reduction` setting.
      :rtype: torch.Tensor



.. py:class:: NegativeLogLikelihoodLoss(distribution_type: str, reduction: str = 'mean', min_log_sigma: float = -20.0, max_log_sigma: float = 20.0, min_log_scale: float = -7.0, max_log_scale: float = 5.0, min_log_df: float = -2.0, max_log_df: float = 6.0, sigma_floor: float = 0.0001, df_floor: float = 1.001, mix_min_sigma: float = 1e-06, mix_min_scale: float = 1e-06, mix_min_df: float = 1.001, mix_lognorm_min_y: float = 1e-09, mix_nb_min_r: float = 1e-06, mix_nb_eps_p: float = 1e-06, mix_fixed_sigma: float = 0.001, **kwargs)

   Bases: :py:obj:`BaseLoss`


   NLL for probabilistic heads.

   Supports:
     - "gaussian"   : preds concat [mu, log_sigma]   -> [B,T,2C] or [B,2C]
     - "student_t"  : preds concat [mu, log_scale, log_df] -> [B,T,3C] or [B,3C]
     - "mixture"    : dict from MixtureOutputHead (univariate); mixture NLL computed inline

   targets: [B,T,C] or [B,T] (univariate) or [B,C] with T==1
   loss_mask: [B,T] or [B,T,C] (broadcasted in BaseLoss._apply_reduction)


   .. py:attribute:: distribution_type


   .. py:attribute:: min_log_sigma


   .. py:attribute:: max_log_sigma


   .. py:attribute:: min_log_scale


   .. py:attribute:: max_log_scale


   .. py:attribute:: min_log_df


   .. py:attribute:: max_log_df


   .. py:attribute:: sigma_floor


   .. py:attribute:: df_floor


   .. py:attribute:: mix_min_sigma


   .. py:attribute:: mix_min_scale


   .. py:attribute:: mix_min_df


   .. py:attribute:: mix_lognorm_min_y


   .. py:attribute:: mix_nb_min_r


   .. py:attribute:: mix_nb_eps_p


   .. py:attribute:: mix_fixed_sigma


   .. py:method:: forward(preds: Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[str]]]], targets: torch.Tensor, loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor

      The forward pass for the loss calculation. Must be implemented by subclasses.



.. py:function:: huber_transform(x: torch.Tensor, delta: float) -> torch.Tensor

   Huber‐style transform: quadratic for |x|<=delta, linear beyond.


.. py:class:: CRPSHuberLoss(estimator: str = 'pwm', axis: int = -1, huber_delta: Optional[float] = None, spread_lambda: float = 0.0, spread_penalty_type: str = 'symmetric_log', spread_penalty_epsilon: float = 0.001, spread_target_spread: float = 0.0, reduction: str = 'mean', **kwargs)

   Bases: :py:obj:`BaseLoss`


   CRPS Loss with an optional Huber‐style transform.

   :param estimator: which finite‐ensemble estimator to use ('pwm','nrg','fair')
   :param axis: ensemble dimension in preds (default last)
   :param huber_delta: if >0, applies a Huber transform with this knee;
                       if 0 or None, no transform (pure CRPS)
   :param spread_lambda / _type / _eps / _target: exactly as in your original CRPSLoss
   :param reduction: one of 'mean','sum','none'


   .. py:attribute:: estimator
      :value: 'pwm'



   .. py:attribute:: axis
      :value: -1



   .. py:attribute:: huber_delta


   .. py:attribute:: spread_lambda
      :value: 0.0



   .. py:attribute:: spread_penalty_fn
      :value: None



   .. py:method:: forward(preds: torch.Tensor, targets: torch.Tensor, loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor

      The forward pass for the loss calculation. Must be implemented by subclasses.



.. py:class:: TimeFlowLoss(reduction: str = 'mean')

   Bases: :py:obj:`BaseLoss`


   The TimeFlow loss module for temporal forecasting.
   Given target sequences and a conditioning vector z, computes a diffusion-style MSE loss.
   This loss is stateful and contains its own neural network.


   .. py:method:: forward(preds: torch.Tensor, targets: torch.Tensor, loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor

      The forward pass for the loss calculation. Must be implemented by subclasses.



