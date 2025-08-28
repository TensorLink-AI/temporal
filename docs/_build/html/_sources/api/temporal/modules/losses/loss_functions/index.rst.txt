temporal.modules.losses.loss_functions
======================================

.. py:module:: temporal.modules.losses.loss_functions


Classes
-------

.. autoapisummary::

   temporal.modules.losses.loss_functions.MQLoss
   temporal.modules.losses.loss_functions.WeightedQuantileLoss
   temporal.modules.losses.loss_functions.QuantileLoss
   temporal.modules.losses.loss_functions.KernelEnergyLoss
   temporal.modules.losses.loss_functions.EnergyDistanceLoss
   temporal.modules.losses.loss_functions.SpectralLoss
   temporal.modules.losses.loss_functions.FastSoftDTWLoss
   temporal.modules.losses.loss_functions.SpreadPenalty
   temporal.modules.losses.loss_functions.MixtureLoss


Module Contents
---------------

.. py:class:: MQLoss(quantiles: list, reduction: str = 'mean', use_crps: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   Computes the Multi-Quantile Loss (MQL) for probabilistic forecasting.

   This loss function computes the average pinball loss over a set of specified
   quantiles. It is a common metric for evaluating the accuracy of quantile
   forecasts.

   .. attribute:: quantiles

      The quantiles to be evaluated.

      :type: torch.Tensor

   .. attribute:: reduction

      The reduction method to apply to the final loss.

      :type: str

   .. attribute:: use_crps

      If True, adds a penalty term to approximate the CRPS.

      :type: bool


   .. py:attribute:: reduction
      :value: 'mean'



   .. py:attribute:: use_crps
      :value: False



   .. py:method:: forward(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor

      Calculates the Multi-Quantile Loss.

      :param preds: The predicted quantiles, shape `(B, T, Q)`.
      :type preds: torch.Tensor
      :param target: The ground truth values, shape `(B, T)`.
      :type target: torch.Tensor

      :returns: The computed loss, as a scalar or a tensor depending
                on the reduction method.
      :rtype: torch.Tensor



.. py:class:: WeightedQuantileLoss(quantiles: tuple = (0.1, 0.5, 0.9), epsilon: float = 1e-08, reduction: str = 'mean')

   Bases: :py:obj:`torch.nn.Module`


   Computes the Weighted Quantile Loss (wQL).

   wQL is a variant of the quantile loss that is normalized by the sum of the
   absolute target values. This can be useful for stabilizing training when
   the target values have a large dynamic range.

   .. attribute:: quantiles

      The quantiles to be evaluated.

      :type: torch.Tensor

   .. attribute:: epsilon

      A small constant to prevent division by zero.

      :type: float

   .. attribute:: reduction

      The reduction method.

      :type: str


   .. py:attribute:: epsilon
      :value: 1e-08



   .. py:attribute:: reduction
      :value: 'mean'



   .. py:method:: forward(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor

      Calculates the Weighted Quantile Loss.

      :param preds: Predicted quantiles, shape `(B, T, Q)`.
      :type preds: torch.Tensor
      :param target: Ground truth values, shape `(B, T)`.
      :type target: torch.Tensor

      :returns: The computed loss.
      :rtype: torch.Tensor



.. py:class:: QuantileLoss(quantile: float, reduction: str = 'mean')

   Bases: :py:obj:`torch.nn.Module`


   Computes the Quantile Loss (also known as Pinball Loss).

   This loss function is used for quantile regression. It asymmetrically
   penalizes over- and under-prediction to encourage the model to output a
   specific quantile of the target distribution.

   .. attribute:: quantile

      The target quantile, between 0 and 1.

      :type: float

   .. attribute:: reduction

      The reduction method.

      :type: str


   .. py:attribute:: quantile


   .. py:attribute:: reduction
      :value: 'mean'



   .. py:method:: forward(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor

      Calculates the Quantile Loss.



.. py:class:: KernelEnergyLoss(reduction: str = 'mean')

   Bases: :py:obj:`torch.nn.Module`


   Computes a kernel-based energy distance loss for probabilistic forecasts.

   This loss function is a proper scoring rule that encourages the distribution
   of predicted samples to match the distribution of the true data. It is
   based on the energy distance between the two distributions.


   .. py:attribute:: reduction
      :value: 'mean'



   .. py:method:: forward(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor

      Calculates the kernel-based energy loss.

      :param preds: A tensor of predicted samples, shape `(B, T, N)`.
      :type preds: torch.Tensor
      :param targets: The ground truth values, shape `(B, T)`.
      :type targets: torch.Tensor

      :returns: The computed loss.
      :rtype: torch.Tensor



.. py:class:: EnergyDistanceLoss(reduction: str = 'mean')

   Bases: :py:obj:`torch.nn.Module`


   Computes the Energy Distance loss.

   This is another implementation of the energy distance, often used for
   evaluating the similarity of two distributions.


   .. py:attribute:: reduction
      :value: 'mean'



   .. py:method:: forward(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor

      Calculates the Energy Distance.



.. py:class:: SpectralLoss(reduction: str = 'mean')

   Bases: :py:obj:`torch.nn.Module`


   Computes a loss in the frequency domain.

   This loss function calculates the L2 distance between the Fast Fourier
   Transforms (FFTs) of the predictions and the targets. It encourages the
   model to match the frequency components of the target sequence.


   .. py:attribute:: reduction
      :value: 'mean'



   .. py:method:: forward(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor

      Calculates the spectral loss.



.. py:class:: FastSoftDTWLoss(gamma: float = 1.0, reduction: str = 'mean')

   Bases: :py:obj:`torch.nn.Module`


   Computes a differentiable approximation of Dynamic Time Warping (DTW).

   Soft-DTW is a differentiable loss function that measures the alignment
   between two time series. It can be useful for tasks where the sequences
   might be out of phase.

   .. attribute:: gamma

      The smoothing parameter. A lower gamma makes the loss
      closer to the non-differentiable DTW.

      :type: float


   .. py:attribute:: gamma
      :value: 1.0



   .. py:attribute:: reduction
      :value: 'mean'



   .. py:method:: forward(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor

      Calculates the Soft-DTW loss.



.. py:class:: SpreadPenalty(penalty_type: str = 'log', epsilon: float = 0.001, reduction: str = 'mean', target_spread: float = 0.0)

   Bases: :py:obj:`torch.nn.Module`


   Computes a penalty on the spread of predicted quantiles.

   This module is used to regularize the uncertainty of a probabilistic
   forecast. It can penalize excessively wide or narrow quantile ranges.


   .. py:attribute:: penalty_type
      :value: 'log'



   .. py:attribute:: epsilon
      :value: 0.001



   .. py:attribute:: reduction
      :value: 'mean'



   .. py:attribute:: target_spread
      :value: 0.0



   .. py:method:: forward(preds: torch.Tensor) -> torch.Tensor

      Calculates the spread penalty.

      :param preds: A tensor of quantile predictions, assumed to be
                    sorted along the last dimension, shape `[B, T, Q]`.
      :type preds: torch.Tensor

      :returns: The computed penalty.
      :rtype: torch.Tensor



.. py:class:: MixtureLoss(reduction='mean', min_df=2.0, fixed_sigma=0.001)

   Bases: :py:obj:`torch.nn.Module`


   Computes the Negative Log-Likelihood for a Mixture Density Network.

   This loss function is designed to work with the output of a
   `MixtureOutputHead`. It calculates the likelihood of the target values
   under a mixture of probability distributions, whose parameters are
   predicted by the model.

   .. attribute:: min_df

      The minimum degrees of freedom for a Student's T distribution.

      :type: float

   .. attribute:: fixed_sigma

      The fixed standard deviation for a Normal distribution
      if its sigma is not predicted.

      :type: float


   .. py:attribute:: reduction
      :value: 'mean'



   .. py:attribute:: min_df
      :value: 2.0



   .. py:attribute:: fixed_sigma
      :value: 0.001



   .. py:method:: forward(preds: dict, targets: torch.Tensor, loss_mask: torch.Tensor = None)

      Calculates the mixture loss.

      :param preds: A dictionary of predicted parameters from a `MixtureOutputHead`.
      :type preds: dict
      :param targets: The ground truth values.
      :type targets: torch.Tensor
      :param loss_mask: An optional mask for the loss.
      :type loss_mask: Optional[torch.Tensor]

      :returns: The final computed loss.
      :rtype: torch.Tensor



