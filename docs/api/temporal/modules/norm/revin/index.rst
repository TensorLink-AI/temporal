temporal.modules.norm.revin
===========================

.. py:module:: temporal.modules.norm.revin


Classes
-------

.. autoapisummary::

   temporal.modules.norm.revin.RevIN
   temporal.modules.norm.revin.RevIN2d


Module Contents
---------------

.. py:class:: RevIN(num_features: int, eps=1e-05, affine=True, subtract_last=False)

   Bases: :py:obj:`torch.nn.Module`


   Reversible Instance Normalization for time-series.
   It is described in https://openreview.net/forum?id=cGDAkQo1C0p

   This implementation assumes the input tensor is of shape (N, L, C), where
   N is the batch size, L is the sequence length, and C is the number of features.
   Normalization is applied over the L dimension.

   :param num_features: The number of features or channels (C).
   :type num_features: int
   :param eps: A value added for numerical stability. Default: 1e-5.
   :type eps: float
   :param affine: If True, this module has learnable affine parameters. Default: True.
   :type affine: bool
   :param subtract_last: If True, subtracts the last element of the sequence from the input. Default: False.
   :type subtract_last: bool


   .. py:attribute:: num_features


   .. py:attribute:: eps
      :value: 1e-05



   .. py:attribute:: affine
      :value: True



   .. py:attribute:: subtract_last
      :value: False



   .. py:attribute:: mean
      :value: None



   .. py:attribute:: stdev
      :value: None



   .. py:attribute:: last
      :value: None



   .. py:method:: forward(x: torch.Tensor, mode: str, mask: Optional[torch.Tensor] = None)


   .. py:method:: transform(x: torch.Tensor, mask: Optional[torch.Tensor] = None)


   .. py:method:: inverse_transform(x)


.. py:class:: RevIN2d(num_features: int, eps=1e-05, affine=True, subtract_last=False)

   Bases: :py:obj:`torch.nn.Module`


   Reversible Instance Normalization for time-series with 2D spatial dimensions.
   It is described in https://openreview.net/forum?id=cGDAkQo1C0p

   :param num_features: The number of features or channels.
   :type num_features: int
   :param eps: A value added for numerical stability. Default: 1e-5.
   :type eps: float
   :param affine: If True, this module has learnable affine parameters. Default: True.
   :type affine: bool
   :param subtract_last: If True, subtracts the last element of the sequence from the input. Default: False.
   :type subtract_last: bool


   .. py:attribute:: num_features


   .. py:attribute:: eps
      :value: 1e-05



   .. py:attribute:: affine
      :value: True



   .. py:attribute:: subtract_last
      :value: False



   .. py:attribute:: mean
      :value: None



   .. py:attribute:: stdev
      :value: None



   .. py:attribute:: last
      :value: None



   .. py:method:: forward(x: torch.Tensor, mode: str, mask: Optional[torch.Tensor] = None)


   .. py:method:: transform(x: torch.Tensor, mask: Optional[torch.Tensor] = None)


   .. py:method:: inverse_transform(x)


