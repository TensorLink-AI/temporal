temporal.modules.norm.rms_norm
==============================

.. py:module:: temporal.modules.norm.rms_norm


Classes
-------

.. autoapisummary::

   temporal.modules.norm.rms_norm.RMSNorm


Module Contents
---------------

.. py:class:: RMSNorm(normalized_shape, eps=1e-08, elementwise_affine=True, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   Root Mean Square Layer Normalization (RMSNorm).

   Reference: https://arxiv.org/abs/1910.07467


   .. py:attribute:: normalized_shape


   .. py:attribute:: eps
      :value: 1e-08



   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      :param x: [B, ..., D] where D = normalized_shape

      :returns: Normalized tensor of same shape as x



