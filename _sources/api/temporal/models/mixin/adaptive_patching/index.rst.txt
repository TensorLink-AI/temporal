temporal.models.mixin.adaptive_patching
=======================================

.. py:module:: temporal.models.mixin.adaptive_patching


Classes
-------

.. autoapisummary::

   temporal.models.mixin.adaptive_patching.MLP
   temporal.models.mixin.adaptive_patching.PatchSplitting
   temporal.models.mixin.adaptive_patching.PatchMerging


Module Contents
---------------

.. py:class:: MLP(input_dim: int, hidden_dim: int, output_dim: int, dropout_prob: float = 0.1, activation_fn: torch.nn.Module = nn.ReLU())

   Bases: :py:obj:`torch.nn.Module`


   A simple 2-layer Multi-Layer Perceptron (MLP) with ReLU activation and Dropout.
   Used for non-linear transformations within modules like PatchSplitting and PatchMerging.


   .. py:attribute:: fc1


   .. py:attribute:: activation_fn


   .. py:attribute:: dropout


   .. py:attribute:: fc2


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Performs the forward pass through the MLP.

      :param x: Input tensor of shape `[B, ..., input_dim]`.
      :type x: torch.Tensor

      :returns: Output tensor of shape `[B, ..., output_dim]`.
      :rtype: torch.Tensor



.. py:class:: PatchSplitting(input_dim: int, expansion_factor: int, use_mlp: bool = True, mlp_expansion_factor: int = 4)

   Bases: :py:obj:`torch.nn.Module`


   Adaptive Patching: Increase number of patches and decrease feature dim.

   If `use_mlp` is True, it applies a full 2-layer MLP to each patch's features
   before splitting them. This allows for a rich, non-linear transformation
   to prepare features for higher-resolution processing.


   .. py:attribute:: K


   .. py:attribute:: use_mlp
      :value: True



   .. py:attribute:: input_dim


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


.. py:class:: PatchMerging(input_dim: int, merge_factor: int, use_mlp: bool = True, mlp_expansion_factor: int = 4)

   Bases: :py:obj:`torch.nn.Module`


   Merge patches back.

   If `use_mlp` is True, it now uses a full 2-layer MLP to learn a rich,
   non-linear transformation of the merged patch features.


   .. py:attribute:: K


   .. py:attribute:: use_mlp
      :value: True



   .. py:attribute:: input_dim


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


