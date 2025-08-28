temporal.modules.feedforward.standard
=====================================

.. py:module:: temporal.modules.feedforward.standard


Classes
-------

.. autoapisummary::

   temporal.modules.feedforward.standard.StandardFeedForward


Module Contents
---------------

.. py:class:: StandardFeedForward(hidden_size: int, intermediate_size: int, activation: str = 'gelu', dropout: float = 0.1, bias: bool = True, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   A standard two-layer feed-forward network (FFN).

   This module implements the position-wise feed-forward network found in
   standard Transformer architectures. It consists of two linear layers
   with a non-linear activation function in between.

   .. attribute:: fc1

      The first linear layer (expansion).

      :type: nn.Linear

   .. attribute:: fc2

      The second linear layer (contraction).

      :type: nn.Linear

   .. attribute:: activation_fn

      The non-linear activation function.

      :type: nn.Module

   .. attribute:: dropout

      The dropout layer.

      :type: nn.Dropout


   .. py:attribute:: fc1


   .. py:attribute:: activation_fn


   .. py:attribute:: fc2


   .. py:attribute:: dropout


   .. py:method:: forward(hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]

      Performs the forward pass of the FFN.

      :param hidden_states: The input tensor of shape
                            `[..., seq_len, hidden_size]`.
      :type hidden_states: torch.Tensor

      :returns:

                A tuple containing:
                    - The output tensor with the same shape as the input.
                    - An auxiliary loss, which is `None` for this standard FFN.
      :rtype: Tuple[torch.Tensor, Optional[torch.Tensor]]



