temporal.modules.feedforward.moe
================================

.. py:module:: temporal.modules.feedforward.moe


Classes
-------

.. autoapisummary::

   temporal.modules.feedforward.moe.MoEFeedForward


Module Contents
---------------

.. py:class:: MoEFeedForward(hidden_size: int, intermediate_size: Optional[int] = None, activation: str = 'gelu', dropout: float = 0.1, num_experts: int = 8, top_k: int = 2, expert_intermediate_size: Optional[int] = None, bias: bool = True, load_balancing_coef: float = 0.01, gate_dropout: Optional[float] = None, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   A Mixture of Experts (MoE) Feed-Forward Network layer.

   This module replaces a standard Feed-Forward Network (FFN) with multiple
   parallel FFN "experts". A gating network learns to route each input token
   to a small subset (the top-k) of these experts. The final output for each
   token is a weighted average of the outputs from its assigned experts.

   This architecture can significantly reduce computational cost while
   increasing model capacity, as only a fraction of the experts are activated
   for any given input.

   An auxiliary load balancing loss is computed during training to encourage
   the gating network to distribute tokens evenly across all experts,
   preventing a state where only a few experts are consistently chosen.

   .. attribute:: d_model

      The hidden dimension of the model.

      :type: int

   .. attribute:: num_experts

      The total number of expert networks.

      :type: int

   .. attribute:: top_k

      The number of experts to route each token to.

      :type: int

   .. attribute:: load_balancing_coef

      The coefficient for the auxiliary loss.

      :type: float

   .. attribute:: gate

      The gating network that determines expert weights.

      :type: nn.Linear

   .. attribute:: experts

      The list of expert FFNs.

      :type: nn.ModuleList


   .. py:attribute:: d_model


   .. py:attribute:: num_experts
      :value: 8



   .. py:attribute:: top_k
      :value: 2



   .. py:attribute:: load_balancing_coef
      :value: 0.01



   .. py:attribute:: gate


   .. py:attribute:: gate_dropout


   .. py:attribute:: experts


   .. py:method:: forward(hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]

      Performs the forward pass for the MoE FFN layer.

      :param hidden_states: The input tensor of shape
                            `[BatchSize, SequenceLength, HiddenDim]`.
      :type hidden_states: torch.Tensor

      :returns:

                A tuple containing:
                    - The final hidden states after MoE processing, with the same shape as the input.
                    - The auxiliary load balancing loss (a scalar tensor) if in training
                      mode, otherwise None.
      :rtype: Tuple[torch.Tensor, Optional[torch.Tensor]]



