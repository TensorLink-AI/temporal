temporal.modules.attentions.diff_attention
==========================================

.. py:module:: temporal.modules.attentions.diff_attention


Classes
-------

.. autoapisummary::

   temporal.modules.attentions.diff_attention.DifferentialAttention


Functions
---------

.. autoapisummary::

   temporal.modules.attentions.diff_attention.lambda_init_fn
   temporal.modules.attentions.diff_attention.repeat_kv


Module Contents
---------------

.. py:function:: lambda_init_fn(depth: int) -> float

   Calculates an initial value for the lambda gating parameter based on layer depth.

   This function implements an exponential decay schedule for the initial lambda
   value, which is used in the gating mechanism of the DifferentialAttention.

   :param depth: The depth or index of the attention layer.
   :type depth: int

   :returns: The initial lambda value.
   :rtype: float


.. py:function:: repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor

   Repeats Key and Value heads for Grouped-Query Attention.

   This function expands the key and value tensors to match the number of query
   heads, a core component of Grouped-Query Attention (GQA).

   :param x: The key or value tensor of shape
             `[batch_size, num_kv_heads, seq_len, head_dim]`.
   :type x: torch.Tensor
   :param n_rep: The repetition factor, which is the ratio of
                 `num_query_heads` to `num_kv_heads`.
   :type n_rep: int

   :returns:

             The expanded tensor of shape
                 `[batch_size, num_query_heads, seq_len, head_dim]`.
   :rtype: torch.Tensor


.. py:class:: DifferentialAttention(embed_dim: int, num_heads: int, dropout: float = 0.1, is_decoder: bool = False, is_cross_attention: bool = False, depth: int = 1, num_kv_heads: Optional[int] = None, use_rope: bool = True, max_position_embeddings: int = 4096, rope_base: int = 10000, **kwargs)

   Bases: :py:obj:`temporal.modules.attentions.base_attention.BaseMultiHeadAttention`


   Differential-Wist (DiffWist) attention mechanism.

   This module implements a novel attention mechanism featuring rotary positional
   embeddings, grouped-query attention, and a learnable gating mechanism. It is
   designed for efficient and effective sequence modeling.

   .. attribute:: depth

      The depth of the layer, used for lambda initialization.

      :type: int

   .. attribute:: num_kv_heads

      The number of key/value heads for grouped-query attention.

      :type: int

   .. attribute:: n_rep

      The repetition factor for key/value heads.

      :type: int

   .. attribute:: lambda_init

      The initial value for the gating parameter.

      :type: float

   .. attribute:: lambda_q1

      Learnable parameter for the first query component.

      :type: nn.Parameter

   .. attribute:: lambda_k1

      Learnable parameter for the first key component.

      :type: nn.Parameter

   .. attribute:: lambda_q2

      Learnable parameter for the second query component.

      :type: nn.Parameter

   .. attribute:: lambda_k2

      Learnable parameter for the second key component.

      :type: nn.Parameter

   .. attribute:: subln

      A Root Mean Square Normalization layer.

      :type: RMSNorm


   .. py:attribute:: depth
      :value: 1



   .. py:attribute:: num_kv_heads


   .. py:attribute:: n_rep


   .. py:attribute:: head_dim


   .. py:attribute:: q_proj


   .. py:attribute:: k_proj


   .. py:attribute:: v_proj


   .. py:attribute:: out_proj


   .. py:attribute:: lambda_init


   .. py:attribute:: lambda_q1


   .. py:attribute:: lambda_k1


   .. py:attribute:: lambda_q2


   .. py:attribute:: lambda_k2


   .. py:attribute:: use_rope
      :value: True



   .. py:attribute:: rotary_proj
      :value: None



   .. py:attribute:: subln


   .. py:method:: forward(hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, attention_mask: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False, use_cache: bool = False, position_ids: Optional[torch.LongTensor] = None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]

      Performs the forward pass of the DifferentialAttention layer.

      :param hidden_states: The input hidden states of shape
                            `[batch_size, seq_len, embed_dim]`.
      :type hidden_states: torch.Tensor
      :param key_value_states: The key and value states for
                               cross-attention. Defaults to None.
      :type key_value_states: Optional[torch.Tensor]
      :param past_key_value: The cached
                             key and value states from previous steps. Defaults to None.
      :type past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]
      :param attention_mask: The attention mask.
                             Defaults to None.
      :type attention_mask: Optional[torch.Tensor]
      :param head_mask: The mask for attention heads.
                        Defaults to None.
      :type head_mask: Optional[torch.Tensor]
      :param output_attentions: Whether to output attention probabilities.
                                Defaults to False.
      :type output_attentions: bool
      :param use_cache: Whether to use caching for the key and value states.
                        Defaults to False.
      :type use_cache: bool
      :param position_ids: The position IDs for RoPE.
                           Defaults to None.
      :type position_ids: Optional[torch.LongTensor]
      :param \*\*kwargs: Additional keyword arguments.

      :returns:     A tuple containing:
                    - The attention output tensor of shape
                      `[batch_size, seq_len, embed_dim]`.
                    - The gated attention weights (if `output_attentions` is True).
                    - The updated key-value cache (`present_key_value`).
      :rtype: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]



