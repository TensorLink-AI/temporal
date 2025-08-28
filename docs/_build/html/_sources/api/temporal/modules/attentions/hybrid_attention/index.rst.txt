temporal.modules.attentions.hybrid_attention
============================================

.. py:module:: temporal.modules.attentions.hybrid_attention


Classes
-------

.. autoapisummary::

   temporal.modules.attentions.hybrid_attention.HybridAttention


Module Contents
---------------

.. py:class:: HybridAttention(embed_dim: int, num_heads: int, dropout: float = 0.1, is_decoder: bool = False, is_cross_attention: bool = False, bias: bool = True, head_splits: Optional[List[int]] = None, head_types: Optional[List[str]] = None, head_agg: str = 'concat', head_agg_kwargs: Optional[dict] = None, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   Implements a hybrid multi-head attention mechanism.

   This module allows different groups of attention heads to use different
   attention kernel implementations (e.g., 'full', 'flash'). The outputs from
   these heterogeneous groups are then fused together to produce a single
   output tensor.


   .. py:attribute:: embed_dim


   .. py:attribute:: num_heads


   .. py:attribute:: head_dim


   .. py:attribute:: head_splits
      :value: None



   .. py:attribute:: group_count


   .. py:attribute:: group_embed_dims


   .. py:attribute:: out_proj


   .. py:attribute:: head_groups


   .. py:attribute:: head_agg
      :value: 'concat'



   .. py:attribute:: head_aggregator
      :value: None



   .. py:method:: forward(hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = None, past_key_value: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None, attention_mask: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[List[Optional[torch.Tensor]]], Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]]


