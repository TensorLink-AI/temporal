temporal.modules.attentions.lse_attention
=========================================

.. py:module:: temporal.modules.attentions.lse_attention


Classes
-------

.. autoapisummary::

   temporal.modules.attentions.lse_attention.LSEAttention


Module Contents
---------------

.. py:class:: LSEAttention(embed_dim: int, num_heads: int, dropout: float = 0.1, is_decoder: bool = False, is_cross_attention: bool = False, bias: bool = True, use_rope: bool = False, use_alibi: bool = False, max_position_embeddings: int = 4096, rope_base: int = 10000, qk_layernorm: bool = False, **kwargs)

   Bases: :py:obj:`temporal.modules.attentions.base_attention.FullAttention`


   Log-Sum-Exp (LSE) Attention.

   This implementation inherits from FullAttention to reuse the setup for
   projections, RoPE, and ALiBi. It overrides the core attention probability
   calculation with a memory-efficient LSE mechanism.

   This version is designed to be used with the refactored BaseMultiHeadAttention
   that includes a `_compute_attn_probs` method.


   .. py:method:: forward(hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, attention_mask: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False, use_cache: bool = False, position_ids: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]

      Performs the forward pass of the attention layer.

      This method calls the `forward` method of the `BaseMultiHeadAttention`
      class. The base class will handle projections, caching, RoPE, and ALiBi,
      and it will call this class's overridden `_compute_attn_probs` method
      to get the attention weights.



