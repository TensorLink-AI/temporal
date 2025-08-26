temporal.modules.attentions.patterned_attention
===============================================

.. py:module:: temporal.modules.attentions.patterned_attention


Classes
-------

.. autoapisummary::

   temporal.modules.attentions.patterned_attention.PatternedMultiHeadAttention


Functions
---------

.. autoapisummary::

   temporal.modules.attentions.patterned_attention.combine_masks


Module Contents
---------------

.. py:function:: combine_masks(patt_mask: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor

   Combines the pattern mask with an optional existing attention mask.


.. py:class:: PatternedMultiHeadAttention(embed_dim: int, num_heads: int, pattern: Dict[str, Any], is_decoder: bool = False, is_cross_attention: bool = False, **kwargs)

   Bases: :py:obj:`temporal.modules.attentions.base_attention.FullAttention`


   Multi-head attention that applies a fixed, predefined pattern (e.g., local,
   sliding, dilated) to the attention matrix.


   .. py:attribute:: pattern_type


   .. py:attribute:: window_size


   .. py:attribute:: stride


   .. py:attribute:: dilation


   .. py:attribute:: global_indices


   .. py:attribute:: is_causal


   .. py:method:: compute_pattern_mask(seq_len: int, device: torch.device) -> torch.Tensor

      Generates the boolean attention mask using efficient vectorized operations.



   .. py:method:: forward(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]

      Overrides the forward pass to inject the pattern mask.
      The mask is created to ensure no rows are entirely masked out, preventing NaNs.



