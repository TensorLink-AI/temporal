temporal.modules.attentions.base_attention
==========================================

.. py:module:: temporal.modules.attentions.base_attention


Classes
-------

.. autoapisummary::

   temporal.modules.attentions.base_attention.BaseMultiHeadAttention
   temporal.modules.attentions.base_attention.FullAttention
   temporal.modules.attentions.base_attention.FlashAttention


Module Contents
---------------

.. py:class:: BaseMultiHeadAttention(embed_dim: int, num_heads: int, dropout: float = 0.1, is_decoder: bool = False, is_cross_attention: bool = False, bias: bool = True, destationary_projector: Optional[temporal.configs.attention_config.DestationaryProjectorConfig] = None, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   Base class for multi-head attention mechanisms.

   This class provides the common structure for query, key, and value
   projections, as well as the output projection. Subclasses should
   implement the core attention logic in the `forward` method.

   .. attribute:: embed_dim

      The embedding dimension of the model.

      :type: int

   .. attribute:: num_heads

      The number of attention heads.

      :type: int

   .. attribute:: dropout

      The dropout rate.

      :type: float

   .. attribute:: head_dim

      The dimension of each attention head.

      :type: int

   .. attribute:: is_decoder

      Whether this module is used in a decoder.

      :type: bool

   .. attribute:: is_cross_attention

      Whether this module is used for cross-attention.

      :type: bool

   .. attribute:: scaling

      The scaling factor for the attention scores.

      :type: float

   .. attribute:: q_proj

      The linear projection for the query.

      :type: nn.Linear

   .. attribute:: k_proj

      The linear projection for the key.

      :type: nn.Linear

   .. attribute:: v_proj

      The linear projection for the value.

      :type: nn.Linear

   .. attribute:: out_proj

      The linear projection for the output.

      :type: nn.Linear


   .. py:attribute:: embed_dim


   .. py:attribute:: num_heads


   .. py:attribute:: head_dim


   .. py:attribute:: scaling


   .. py:attribute:: q_proj


   .. py:attribute:: k_proj


   .. py:attribute:: v_proj


   .. py:attribute:: out_proj


   .. py:attribute:: use_qk_layernorm


   .. py:attribute:: destationary_projector
      :value: None



   .. py:attribute:: dropout
      :value: 0.1



   .. py:attribute:: is_decoder
      :value: False



   .. py:attribute:: is_cross_attention
      :value: False



   .. py:method:: compute_attention_scores(q: torch.Tensor, k: torch.Tensor, x_raw: torch.Tensor = None) -> torch.Tensor

      Computes the attention scores.

      :param q: The query tensor.
      :type q: torch.Tensor
      :param k: The key tensor.
      :type k: torch.Tensor

      :returns: The attention scores.
      :rtype: torch.Tensor



   .. py:method:: forward(hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, attention_mask: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False, use_cache: bool = False, position_ids: Optional[torch.LongTensor] = None, rotary_proj: Optional[torch.nn.Module] = None, alibi_bias_generator: Optional[torch.nn.Module] = None, x_raw: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]


.. py:class:: FullAttention(embed_dim: int, num_heads: int, **kwargs)

   Bases: :py:obj:`BaseMultiHeadAttention`


   Standard multi-head attention with optional RoPE and ALiBi.

   This class implements a standard multi-head attention mechanism using
   scaled dot-product attention. It can be configured to use Rotary
   Positional Embeddings (RoPE) and ALiBi positional embeddings.

   .. attribute:: use_rope

      Whether to use RoPE.

      :type: bool

   .. attribute:: use_alibi

      Whether to use ALiBi.

      :type: bool

   .. attribute:: rotary_proj

      The RoPE module.

      :type: Optional[RotaryPositionalEmbedding]

   .. attribute:: alibi_bias_generator

      The ALiBi bias
      generator.

      :type: Optional[ALiBiPositionalBias]


   .. py:attribute:: use_rope


   .. py:attribute:: use_alibi


   .. py:attribute:: rotary_proj
      :value: None



   .. py:attribute:: alibi_bias_generator
      :value: None



   .. py:method:: forward(hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, attention_mask: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False, use_cache: bool = False, position_ids: Optional[torch.LongTensor] = None, x_raw: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]

      Performs the forward pass of the attention layer.

      This method calls the `forward` method of the `BaseMultiHeadAttention`
      class, passing the RoPE and ALiBi modules if they are enabled.



.. py:class:: FlashAttention(embed_dim: int, num_heads: int, **kwargs)

   Bases: :py:obj:`BaseMultiHeadAttention`


   Flash attention implementation.

   This class implements multi-head attention using the `flash_attn` library,
   which provides a more efficient implementation of attention.

   .. attribute:: softmax_scale

      The softmax scale.

      :type: Optional[float]

   .. attribute:: causal

      Whether to use causal attention.

      :type: bool


   .. py:attribute:: softmax_scale


   .. py:attribute:: causal


   .. py:method:: forward(hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = None, past_key_value=None, attention_mask=None, head_mask=None, output_attentions=False, use_cache=False, position_ids=None, rotary_proj=None, alibi_bias_generator=None, x_raw: Optional[torch.Tensor] = None)

      Performs the forward pass of the attention layer using flash_attn_func.



