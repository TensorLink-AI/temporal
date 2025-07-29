import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
# Make sure to import the modified BaseMultiHeadAttention and FullAttention
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention, FullAttention
from temporal.modules.embedders.embedding import (
    RotaryPositionalEmbedding,
    ALiBiPositionalBias,
    rotate_half,
)


@register_module("attention", "lse")
class LSEAttention(FullAttention):
    """
    Log-Sum-Exp (LSE) Attention.

    This implementation inherits from FullAttention to reuse the setup for
    projections, RoPE, and ALiBi. It overrides the core attention probability
    calculation with a memory-efficient LSE mechanism.

    This version is designed to be used with the refactored BaseMultiHeadAttention
    that includes a `_compute_attn_probs` method.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        bias: bool = True,
        use_rope: bool = False,
        use_alibi: bool = False,
        max_position_embeddings: int = 4096,
        rope_base: int = 10000,
        qk_layernorm: bool = False,
        **kwargs
    ):
        """
        Initializes the LSEAttention module.

        All arguments are passed to the parent FullAttention class to handle
        the setup of linear layers, RoPE, and ALiBi modules.
        """
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            bias=bias,
            use_rope=use_rope,
            use_alibi=use_alibi,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
            qk_layernorm=qk_layernorm,
            **kwargs
        )

    def _compute_attn_probs(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Overrides the base method to implement LSE attention probabilities.

        This approach is memory-efficient and numerically stable.

        Args:
            scores (torch.Tensor): The raw attention scores of shape
                                   [batch, heads, query_seq_len, key_seq_len].

        Returns:
            torch.Tensor: The computed attention probabilities.
        """
        # --- 1. Numerically-stable Log-Sum-Exp ---
        # torch.logsumexp is an optimized function that handles the max subtraction
        # internally to prevent overflow, making it efficient.
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)

        # --- 2. GELU non-linearity ---
        # Apply the non-linear transformation as described in the LSE paper.
        lse = F.gelu(lse)

        # --- 3. Compute probabilities ---
        # This computes the final probabilities by subtracting the LSE term.
        # It creates one large tensor, which is a significant improvement
        # over the original implementation's three large tensors.
        probs = torch.exp(scores - lse)

        return probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs the forward pass of the attention layer.

        This method calls the `forward` method of the `BaseMultiHeadAttention`
        class. The base class will handle projections, caching, RoPE, and ALiBi,
        and it will call this class's overridden `_compute_attn_probs` method
        to get the attention weights.
        """
        return super(FullAttention, self).forward(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_ids=position_ids,
            # Pass the RoPE and ALiBi modules initialized by the parent class
            rotary_proj=self.rotary_proj,
            alibi_bias_generator=self.alibi_bias_generator,
        )