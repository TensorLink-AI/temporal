import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention
from temporal.modules.embedders.embedding import (
    RotaryPositionalEmbedding,
    ALiBiPositionalBias,
    apply_rotary_pos_emb,
)


@register_module("attention", "lse")
class LSEAttention(BaseMultiHeadAttention):
    """
    LSEAttention (Log-Sum-Exp Attention) is a numerically stable self-attention mechanism
    that can incorporate non-linearities like GELU. This implementation corrects the
    order of operations to ensure stability by applying all score modifications *before*
    the LogSumExp normalization.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        bias: bool = True,
        qk_layernorm: bool = False,
        use_rope: bool = False,
        use_alibi: bool = False,
        max_position_embeddings: int = 4096,
        rope_base: int = 10000,
        **kwargs
    ):
        super().__init__(
            embed_dim, num_heads, dropout,
            is_decoder, is_cross_attention,
            bias,
            qk_layernorm=qk_layernorm,
            **kwargs
        )
        self.use_rope = use_rope
        self.use_alibi = use_alibi

        self.rotary_proj = None
        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError("RoPE requires head_dim to be even.")
            self.rotary_proj = RotaryPositionalEmbedding(
                d_model=self.head_dim,
                max_seq_len=max_position_embeddings,
                base=rope_base,
            )

        self.alibi_bias_generator = None
        if self.use_alibi:
            self.alibi_bias_generator = ALiBiPositionalBias(
                num_heads=self.num_heads,
                max_seq_len=max_position_embeddings,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs, # Absorb other potential arguments
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = hidden_states.size()
        is_cross = key_value_states is not None
        kv_source = key_value_states if is_cross else hidden_states

        # Project and reshape Q, K, V
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_len_for_view = past_key_value[0].size(2) if past_key_value else kv_source.size(1)
        k = self.k_proj(kv_source).view(B, k_len_for_view, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_source).view(B, k_len_for_view, self.num_heads, self.head_dim).transpose(1, 2)

        # Optional Q/K LayerNorm
        if self.use_qk_layernorm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Handle KV caching
        if use_cache:
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)

        # Apply RoPE if enabled
        if self.rotary_proj is not None:
            seq_len = k.size(-2)
            cos, sin = self.rotary_proj(v, seq_len=seq_len) # Use v for device/dtype
            q_len = q.size(-2)
            q_cos, q_sin = cos[..., -q_len:, :], sin[..., -q_len:, :]
            q = apply_rotary_pos_emb(q, cos=q_cos, sin=q_sin)
            k = apply_rotary_pos_emb(k, cos=cos, sin=sin)

        # --- CORRECTED LOGIC ---
        
        # 1. Compute initial scores
        scores = torch.einsum("bhqd, bhkd -> bhqk", q, k) * self.scaling

        # 2. Apply all score modifications *before* normalization
        if self.alibi_bias_generator is not None:
            alibi = self.alibi_bias_generator(batch_size=B, seq_len=k.size(-2))
            scores = scores + alibi

        if attention_mask is not None:
            scores = scores + attention_mask

        # Apply custom GELU non-linearity to the scores
        scores = F.gelu(scores)

        # 3. Compute stable LSE normalizer on the *final* scores
        a = scores.max(dim=-1, keepdim=True).values
        lse = a + torch.log(torch.exp(scores - a).sum(dim=-1, keepdim=True) + 1e-9) # Add epsilon

        # 4. Compute final probabilities
        probs = torch.exp(scores - lse)
        
        # 5. Apply dropout
        if head_mask is not None:
            probs = probs * head_mask.view(1, -1, 1, 1) # Broadcastable head mask
        probs = F.dropout(probs, p=self.dropout, training=self.training)

        # 6. Compute final output
        out = torch.einsum("bhqk, bhkd -> bhqd", probs, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        out = self.out_proj(out)

        # Prepare outputs
        attn_weights = probs if output_attentions else None
        present = (k, v) if use_cache else None

        return out, attn_weights, present