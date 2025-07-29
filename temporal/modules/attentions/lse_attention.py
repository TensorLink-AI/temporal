import torch
import torch.nn as nn
import torch.nn.functional as F # Import F for nn.functional.gelu
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention
from temporal.modules.embedders.embedding import (
    RotaryPositionalEmbedding,
    ALiBiPositionalBias,
    apply_rotary_pos_emb,
    rotate_half,
)


@register_module("attention", "lse")
class LSEAttention(BaseMultiHeadAttention):
    """
    LSEAttention (Log-Sum-Exp Attention) is a numerically stable self-attention mechanism
    that replaces the standard softmax with a combination of the Log-Sum-Exp trick and
    GELU activation. This approach aims to address issues like numerical instability
    and entropy collapse often observed in softmax-based attention, especially in
    long-term multivariate forecasting.
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
        use_rope: bool = False, # LSE should also support RoPE/ALiBi
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
        position_ids: Optional[torch.LongTensor] = None, # Unused for now, but kept for signature
        rotary_proj: Optional[nn.Module] = None, # Passed by FullAttention, but LSE has its own
        alibi_bias_generator: Optional[nn.Module] = None, # Passed by FullAttention, but LSE has its own
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs the forward pass of the LSE (Log-Sum-Exp) Attention mechanism.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            key_value_states (Optional[torch.Tensor]): The key and value states for
                cross-attention. Defaults to None.
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): The cached
                key and value states from previous steps. Defaults to None.
            attention_mask (Optional[torch.Tensor]): The attention mask.
                Defaults to None.
            head_mask (Optional[torch.Tensor]): The mask for attention heads.
                Defaults to None.
            output_attentions (bool): Whether to output attention probabilities.
                Defaults to False.
            use_cache (bool): Whether to use caching for the key and value states.
                Defaults to False.
            position_ids (Optional[torch.LongTensor]): The position IDs for RoPE.
                Defaults to None.
            rotary_proj (Optional[nn.Module]): The RoPE projection module.
                Defaults to None.
            alibi_bias_generator (Optional[nn.Module]): The ALiBi bias generator.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                A tuple containing the attention output, the attention probabilities
                (if output_attentions is True), and the updated key and value states
                (if use_cache is True).
        """
        B, T, _ = hidden_states.size()
        is_cross = key_value_states is not None
        kv_source = key_value_states if is_cross else hidden_states

        # project
        q = self.q_proj(hidden_states)                 # [B, T, E]
        k = self.k_proj(kv_source)
        v = self.v_proj(kv_source)

        # reshape to heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        
        # Determine sequence length for K, V correctly when past_key_value is involved
        k_len = (past_key_value[0].size(2) if past_key_value else kv_source.size(1))
        k = k.view(B, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        # optional Q/K LayerNorm
        if self.use_qk_layernorm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # handle caching
        present = None
        if use_cache:
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            present = (k, v)

        # --- Apply RoPE if provided (self.rotary_proj) ---
        if self.rotary_proj is not None:
            seq_len = k.size(-2) # Current sequence length including cached elements
            # RoPE's forward expects `x` (tensor to infer device/dtype) and `seq_len`
            # If using `apply_rotary_pos_emb`, it expects `cos` and `sin` directly.
            # Here, we generate cos/sin from self.rotary_proj
            cos, sin = self.rotary_proj(v, seq_len=seq_len) # Pass `v` for device/dtype inference
            
            # Apply RoPE to current query (new tokens) based on its current length `T`
            # and to all keys `k` (cached + current) if not cross-attention.
            # The `cos` and `sin` from rotary_proj are typically `[seq_len, dim]`
            # So we slice them for the current sequence length of `q`
            q_cos = cos[..., -T:, :]
            q_sin = sin[..., -T:, :]
            q_embed, k_embed = apply_rotary_pos_emb(q, k, q_cos, q_sin) # Apply to Q and K
            q = q_embed
            if not is_cross: 
                k = k_embed # Only apply to keys if self-attention


        # --- Compute scores ---
        # Note: self.scaling is already head_dim ** -0.5
        scores = torch.einsum("bhqd, bhkd -> bhqk", q, k) * self.scaling

        # --- ① Numerical-stable LSE ---
        a = scores.max(dim=-1, keepdim=True).values      # “a” in the paper
        lse = a + torch.log(torch.exp(scores - a).sum(dim=-1, keepdim=True))

        # --- ② GELU non-linearity ---
        # Apply GELU to the LSE values. This makes LSE attention non-linear.
        lse = F.gelu(lse) # Use F.gelu for consistency with nn.functional

        # --- Apply ALiBi bias if provided (self.alibi_bias_generator) ---
        if self.alibi_bias_generator is not None:
            # ALiBi bias is typically added *before* softmax, so add it here.
            # It generates bias for [1, num_heads, seq_len, seq_len]
            bias = self.alibi_bias_generator(batch_size=B, seq_len=k.size(-2))
            lse = lse + bias # Add bias to the LSE values

        # --- Apply attention_mask ---
        # attention_mask is typically 4D [B, 1, T, S] with -inf for masked.
        if attention_mask is not None:
            lse = lse + attention_mask


        # --- ③ Re-normalise like softmax ---
        # This is the "softmax" equivalent for LSE attention after the GELU
        # IMPORTANT: The paper suggests using original `scores` and `lse` to compute `probs`.
        probs = torch.exp(scores - lse)

        # --- Apply head mask & dropout ---
        if head_mask is not None:
            probs = probs * head_mask.view(1, -1, 1, 1) # Ensure head_mask is broadcastable

        # Renormalize after masking if needed, based on the original paper this is done after masking
        # This re-normalization step is crucial for LSE after masking, as masked elements are now 0.
        # Avoid division by zero if a row sums to zero (e.g., all masked)
        sum_probs = probs.sum(dim=-1, keepdim=True)
        # Add a small epsilon to the sum to prevent division by zero for fully masked rows.
        probs = F.dropout(probs / (sum_probs + 1e-9), p=self.dropout, training=self.training)


        # --- Output computation ---
        out = torch.einsum("bhqk, bhkd -> bhqd", probs, v)                  # [B, H, T, D]
        out = out.transpose(1, 2).reshape(B, T, -1)                         # [B, T, E]
        out = self.out_proj(out)

        # We need to return attention weights if output_attentions is True
        attn_weights = probs if output_attentions else None
        
        return out, attn_weights, present
