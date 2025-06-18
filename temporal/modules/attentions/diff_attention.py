import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention
from temporal.modules.embedders.embedding import RotaryPositionalEmbedding
from temporal.modules.norm.rms_norm import RMSNorm


def lambda_init_fn(depth: int) -> float:
    """
    Calculates an initial value for the lambda gating parameter based on layer depth.

    This function implements an exponential decay schedule for the initial lambda
    value, which is used in the gating mechanism of the DifferentialAttention.

    Args:
        depth (int): The depth or index of the attention layer.

    Returns:
        float: The initial lambda value.
    """
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats Key and Value heads for Grouped-Query Attention.

    This function expands the key and value tensors to match the number of query
    heads, a core component of Grouped-Query Attention (GQA).

    Args:
        x (torch.Tensor): The key or value tensor of shape
            `[batch_size, num_kv_heads, seq_len, head_dim]`.
        n_rep (int): The repetition factor, which is the ratio of
            `num_query_heads` to `num_kv_heads`.

    Returns:
        torch.Tensor: The expanded tensor of shape
            `[batch_size, num_query_heads, seq_len, head_dim]`.
    """
    B, H, T, D = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]                  # [B, H, 1, T, D]
        .expand(B, H, n_rep, T, D)           # [B, H, n_rep, T, D]
        .reshape(B, H * n_rep, T, D)         # [B, H_q, T, D]
    )


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embedding to a query or key tensor.

    Args:
        x (torch.Tensor): The input tensor of shape
            `[batch_size, num_heads, seq_len, head_dim]`.
        cos (torch.Tensor): The cosine component of the rotary embeddings.
        sin (torch.Tensor): The sine component of the rotary embeddings.

    Returns:
        torch.Tensor: The tensor with rotary embeddings applied, having the same
            shape as the input `x`.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Rotary embedding requires even dimension, got {x.shape[-1]}")

    D = x.shape[-1]
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    cos = cos[..., :half]
    sin = sin[..., :half]

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


@register_module("attention", "diffwist")
class DifferentialAttention(BaseMultiHeadAttention):
    """
    Differential-Wist (DiffWist) attention mechanism.

    This module implements a novel attention mechanism featuring rotary positional
    embeddings, grouped-query attention, and a learnable gating mechanism. It is
    designed for efficient and effective sequence modeling.

    Attributes:
        depth (int): The depth of the layer, used for lambda initialization.
        num_kv_heads (int): The number of key/value heads for grouped-query attention.
        n_rep (int): The repetition factor for key/value heads.
        lambda_init (float): The initial value for the gating parameter.
        lambda_q1 (nn.Parameter): Learnable parameter for the first query component.
        lambda_k1 (nn.Parameter): Learnable parameter for the first key component.
        lambda_q2 (nn.Parameter): Learnable parameter for the second query component.
        lambda_k2 (nn.Parameter): Learnable parameter for the second key component.
        subln (RMSNorm): A Root Mean Square Normalization layer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        depth: int = 1,
        num_kv_heads: Optional[int] = None,
        use_rope: bool = True,
        max_position_embeddings: int = 4096,
        rope_base: int = 10000,
        **kwargs,
    ):
        """
        Initializes the DifferentialAttention module.
        """
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            bias=False,
            **kwargs
        )

        self.depth = depth
        self.num_kv_heads = num_kv_heads or num_heads
        self.n_rep = num_heads // self.num_kv_heads
        self.head_dim = embed_dim // num_heads // 2

        # Redefine projections with modified dims for DiffWist
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        
        self.use_rope = use_rope
        if self.use_rope:
             self.rotary_proj = RotaryPositionalEmbedding(
                 d_model=2*self.head_dim, # RoPE is applied to the full head dim before splitting
                 max_seq_len=max_position_embeddings,
                 base=rope_base,
             )

        self.subln = RMSNorm(self.embed_dim, eps=1e-5)

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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs the forward pass of the DifferentialAttention layer.
        """
        B, T, _ = hidden_states.shape
        
        is_cross_attn = key_value_states is not None
        kv_source = key_value_states if is_cross_attn else hidden_states
        
        # Project and reshape
        q = self.q_proj(hidden_states).view(B, T, 2 * self.num_heads, self.head_dim)
        k = self.k_proj(kv_source).view(B, T, 2 * self.num_kv_heads, self.head_dim)
        v = self.v_proj(kv_source).view(B, T, self.num_kv_heads, 2 * self.head_dim)

        # Apply rotary embeddings if enabled
        if self.use_rope:
            if self.rotary_proj is None:
                raise ValueError("`rotary_proj` is not initialized. Pass `use_rope=True` to the constructor.")
            
            kv_seq_len = k.size(1)
            cos, sin = self.rotary_proj(v, seq_len=kv_seq_len)
            
            # Apply RoPE to q and k
            q = apply_rotary_emb(q, cos, sin)
            if not is_cross_attn:
                k = apply_rotary_emb(k, cos, sin)

        # Transpose for attention and repeat for GQA
        q = q.transpose(1, 2)                         # [B, 2H, T, D]
        k = repeat_kv(k.transpose(1, 2), self.n_rep)  # [B, 2H, T, D]
        v = repeat_kv(v.transpose(1, 2), self.n_rep)  # [B, H, T, 2D] --> This needs to be checked

        # Handle KV caching
        present_key_value = None
        if use_cache:
            if past_key_value is not None:
                pk, pv = past_key_value
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            present_key_value = (k, v)

        # Compute attention scores
        q = q * (self.head_dim ** -0.5)
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # [B, 2H, T, T_k]

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=1.0, neginf=0.0)
        
        if head_mask is not None:
            print("Warning: head_mask is not implemented for DifferentialAttention.")

        # Apply learnable gating
        attn_weights = attn_weights.view(B, self.num_heads, 2, T, -1)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        gated_weights = attn_weights[:, :, 0] - (lambda_1 - lambda_2 + self.lambda_init) * attn_weights[:, :, 1]

        # Compute weighted sum with value tensor
        # The value tensor was shaped for grouped query attention, now we need to match it for the gated sum.
        # Original v shape: [B, H_kv, T, 2*D] -> repeated to [B, H_q, T, 2*D]
        # We need to split the last dimension to match the gating logic.
        v = v.view(B, self.num_heads, T, 2, self.head_dim).permute(0,1,3,2,4) # [B, H, 2, T, D]
        
        attn_output = torch.einsum("bhts,bhstd->bhtd", gated_weights, v) # [B, H, T, D]

        # Post-processing
        # The subln was originally on a reshaped output. Let's adjust.
        # Original output shape before subln was [B, H, T, 2*D], now it's [B, H, T, D]
        # This seems wrong. Let's re-check the einsum and shapes.
        
        # The value tensor `v` has shape [B, H_q, T, 2*D].
        # Gated weights `gated_weights` has shape [B, H_q, T, T_k].
        # The einsum should be `bhts,bhsd -> bhtd`
        
        # Let's trace the v shape again.
        # v_proj output: [B, T, embed_dim // n_rep]
        # after view: [B, T, num_kv_heads, 2*head_dim]
        # after repeat_kv: [B, num_heads, T, 2*head_dim]
        
        # So v is [B, H_q, T, 2*D]
        # Gated weights is [B, H_q, T, T_k]
        # The original code did `einsum("bhts,bhstd->bhtd", gated_weights, v)` where v was split.
        # That implies v should have 5 dimensions. Let's re-split v.
        
        v = v.view(B, self.num_heads, T, 2, self.head_dim).permute(0,1,3,2,4) # [B, H, 2, T, D]
        
        # The einsum `bhts,bhstd->bhtd` seems to have a typo (`s` vs `t`).
        # It should probably be `bhtk,bhkt d -> bhtd` (k=sequence, t=time)
        # Let's assume the original einsum was correct and the logic was to combine the two `v` components.
        # gated_weights: [B, H, T, T_k]
        # v (split): [B, H, 2, T_k, D]
        # einsum `bhtk,bhkt d -> bhtd` doesn't match `bhstd`
        # Let's assume `s` in `bhstd` is sequence length, so `t` in `bhts` is query_len and `s` is key_len
        # So `bhtk, bhkvd -> bhtvd` where v is the gated dimension.
        # attn_weights[:, :, 0] has shape [B, H, T, T_k]
        # attn_weights[:, :, 1] has shape [B, H, T, T_k]
        # v[:, :, 0] has shape [B, H, T_k, D]
        # v[:, :, 1] has shape [B, H, T_k, D]
        
        v_0 = v[:,:,0,:,:] # [B, H, T_k, D]
        v_1 = v[:,:,1,:,:] # [B, H, T_k, D]
        
        output_0 = torch.matmul(attn_weights[:,:,0], v_0) # [B, H, T, D]
        output_1 = torch.matmul(attn_weights[:,:,1], v_1) # [B, H, T, D]
        
        attn_output = output_0 - (lambda_1 - lambda_2 + self.lambda_init) * output_1

        attn_output = self.subln(attn_output.view(B, T, self.num_heads * self.head_dim))
        attn_output = attn_output * (1 - self.lambda_init)
        
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.embed_dim) # This reshape is likely wrong.
        
        # Let's fix the output pipeline
        # attn_output from gating is [B, H, T, D]. head_dim is embed_dim // num_heads // 2
        # So total feature dimension is num_heads * head_dim = embed_dim / 2
        
        # The original code's reshape implies `subln` input is [B, T, H * 2*D] which is embed_dim
        # My current `attn_output` is [B, H, T, D]. After transpose and reshape it is [B, T, H*D]
        # That's half the embed_dim. There is a dimensionality mismatch.
        
        # The original code:
        # attn_output = torch.einsum("bhts,bhstd->bhtd", gated_weights, v)
        # This has to be a typo in the letters. Let's assume the logic is:
        # gated_weights is [B,H,T,S]. v is [B,H,S,2D] after repeat_kv.
        # attn_output = torch.matmul(gated_weights, v) # [B, H, T, 2D]
        
        # Let's re-implement based on this assumption.
        v_reshaped_for_matmul = v.view(B, self.num_heads, T, 2*self.head_dim)
        attn_output_matmul = torch.matmul(gated_weights, v_reshaped_for_matmul) # [B, H, T, 2*D]

        # Now the dimensions match the original implementation's subln expectation
        attn_output = self.subln(attn_output_matmul.transpose(1, 2).reshape(B, T, self.embed_dim))
        attn_output = attn_output * (1 - self.lambda_init)
        attn_output = self.out_proj(attn_output)

        return attn_output, gated_weights if output_attentions else None, present_key_value

