import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention
from temporal.modules.norm.rms_norm import RMSNorm


def lambda_init_fn(depth: int) -> float:
    """Calculates an initial value for the lambda gating parameter based on layer depth.

    This function implements an exponential decay schedule for the initial lambda
    value, which is used in the gating mechanism of the DifferentialAttention.

    Args:
        depth (int): The depth or index of the attention layer.

    Returns:
        float: The initial lambda value.
    """
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats Key and Value heads for Grouped-Query Attention.

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


def reshape_for_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reshapes an input tensor to accommodate multiple attention heads.

    Converts a tensor from `[batch_size, seq_len, embed_dim]` to
    `[batch_size, num_heads, seq_len, head_dim]`.

    Args:
        x (torch.Tensor): The input tensor.
        num_heads (int): The number of attention heads.

    Returns:
        torch.Tensor: The reshaped tensor.
    """
    B, T, D = x.shape
    head_dim = D // num_heads
    return x.view(B, T, num_heads, head_dim).permute(0, 2, 1, 3)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embedding to a query or key tensor.

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
    """Differential-Wist (DiffWist) attention mechanism.

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
        **kwargs,
    ):
        """Initializes the DifferentialAttention module.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of query heads.
            dropout (float): The dropout rate.
            is_decoder (bool): Whether the module is used in a decoder.
            is_cross_attention (bool): Whether the module is used for cross-attention.
            depth (int): The layer depth, used for initializing the gating parameter.
            num_kv_heads (Optional[int]): The number of key/value heads. If None,
                it defaults to `num_heads`.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            bias=False,
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

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rel_pos: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Performs the forward pass of the DifferentialAttention layer.

        Args:
            hidden_states (torch.Tensor): The input hidden states of shape
                `[batch_size, seq_len, embed_dim]`.
            rel_pos (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the
                cosine and sine components of the rotary embeddings.
            attention_mask (Optional[torch.Tensor]): An optional mask to apply to
                the attention scores. Defaults to None.
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): An
                optional tuple containing cached key and value states for
                autoregressive decoding. Defaults to None.
            output_attentions (bool): Whether to return the attention weights.
                Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                A tuple containing:
                - The attention output tensor of shape
                  `[batch_size, seq_len, embed_dim]`.
                - The gated attention weights (if `output_attentions` is True).
                - The updated key-value cache (`present_key_value`).
        """
        B, T, _ = hidden_states.shape
        cos, sin = rel_pos

        # Project and reshape
        q = self.q_proj(hidden_states).view(B, T, 2 * self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, 2 * self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, 2 * self.head_dim)

        # Apply rotary embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Transpose for attention and repeat for GQA
        q = q.transpose(1, 2)                         # [B, 2H, T, D]
        k = repeat_kv(k.transpose(1, 2), self.n_rep)  # [B, 2H, T, D]
        v = repeat_kv(v.transpose(1, 2), self.n_rep)  # [B, 2H, T, 2D]

        # Handle KV caching
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

        # Apply learnable gating
        attn_weights = attn_weights.view(B, self.num_heads, 2, T, -1)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        gated_weights = attn_weights[:, :, 0] - (lambda_1 - lambda_2 + self.lambda_init) * attn_weights[:, :, 1]

        # Compute weighted sum with value tensor
        v = v.view(B, self.num_heads, 2, -1, self.head_dim)
        attn_output = torch.einsum("bhts,bhstd->bhtd", gated_weights, v)

        # Post-processing
        attn_output = self.subln(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, gated_weights if output_attentions else None, present_key_value
