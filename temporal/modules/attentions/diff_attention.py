import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention, apply_rotary_pos_emb
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

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of query heads.
            dropout (float): The dropout rate.
            is_decoder (bool): Whether the module is used in a decoder.
            is_cross_attention (bool): Whether the module is used for cross-attention.
            depth (int): The layer depth, used for initializing the gating parameter.
            num_kv_heads (Optional[int]): The number of key/value heads. If None,
                it defaults to `num_heads`.
            use_rope (bool): Whether to use RoPE.
            max_position_embeddings (int): The maximum sequence length for RoPE.
            rope_base (int): The base for RoPE frequencies.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            bias=False,
            **kwargs,
        )

        self.depth = depth
        self.num_kv_heads = num_kv_heads or num_heads
        self.n_rep = num_heads // self.num_kv_heads
        self.head_dim = embed_dim // num_heads

        # Redefine projections with modified dims for DiffWist
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim // 2) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim // 2) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim // 2) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim // 2) * 0.1)
        
        self.use_rope = use_rope
        self.rotary_proj = None
        if self.use_rope:
             self.rotary_proj = RotaryPositionalEmbedding(
                 d_model=self.head_dim,
                 max_seq_len=max_position_embeddings,
                 base=rope_base,
             )

        self.subln = RMSNorm(embed_dim, eps=1e-5)

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

        Args:
            hidden_states (torch.Tensor): The input hidden states of shape
                `[batch_size, seq_len, embed_dim]`.
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
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                A tuple containing:
                - The attention output tensor of shape
                  `[batch_size, seq_len, embed_dim]`.
                - The gated attention weights (if `output_attentions` is True).
                - The updated key-value cache (`present_key_value`).
        """
        B, T, _ = hidden_states.shape
        
        is_cross_attn = key_value_states is not None
        kv_source = key_value_states if is_cross_attn else hidden_states
        
        # Project and reshape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(kv_source).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(kv_source).view(B, T, self.num_kv_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply rotary embeddings if enabled
        if self.use_rope:
            if self.rotary_proj is None:
                raise ValueError("`rotary_proj` is not initialized. Pass `use_rope=True` to the constructor.")
            
            kv_seq_len = k.size(-2)
            cos, sin = self.rotary_proj(v)
            
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Repeat K,V for GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Handle KV caching
        present_key_value = None
        if use_cache:
            if past_key_value is not None:
                pk, pv = past_key_value
                k = torch.cat([pk, k], dim=-2)
                v = torch.cat([pv, v], dim=-2)
            present_key_value = (k, v)
        
        # Split heads for gating
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        v1, v2 = v.chunk(2, dim=-1)

        # Compute attention scores
        attn_weights1 = torch.matmul(q1, k1.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attn_weights2 = torch.matmul(q2, k2.transpose(-1, -2)) * (self.head_dim ** -0.5)

        if attention_mask is not None:
            attn_weights1 = attn_weights1 + attention_mask
            attn_weights2 = attn_weights2 + attention_mask

        attn_weights1 = F.softmax(attn_weights1, dim=-1)
        attn_weights2 = F.softmax(attn_weights2, dim=-1)
        
        if head_mask is not None:
            # This is a simple way to apply head mask, more complex logic might be needed
            # depending on the shape of the head_mask
            attn_weights1 = attn_weights1 * head_mask
            attn_weights2 = attn_weights2 * head_mask


        # Apply learnable gating
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        
        output1 = torch.matmul(attn_weights1, v1)
        output2 = torch.matmul(attn_weights2, v2)

        gated_output = output1 - (lambda_1 - lambda_2 + self.lambda_init) * output2
        
        attn_output = torch.cat([gated_output, gated_output], dim=-1)

        # Post-processing
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        
        attn_output = self.subln(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)
        
        attn_output = self.out_proj(attn_output)
        
        gated_weights = (attn_weights1, attn_weights2) if output_attentions else None

        return attn_output, gated_weights, present_key_value