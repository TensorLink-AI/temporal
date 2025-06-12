# temporal/modules/attentions/time_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from temporal.registry.core import register_module
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention

# --- Import RoPE helper and Embedding Module from their locations ---
from temporal.modules.embedders.embedding import (
    apply_rotary_pos_emb,
    RotaryPositionalEmbedding # Import the class now used for RoPE
)
# --- End Import ---

# --- REMOVED RotaryProjection Class Definition ---

class BinaryAttentionBias(nn.Module):
    """A learnable relative position bias for attention scores.

    This module adds a bias to the attention scores based on the relative
    distance between key and query positions. The bias is learned during
    training.

    Attributes:
        num_heads (int): The number of attention heads.
        num_buckets (int): The number of buckets to group relative positions into.
        max_distance (int): The maximum relative distance to consider.
        is_decoder (bool): Whether this module is used in a decoder.
        relative_attention_bias (nn.Embedding): The learnable bias weights.
    """
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128, is_decoder: bool = True):
        """Initializes the BinaryAttentionBias module.

        Args:
            num_heads (int): The number of attention heads.
            num_buckets (int): The number of buckets for relative position bias.
            max_distance (int): The maximum distance for relative position bias.
            is_decoder (bool): Whether this module is used in a decoder.
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.is_decoder = is_decoder
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Calculates the relative position bucket for each relative position.

        Args:
            relative_position (torch.Tensor): A tensor of relative positions.

        Returns:
            torch.Tensor: A tensor of bucket indices.
        """
        bucket_indices = 0
        neg_relative_position = -relative_position
        num_buckets = self.num_buckets
        if not self.is_decoder:
            if num_buckets % 2 != 0: raise ValueError("num_buckets must be even for bidirectional bias.")
            half_buckets = num_buckets // 2
            bucket_indices += (neg_relative_position < 0).long() * half_buckets
            neg_relative_position = torch.abs(neg_relative_position)
        else:
             neg_relative_position = torch.max(neg_relative_position, torch.zeros_like(neg_relative_position))
             half_buckets = num_buckets

        max_exact = half_buckets // 2
        is_small = neg_relative_position < max_exact
        # Ensure denominators are not zero
        denom_log = math.log(self.max_distance / max_exact if max_exact > 0 else self.max_distance)
        denom_log = denom_log if denom_log > 1e-6 else 1.0 # Avoid log(1)=0 issues

        val_if_large = max_exact + (
            torch.log(neg_relative_position.float().clamp(min=1e-6) / max_exact if max_exact > 0 else neg_relative_position.float().clamp(min=1e-6))
            / denom_log
            * (half_buckets - max_exact)
        ).long()

        val_if_large = torch.min(val_if_large, torch.full_like(neg_relative_position, half_buckets - 1))
        bucket_indices += torch.where(is_small, neg_relative_position, val_if_large)
        bucket_indices = torch.min(bucket_indices, torch.tensor(num_buckets - 1, device=bucket_indices.device))
        return bucket_indices

    def compute_bias(self, query_length: int, key_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Computes the relative position bias tensor.

        Args:
            query_length (int): The length of the query sequence.
            key_length (int): The length of the key sequence.
            device (Optional[torch.device]): The device to create the tensor on.

        Returns:
            torch.Tensor: The relative position bias tensor.
        """
        relative_position = torch.arange(key_length, device=device)[None, :] - torch.arange(query_length, device=device)[:, None]
        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """Applies the relative position bias to the attention scores.

        Args:
            attn_scores (torch.Tensor): The attention scores.

        Returns:
            torch.Tensor: The attention scores with the bias applied.
        """
        batch_size, num_heads, query_length, key_length = attn_scores.shape
        bias = self.compute_bias(query_length, key_length, device=attn_scores.device)
        return attn_scores + bias


@register_module("attention", "time")
class TimeAttention(BaseMultiHeadAttention):
    """Time-aware attention using Rotary Positional Embeddings (RoPE) and
    relative position bias.

    This module uses RotaryPositionalEmbedding from embedding.py to generate
    cosine and sine positional embeddings and BinaryAttentionBias for the bias.

    Attributes:
        rotary_embed (RotaryPositionalEmbedding): The RoPE generator.
        rel_pos_bias (BinaryAttentionBias): The relative position bias module.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        bias: bool = True,
        max_position_embeddings: int = 2048, # Max length for RoPE/Bias
        rope_base: int = 10000, # Base for RoPE frequencies
        num_rel_pos_buckets: int = 32, # Buckets for relative position bias
        max_rel_pos_distance: int = 128, # Max distance for relative position bias
        **kwargs
    ):
        """Initializes the TimeAttention module.

        Args:
            embed_dim (int): The embedding dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            is_decoder (bool): Whether this module is used in a decoder.
            is_cross_attention (bool): Whether this module is used for cross-attention.
            bias (bool): Whether to include a bias in the linear projections.
            max_position_embeddings (int): The maximum sequence length for RoPE and bias.
            rope_base (int): The base for RoPE frequencies.
            num_rel_pos_buckets (int): The number of buckets for relative position bias.
            max_rel_pos_distance (int): The maximum distance for relative position bias.
            **kwargs: Additional keyword arguments for the parent class.
        """
        super().__init__(embed_dim, num_heads, dropout, is_decoder, is_cross_attention, bias, **kwargs)

        # Initialize RoPE generator (using the embedding module class)
        if self.head_dim % 2 != 0:
             raise ValueError("RoPE requires head_dim to be even.")
        self.rotary_embed = RotaryPositionalEmbedding(
            d_model=self.head_dim,
            max_seq_len=max_position_embeddings,
            base=rope_base
        )

        # Initialize Relative Position Bias
        self.rel_pos_bias = BinaryAttentionBias(
            num_heads=self.num_heads,
            num_buckets=num_rel_pos_buckets,
            max_distance=max_rel_pos_distance,
            is_decoder=is_decoder
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
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Performs the forward pass of the attention layer.

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

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                A tuple containing the attention output, the attention probabilities
                (if output_attentions is True), and the updated key and value states
                (if use_cache is True).
        """

        bsz, tgt_len, _ = hidden_states.size()
        is_cross_attn = key_value_states is not None
        kv_source = key_value_states if is_cross_attn else hidden_states
        src_len = kv_source.size(1)

        q = self.q_proj(hidden_states)
        k = self.k_proj(kv_source)
        v = self.v_proj(kv_source)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        present_key_value = None
        if use_cache:
            if past_key_value is not None:
                 past_k, past_v = past_key_value
                 k = torch.cat([past_k, k], dim=2)
                 v = torch.cat([past_v, v], dim=2)
            present_key_value = (k, v)

        total_kv_seq_len = k.size(2)

        # === Apply RoPE ===
        cos, sin = self.rotary_embed(v, seq_len=total_kv_seq_len)
        if is_cross_attn:
             q, _ = apply_rotary_pos_emb(q, k, cos[:tgt_len,:], sin[:tgt_len,:], position_ids=position_ids)
        else:
             q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

        # === Compute attention scores ===
        attn_scores = torch.matmul(q * self.scaling, k.transpose(-1, -2))

        # === Apply Relative Position Bias ===
        attn_scores = self.rel_pos_bias(attn_scores)

        # === Apply Attention Mask ===
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # === Compute probabilities, dropout, output ===
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        if head_mask is not None:
             if head_mask.dim() == 1: head_mask = head_mask[None, :, None, None]
             elif head_mask.dim() == 2: head_mask = head_mask[:, :, None, None]
             attn_probs = attn_probs * head_mask
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not output_attentions: attn_probs = None
        return attn_output, attn_probs, present_key_value
