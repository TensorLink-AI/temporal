# temporal/modules/attentions/time_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math # Import math for BinaryAttentionBias

from temporal.registry.core import register_module
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention # Removed expand_mask import, assuming it's unused or handled elsewhere

# --- Import RoPE helper from its new location ---
from temporal.modules.embedders.embedding import apply_rotary_pos_emb
# --- End Import ---

# --- REMOVED RoPE Helper function definitions: rotate_half, apply_rotary_pos_emb ---

class RotaryProjection(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE).
    Applies positional rotations to Queries and Keys.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device=None):
        """
        Args:
            dim (int): Dimension of the head (embedding dimension divided by number of heads).
            max_position_embeddings (int): Maximum sequence length.
            base (int): Base value for frequency calculation.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate inverse frequencies
        # Shape: [dim / 2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype) # Use inv_freq dtype

        # freqs shape: [max_seq_len, dim / 2]
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but following HF implementation:
        # freqs = torch.cat((freqs, freqs), dim=-1) # Shape: [max_seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1) # Shape: [max_seq_len, dim]

        # Register cos and sin caches
        # Shape: [max_seq_len, dim]
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # --- Modified forward to return cos/sin, not apply directly --- 
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the cosine and sine frequencies for RoPE.

        Args:
            x (torch.Tensor): A dummy tensor to determine the device and dtype.
            seq_len (int, optional): Sequence length. If None, uses max_position_embeddings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cos and sin caches sliced to seq_len.
                                               Shape: [seq_len, dim]
        """
        _seq_len = seq_len if seq_len is not None else self.max_position_embeddings

        # Ensure cache is large enough and on correct device/dtype
        # Use x.device and x.dtype
        if _seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device or self.cos_cached.dtype != x.dtype:
             self._set_cos_sin_cache(seq_len=_seq_len, device=x.device, dtype=x.dtype)

        # Return precomputed cos/sin values sliced to the required sequence length
        return self.cos_cached[:_seq_len, ...], self.sin_cached[:_seq_len, ...]

    # --- REMOVED direct application logic from forward --- 
    # Original logic that applied RoPE:
    # # Apply rotation
    # # Position IDs are implicitly 0 to seq_len-1 here
    # q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)
    # return q_rotated, k_rotated


class BinaryAttentionBias(nn.Module):
    """
    Learnable Relative Position Bias for attention scores.
    Adds a bias based on the relative distance between key and query positions.
    """
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128, is_decoder: bool = True):
        """
        Args:
            num_heads (int): Number of attention heads.
            num_buckets (int): Number of buckets to group relative distances.
            max_distance (int): Maximum relative distance considered.
            is_decoder (bool): If True, uses causal bucketing (only considers past positions).
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.is_decoder = is_decoder

        # Learnable bias table: maps bucket index to a bias value for each head
        # Shape: [num_buckets, num_heads]
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    def _relative_position_bucket(self, relative_position):
        """Translates relative position to a bucket number."""
        ret = 0
        n = -relative_position # More positive values denote farther past positions

        num_buckets = self.num_buckets # Use instance variable

        # Handle causal vs non-causal
        if not self.is_decoder: # Bidirectional attention
            # Check if num_buckets is even for bidirectional splitting
            if num_buckets % 2 != 0:
                raise ValueError("num_buckets must be even for bidirectional relative position bias.")
            half_buckets = num_buckets // 2
            ret += (n < 0).long() * half_buckets # Offset for future positions (n < 0)
            n = torch.abs(n)
        else: # Causal attention (decoder)
             n = torch.max(n, torch.zeros_like(n)) # Consider only past positions (n>=0)
             half_buckets = num_buckets # All buckets for past positions

        # Half of the buckets (or all if causal) are for exact distances near 0
        # Use half_buckets here
        max_exact = half_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically spaced larger distances
        # Avoid division by zero or log(0) if max_exact is 0 or n is max_exact
        val_if_large = max_exact + (
            torch.log(n.float().clamp(min=1e-6) / max_exact.clamp(min=1))
            / math.log(self.max_distance / max_exact.clamp(min=1))
            * (half_buckets - max_exact)
        ).long()

        # Clamp values to be within valid bucket indices [0, num_buckets-1]
        # Max index is num_buckets - 1
        val_if_large = torch.min(val_if_large, torch.full_like(n, half_buckets - 1))

        # Combine results and ensure final index is within [0, num_buckets - 1]
        ret += torch.where(is_small, n, val_if_large)
        ret = torch.min(ret, torch.tensor(num_buckets - 1, device=ret.device)) # Final clamp
        return ret

    def compute_bias(self, query_length: int, key_length: int, device=None) -> torch.Tensor:
        """ Computes the bias tensor based on query and key lengths. """
        # [T_query, T_key] tensor of relative positions
        relative_position = torch.arange(key_length, device=device)[None, :] - torch.arange(query_length, device=device)[:, None]

        # Calculate bucket indices for each relative position
        rp_bucket = self._relative_position_bucket(relative_position)

        # Look up bias values from the learnable table
        # [T_query, T_key, num_heads]
        values = self.relative_attention_bias(rp_bucket)

        # Reshape to [1, num_heads, T_query, T_key] for adding to attention scores
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        Adds the computed relative position bias to the attention scores.

        Args:
            attn_scores (torch.Tensor): Attention scores, shape [B, H, T_query, T_key].

        Returns:
            torch.Tensor: Attention scores with added bias.
        """
        # Get query and key lengths from attention scores shape
        batch_size, num_heads, query_length, key_length = attn_scores.shape

        # Compute the bias tensor [1, H, T_query, T_key]
        bias = self.compute_bias(query_length, key_length, device=attn_scores.device)

        # Add bias to attention scores (broadcasting handles batch dim)
        return attn_scores + bias


@register_module("attention", "time")
class TimeAttention(BaseMultiHeadAttention):
    """
    Example time-aware attention using RoPE and relative position bias.
    Uses RotaryProjection to generate cos/sin and BinaryAttentionBias for bias.
    Applies RoPE using the helper function from embedding module.
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
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            bias=bias,
            **kwargs # Pass any extra args
        )

        # Initialize RoPE projection (generates cos/sin)
        self.rotary_proj = RotaryProjection(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base
        )

        # Initialize Relative Position Bias
        self.rel_pos_bias = BinaryAttentionBias(
            num_heads=self.num_heads,
            num_buckets=num_rel_pos_buckets,
            max_distance=max_rel_pos_distance,
            is_decoder=is_decoder # Use causal bucketing if it's a decoder layer
        )

    # Override the forward method to incorporate RoPE and Relative Bias
    def forward(
        self,
        hidden_states: torch.Tensor, # query tensor, shape [B, T, D]
        key_value_states: Optional[torch.Tensor] = None, # optional key/value tensor for cross-attn, shape [B, S, D]
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # for caching in decoding, shapes [(B, H, S_prev, D_head), (B, H, S_prev, D_head)]
        attention_mask: Optional[torch.Tensor] = None, # shape [B, 1, T, S] (additive mask: 0 for keep, -inf for mask)
        head_mask: Optional[torch.Tensor] = None, # shape [H,] or [B, H] (multiplicative mask) - Not typically used here
        output_attentions: bool = False,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None, # Position IDs for RoPE
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        bsz, tgt_len, _ = hidden_states.size()

        # Determine key/value source
        is_cross_attn = key_value_states is not None
        kv_source = key_value_states if is_cross_attn else hidden_states
        src_len = kv_source.size(1) # Key/Value sequence length before caching

        # === Q, K, V projection ===
        q = self.q_proj(hidden_states)
        k = self.k_proj(kv_source)
        v = self.v_proj(kv_source)

        # === Reshape Q, K, V to [B, H, T, D_head] ===
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # === Handle past key/values (for decoding) ===
        present_key_value = None
        if use_cache:
            if past_key_value is not None:
                 past_k, past_v = past_key_value
                 k = torch.cat([past_k, k], dim=2)
                 v = torch.cat([past_v, v], dim=2)
            present_key_value = (k, v)

        # Total key/value sequence length after caching
        total_kv_seq_len = k.size(2)

        # === Apply Rotary Positional Embedding (RoPE) ===
        # Get cos/sin cache from RotaryProjection module
        # Pass dummy tensor v for device/dtype, use total_kv_seq_len
        cos, sin = self.rotary_proj(v, seq_len=total_kv_seq_len)
        # Apply RoPE using the imported helper function
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

        # === Compute attention scores [B, H, T, S_total] ===
        attn_scores = torch.matmul(q * self.scaling, k.transpose(-1, -2))

        # === Apply Relative Position Bias ===
        # Bias depends on query length (tgt_len) and total key length (total_kv_seq_len)
        # Call the forward method of the bias module
        attn_scores = self.rel_pos_bias(attn_scores)

        # === Apply Attention Mask ===
        if attention_mask is not None:
            # Adjust mask shape checks or expansion as needed for caching
            # Assuming mask is already additive and correctly shaped for total_kv_seq_len
            attn_scores = attn_scores + attention_mask

        # === Compute attention probabilities ===
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Optional head mask application
        if head_mask is not None:
             if head_mask.dim() == 1: head_mask = head_mask[None, :, None, None]
             elif head_mask.dim() == 2: head_mask = head_mask[:, :, None, None]
             attn_probs = attn_probs * head_mask

        # === Compute final output ===
        attn_output = torch.matmul(attn_probs, v) # Use original (non-rotated) V

        # Reshape back to [B, T, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_probs = None

        # Return tuple: (final_output, attention_probs, present_key_value_state)
        return attn_output, attn_probs, present_key_value
