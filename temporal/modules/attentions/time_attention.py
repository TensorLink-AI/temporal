# temporal/modules/attentions/time_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention, expand_mask

# --- Helper functions for RoPE ---
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies Rotary Positional Embedding to query and key tensors."""
    # cos, sin: [seq_len, dim] or [bsz, 1, seq_len, dim]
    # q, k: [bsz, num_heads, seq_len, head_dim]
    
    # Ensure cos/sin are broadcastable over heads dimension if needed
    # Simple case assumes cos/sin are [seq_len, head_dim] or similar that can be indexed by position_ids
    # and then applied element-wise after reshaping/repeating.

    if cos.dim() == 2: # [seq_len, dim] -> need to gather based on position_ids if provided
        if position_ids is None:
            # Assuming standard range if position_ids not given
             cos = cos[None, None, :, :] # -> [1, 1, seq_len, dim]
             sin = sin[None, None, :, :] # -> [1, 1, seq_len, dim]
        else:
            # Gather based on position IDs: [bsz, seq_len] -> [bsz, seq_len, dim]
             cos = cos[position_ids].unsqueeze(1) # -> [bsz, 1, seq_len, dim]
             sin = sin[position_ids].unsqueeze(1) # -> [bsz, 1, seq_len, dim]
    # else: assume cos/sin already have correct shape e.g. [bsz, 1, seq_len, dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# --- End RoPE Helpers ---


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

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q (torch.Tensor): Query tensor, shape [B, H, T, D_head].
            k (torch.Tensor): Key tensor, shape [B, H, T, D_head].
            seq_len (int, optional): Sequence length. If None, inferred from q.shape[2].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
        """
        if seq_len is None:
            seq_len = q.shape[2] # T (target sequence length)

        # Ensure cache is large enough and on correct device/dtype
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != q.device or self.cos_cached.dtype != q.dtype:
             self._set_cos_sin_cache(seq_len=seq_len, device=q.device, dtype=q.dtype)

        # Get precomputed cos/sin values for the sequence length
        # Slice the cache: Shape [seq_len, dim]
        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]
        
        # Apply rotation
        # Position IDs are implicitly 0 to seq_len-1 here
        q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)

        return q_rotated, k_rotated


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
        n = -relative_position
        
        # Handle causal vs non-causal
        if not self.is_decoder: # Bidirectional attention
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets # Offset for negative positions
            n = torch.abs(n)
        else: # Causal attention (decoder)
             n = torch.max(n, torch.zeros_like(n)) # Consider only past positions (n>=0)

        # Half of the buckets are for exact distances near 0
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically spaced larger distances
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        
        # Clamp values to be within valid bucket indices [0, num_buckets-1]
        val_if_large = torch.min(val_if_large, torch.full_like(n, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
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

        # Initialize RoPE projection
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        bsz, tgt_len, _ = hidden_states.size()

        # Determine key/value source
        is_cross_attn = key_value_states is not None
        kv_source = key_value_states if is_cross_attn else hidden_states
        src_len = kv_source.size(1) # Key/Value sequence length before caching

        # === Q, K, V projection ===
        q = self.q_proj(hidden_states) # Apply scaling later or handle in RoPE if needed
        k = self.k_proj(kv_source)
        v = self.v_proj(kv_source)

        # === Reshape Q, K, V to [B, H, T, D_head] ===
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # === Handle past key/values (for decoding) ===
        # Important: Caching happens BEFORE RoPE application if RoPE depends on absolute positions
        present_key_value = None
        current_kv_seq_len = src_len # Length of K/V for the current step
        if past_key_value is not None:
             # Prepend past k/v: [B, H, S_prev + S_curr, D_head]
             past_k, past_v = past_key_value
             k = torch.cat([past_k, k], dim=2)
             v = torch.cat([past_v, v], dim=2)
        
        # Total key/value sequence length after caching
        total_kv_seq_len = k.size(2) 
        
        if use_cache:
            present_key_value = (k, v)

        # === Apply Rotary Positional Embedding (RoPE) ===
        # seq_len for RoPE should be the length of Q and K *after* potential caching
        # Assuming RoPE uses absolute positions up to total_kv_seq_len
        # Note: RoPE might need adjustment if only applied to current query/key tokens in cached scenario
        q_rotated, k_rotated = self.rotary_proj(q, k, seq_len=total_kv_seq_len) 

        # === Compute attention scores [B, H, T, S_total] ===
        # Use rotated Q and K, apply scaling here
        attn_scores = torch.matmul(q_rotated * self.scaling, k_rotated.transpose(-1, -2))

        # === Apply Relative Position Bias ===
        # Bias depends on query length (tgt_len) and total key length (total_kv_seq_len)
        rel_pos_bias = self.rel_pos_bias(attn_scores) # Bias calculated based on score shape
        attn_scores = attn_scores + rel_pos_bias

        # === Apply Attention Mask ===
        if attention_mask is not None:
            # Mask shape should align with attn_scores [B, H, T, S_total]
            # Base class mask expansion logic might need verification for cached scenarios
            # Assuming additive mask (-inf for masked)
            # Ensure mask matches the final key length if caching is used
            if attention_mask.shape[-1] != total_kv_seq_len:
                  # This might happen if the input mask only covers the initial sequence. Needs careful handling.
                  # Example: If mask is [B, 1, T, T] (causal) it needs to allow attention to past keys.
                  # For simplicity here, assume the mask provided already accounts for past_key_values if needed.
                  # A robust implementation might need to construct the mask dynamically.
                  # print(f"Warning: Attention mask shape {attention_mask.shape} doesn't match total KV length {total_kv_seq_len}. Masking might be incorrect with caching.")
                  pass # Allow potentially mismatched mask for now, user must provide correct one

            attn_scores = attn_scores + attention_mask 

        # === Compute attention probabilities ===
        attn_probs = F.softmax(attn_scores, dim=-1) 
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Optional head mask application
        if head_mask is not None:
             # ... (head mask logic as in base class) ...
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
