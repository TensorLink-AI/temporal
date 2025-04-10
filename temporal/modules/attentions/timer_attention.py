import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention

# Re-use your rotary helpers
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    split = x.shape[-1] // 2
    x1 = x[..., :split]
    x2 = x[..., split:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    position_ids: torch.LongTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to Q/K.
    position_ids: (L,) or (B,L)
    cos, sin: shape depends on your caching strategy, e.g. (max_seq_len, head_dim)
    """
    # Expand cos/sin if needed so they broadcast to (B, H, L, head_dim).
    cos = cos[position_ids]
    sin = sin[position_ids]
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed

class TimeMoeRotaryEmbedding(nn.Module):
    """
    Caches cos/sin for rotary embeddings up to max_position_embeddings.
    """
    def __init__(self, head_dim, max_position_embeddings=1024, base=10000, device=None):
        super().__init__()
        self.dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (
            torch.arange(0, self.dim, 2, device=device, dtype=torch.float) / self.dim
        ))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, 
            device=device, 
            dtype=torch.float
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        # freq shape => (dim/2,)
        # positions outer-product => (seq_len, dim/2)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # double to match head_dim
        cos_emb = emb.cos().to(dtype)
        sin_emb = emb.sin().to(dtype)

        self.register_buffer("cos_cached", cos_emb, persistent=False)
        self.register_buffer("sin_cached", sin_emb, persistent=False)

    def forward(self, seq_len: int, device=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device or self.cos_cached.device, self.cos_cached.dtype)
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]
        return cos, sin

# -------------------------------------------------------------------
# Inherit from your provided BaseMultiHeadAttention
# -------------------------------------------------------------------
from typing import Optional, Tuple
# from .base_multihead_attn import BaseMultiHeadAttention  # Adjust import as needed

class TimerAttention(BaseMultiHeadAttention):
    """
    A multi-head attention module that injects rotary embeddings into Q/K,
    returning (attn_output, attn_probs, present_key_value).
    Compatible with your 'BaseMultiHeadAttention' code and caching logic.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        """
        config: Must have:
          - hidden_size
          - num_attention_heads
          - attention_dropout
          - is_decoder (bool), if you use caching
          - max_position_embeddings
        layer_idx: used for referencing which layer's 'past_key_value'.
        """
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=getattr(config, "is_decoder", False),
            bias=True,
            is_cross_attention=False,  # or set True if you want cross-attn
        )
        self.layer_idx = layer_idx
        self.head_dim = self.embed_dim // self.num_heads
        self.rotary_emb = TimeMoeRotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Inherits arguments from BaseMultiHeadAttention forward, plus 'position_ids' for rotary.
        Returns:
          attn_output: (B, tgt_len, embed_dim)
          attn_probs:   (B, num_heads, tgt_len, src_len) if output_attentions=True else None
          present_key_value: for caching if self.is_decoder is True
        """

        # 1) Parse shapes & cross-attention
        bsz, tgt_len, _ = hidden_states.size()
        is_cross_attention = key_value_states is not None

        # 2) Project Q
        # scale => self.scaling = self.head_dim ** -0.5
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self._shape(query_states, tgt_len, bsz)  # => [B, H, tgt_len, head_dim]

        # 3) K/V from cross or self
        if is_cross_attention:
            # [B, src_len, embed_dim] => shape => [B, H, src_len, head_dim]
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # 4) Past Key/Value (caching)
        if past_key_value is not None:
            # Concat old + new along seq_len axis=2
            # past_key_value[0,1] => old K,V
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Prepare to return updated K/V if we're a decoder
        present_key_value = (key_states, value_states) if self.is_decoder else None

        # 5) If not cross-attention, apply rotary embeddings to Q/K
        #    Usually we do rotary only on the "self-attention" portion. 
        if not is_cross_attention:
            kv_seq_len = key_states.size(2)  # shape => (B,H,K_len,head_dim)
            # For partial decoding, kv_seq_len might be larger than tgt_len.
            # We'll handle position_ids or default to 0..kv_seq_len-1
            if position_ids is None:
                # shape => (tgt_len,) but we have partial updates
                # if we are decoding, we might have offset
                position_ids = torch.arange(
                    key_states.size(2), device=key_states.device, dtype=torch.long
                )
            cos, sin = self.rotary_emb(seq_len=kv_seq_len, device=key_states.device)

            # apply rotaries
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # 6) Flatten for matmul => shape => [B*H, T, D]
        #    The parent's code calls self.compute_attention_scores(query, key)
        #    then does masking, softmax, dropout, etc. We'll replicate that logic:
        bsz_heads = bsz * self.num_heads
        query_states = query_states.view(bsz_heads, -1, self.head_dim)       # (B*H, tgt_len, head_dim)
        key_states   = key_states.view(bsz_heads, -1, self.head_dim)         # (B*H, src_len, head_dim)
        value_states = value_states.view(bsz_heads, -1, self.head_dim)

        # 7) Attention scores: (B*H, tgt_len, src_len)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # 8) Possibly add attention_mask
        if attention_mask is not None:
            # shape => [B,1,tgt_len,src_len] => we add to attn_weights
            # expand to [B,H,tgt_len,src_len], then flatten
            bsz_, _, tgt_, src_ = attention_mask.shape
            if (bsz_ != bsz) or (tgt_ != tgt_len):
                raise ValueError(f"Mismatch in attention_mask shape: {attention_mask.shape}")

            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, -1)
            attn_weights = attn_weights + attention_mask  # broadcast
            attn_weights = attn_weights.view(bsz_heads, tgt_len, -1)

        # 9) Softmax
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # 10) Weighted sum => [B*H, tgt_len, head_dim]
        attn_output = torch.bmm(attn_probs, value_states)

        # 11) Reshape back => (B, tgt_len, embed_dim)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # 12) If output_attentions=True, reshape attn_probs => (B,H,tgt_len,src_len)
        if output_attentions:
            attn_probs = attn_probs.view(bsz, self.num_heads, tgt_len, -1)
        else:
            attn_probs = None

        return attn_output, attn_probs, present_key_value
