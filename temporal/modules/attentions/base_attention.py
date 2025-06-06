# temporal/modules/attentions/base_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from temporal.registry.core import register_module
from temporal.modules.embedders.embedding import (
    RotaryPositionalEmbedding, # Use the RoPE module from embedding.py
    ALiBiPositionalBias,       # Module for generating ALiBi bias
    apply_rotary_pos_emb,      # Helper function to apply RoPE using cos/sin
)

# --- Attempt to import flash attention ---
try:
    from flash_attn import flash_attn_func
    _flash_attn_available = True
except ImportError:
    flash_attn_func = None
    _flash_attn_available = False



class BaseMultiHeadAttention(nn.Module):
    """
    Base class for Multi-Head Attention mechanisms.
    Provides common structure for QKV projection and output projection.
    Subclasses should implement the core attention logic in `forward` or `compute_attention_scores`.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        bias: bool = True,
        **kwargs # Accept additional unused kwargs for flexibility
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if self.head_dim % 2 != 0:
             # RoPE requires head_dim to be even for splitting
             print(f"Warning: RoPE requires head_dim to be even, but got {self.head_dim}. RoPE may fail if enabled.")

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def compute_attention_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        attn_scores = torch.matmul(q * self.scaling, k.transpose(-1, -2))
        return attn_scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        # --- RoPE/ALiBi specific args needed by FullAttention's forward ---
        position_ids: Optional[torch.LongTensor] = None,
        rotary_proj: Optional[nn.Module] = None, # Pass instance of RotaryPositionalEmbedding
        alibi_bias_generator: Optional[nn.Module] = None # Pass instance of ALiBiPositionalBias
        # --- End ---
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

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
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            present_key_value = (k, v)

        src_len = k.size(2) # Updated key sequence length

        # --- 1) Apply RoPE if rotary_proj is provided ---
        if rotary_proj is not None:
             # Call the forward of the passed RotaryPositionalEmbedding instance
             # It expects a dummy tensor for device/dtype and the required seq_len
             kv_seq_len = k.shape[-2] # Use the actual (potentially cached) length of K
             cos, sin = rotary_proj(v, seq_len=kv_seq_len) # Get cos/sin caches

             if is_cross_attn:
                  # Only rotate query in cross-attention. Pass q for k to match sliced cos/sin dimensions.
                  q, _ = apply_rotary_pos_emb(q, q, cos[:tgt_len,:], sin[:tgt_len,:], position_ids=position_ids)
             else:
                  # Rotate both query and key in self-attention
                  q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

        # === Compute attention scores [B, H, T, S] ===
        attn_scores = self.compute_attention_scores(q, k)

        # --- 2) Add ALiBi relative-bias if enabled ---
        if alibi_bias_generator is not None:
            # Call the forward of the passed ALiBiPositionalBias instance
            # Assuming it generates bias based on seq_len (T x T or S x S)
            alibi_bias = alibi_bias_generator(batch_size=bsz, seq_len=src_len)
            # Slice the bias if necessary (e.g., for T x S attention from S x S bias)
            if alibi_bias.shape[-2] == src_len and alibi_bias.shape[-1] == src_len:
                 alibi_bias = alibi_bias[..., -tgt_len:, :src_len] # Assumes causal slicing
            elif alibi_bias.shape[-2] == tgt_len and alibi_bias.shape[-1] == src_len:
                 pass # Shape already T x S
            else:
                 # Check broadcast compatibility carefully
                 try: _ = attn_scores + alibi_bias
                 except RuntimeError as e: raise ValueError(f"ALiBi bias shape {alibi_bias.shape} not compatible with attention scores shape {attn_scores.shape}. Error: {e}")
            attn_scores = attn_scores + alibi_bias

        # === Apply Attention Mask ===
        if attention_mask is not None:
            # Additive mask assumed
            attn_scores = attn_scores + attention_mask

        # === Compute attention probabilities ===
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # === (Optional) Apply head mask ===
        if head_mask is not None:
             if head_mask.dim() == 1: head_mask = head_mask[None, :, None, None]
             elif head_mask.dim() == 2: head_mask = head_mask[:, :, None, None]
             attn_probs = attn_probs * head_mask

        # === Compute final output ===
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, tgt_len, self.num_heads * self.head_dim)
        assert attn_output.shape[-1] == self.embed_dim, \
            f"Expected embed_dim={self.embed_dim}, got {attn_output.shape[-1]}"
        attn_output = self.out_proj(attn_output)

        if not output_attentions: attn_probs = None
        return attn_output, attn_probs, present_key_value


@register_module("attention", "full")
class FullAttention(BaseMultiHeadAttention):
    """
    Standard multi-head attention implementation using scaled dot-product attention.
    Optionally includes RoPE and ALiBi positional embeddings controlled by flags.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        bias: bool = True,
        # --- RoPE/ALiBi Flags and Parameters ---
        use_rope: bool = False,
        use_alibi: bool = False,
        max_position_embeddings: int = 4096,
        rope_base: int = 10000,
        # --- End RoPE/ALiBi ---
        **kwargs
    ):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, is_cross_attention, bias, **kwargs)
        self.use_rope = use_rope
        self.use_alibi = use_alibi

        # --- Conditionally Initialize RoPE (using imported RotaryPositionalEmbedding) ---
        self.rotary_proj = None
        if self.use_rope:
             # Note: RotaryPositionalEmbedding takes d_model (head_dim), not embed_dim
             if self.head_dim % 2 != 0:
                  raise ValueError("RoPE requires head_dim to be even.")
             self.rotary_proj = RotaryPositionalEmbedding(
                 d_model=self.head_dim,
                 max_seq_len=max_position_embeddings,
                 base=rope_base,
             )

        # --- Conditionally Initialize ALiBi (using imported ALiBiPositionalBias) ---
        self.alibi_bias_generator = None
        if self.use_alibi:
            self.alibi_bias_generator = ALiBiPositionalBias(
                num_heads=self.num_heads,
                max_seq_len=max_position_embeddings # Pass max_seq_len if needed by ALiBi class
            )

    # Override forward to pass modules to base class forward
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
        return super().forward(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_ids=position_ids,
            rotary_proj=self.rotary_proj,
            alibi_bias_generator=self.alibi_bias_generator
        )


@register_module("attention", "flash")
class FlashAttention(BaseMultiHeadAttention):
    # ... (FlashAttention code remains the same) ...
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        bias: bool = True,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        **kwargs
    ):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, is_cross_attention, bias, **kwargs)
        self.softmax_scale = softmax_scale
        self.causal = causal or (is_decoder and not is_cross_attention)
        if not _flash_attn_available: raise ImportError("FlashAttention requires flash_attn.")
        if is_cross_attention: print("Warning: FlashAttention cross-attention NYI.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        use_cache=False,
        position_ids=None, # Accept but ignore
        rotary_proj=None, # Accept but ignore
        alibi_bias_generator=None # Accept but ignore
    ):
        if flash_attn_func is None: raise ImportError("flash_attn_func is not available.")
        if use_cache: print("Warning: FlashAttention KV caching NYI.")
        if rotary_proj is not None or alibi_bias_generator is not None: print("Warning: FlashAttention ignores RoPE/ALiBi.")
        if attention_mask is not None: print("Warning: FlashAttention ignores mask; use causal flag.")
        if head_mask is not None: print("Warning: FlashAttention ignores head_mask.")
        if output_attentions: print("Warning: FlashAttention doesn't return probs.")

        B, T, _ = hidden_states.shape
        kv_source = key_value_states if self.is_cross_attention and key_value_states is not None else hidden_states
        q, k, v = self.q_proj(hidden_states), self.k_proj(kv_source), self.v_proj(kv_source)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, -1, self.num_heads, self.head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim)

        attn_output = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, softmax_scale=self.softmax_scale, causal=self.causal)
        attn_output = attn_output.view(B, T, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, None
