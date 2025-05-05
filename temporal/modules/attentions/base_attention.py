# temporal/modules/attentions/base_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from temporal.registry.core import register_module
# --- CORRECTED IMPORTS for RoPE/ALiBi Components ---
from temporal.modules.attentions.time_attention import (
    RotaryProjection, # Module for generating cos/sin cache
)
from temporal.modules.embedders.embedding import (
    ALiBiPositionalBias,  # Module for generating ALiBi bias
    apply_rotary_pos_emb, # Helper function to apply RoPE using cos/sin
)
# --- End CORRECTED IMPORTS ---

# --- Attempt to import flash attention ---
try:
    from flash_attn import flash_attn_func
    _flash_attn_available = True
except ImportError:
    flash_attn_func = None
    _flash_attn_available = False
# --- End ---


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
        self.dropout = dropout # Dropout is usually applied *after* softmax in subclasses
        self.head_dim = embed_dim // num_heads
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.scaling = self.head_dim ** -0.5 # Used for scaled dot-product attention

        # Standard QKV and output projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def compute_attention_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Computes the raw attention scores. Subclasses might override this.
        Default is scaled dot-product.
        Args:
            q: Query tensor [B, H, T, D_head]
            k: Key tensor [B, H, S, D_head]
        Returns:
            Attention scores [B, H, T, S]
        """
        # Standard scaled dot-product
        attn_scores = torch.matmul(q * self.scaling, k.transpose(-1, -2))
        return attn_scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None, # Optional: For masking specific heads
        output_attentions: bool = False,
        use_cache: bool = False,
        # --- Add RoPE/ALiBi specific args needed by FullAttention's forward ---
        position_ids: Optional[torch.LongTensor] = None, # Needed if RoPE is used with caching/offsets
        rotary_proj: Optional[nn.Module] = None, # Pass initialized module
        alibi_bias_generator: Optional[nn.Module] = None # Pass initialized module
        # --- End ---
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Multi-Head Attention. Includes QKV projection, optional RoPE/ALiBi,
        attention computation, and output projection.
        """
        bsz, tgt_len, _ = hidden_states.size()

        # Determine key/value source and sequence length
        is_cross_attn = key_value_states is not None
        kv_source = key_value_states if is_cross_attn else hidden_states
        src_len = kv_source.size(1)

        # === Project Q, K, V ===
        q = self.q_proj(hidden_states)
        k = self.k_proj(kv_source)
        v = self.v_proj(kv_source)

        # === Reshape Q, K, V to [B, H, T/S, D_head] ===
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # === Handle past key/values (for decoding/caching) ===
        present_key_value = None
        if use_cache:
            # Important: If using RoPE with caching, position_ids must account for the offset!
            if past_key_value is not None:
                # Concatenate past_key_value to current k, v
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            # Store the potentially updated k, v for the next step
            present_key_value = (k, v)

        # Update source length after potentially incorporating past keys/values
        src_len = k.size(2)

        # --- 1) Apply Rotary Positional Embeddings (RoPE) if enabled ---
        if rotary_proj is not None:
            if is_cross_attn:
                 # Only apply RoPE to query in cross-attention, not keys from encoder
                 # (Standard practice) Get cos/sin based on query length T
                 cos, sin = rotary_proj(q, seq_len=tgt_len)
                 q, _ = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids) # Only rotate q
                 # k remains unrotated as it comes from a different context (encoder)
            else:
                 # Self-attention: Apply RoPE to both Q and K
                 kv_seq_len = k.shape[-2] # Use the actual (potentially cached) length of K
                 cos, sin = rotary_proj(v, seq_len=kv_seq_len) # Use dummy v for device/dtype
                 # Apply RoPE using the helper function from embedding module
                 q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)


        # === Compute attention scores [B, H, T, S] ===
        # Subclasses might override compute_attention_scores, or use this default
        attn_scores = self.compute_attention_scores(q, k) # Uses scaled dot-product by default

        # --- 2) Add ALiBi relative-bias if enabled ---
        if alibi_bias_generator is not None:
            # Calculate the bias dynamically based on current query/key lengths
            # Assume alibi_bias_generator.forward(batch_size, seq_len) -> [B, H, T, S] or broadcastable
            alibi_bias = alibi_bias_generator(batch_size=bsz, seq_len=src_len)
            # Adjust slicing based on actual ALiBi output shape and needs
            # Example: If alibi_bias is [1, H, S, S] representing causal mask bias:
            if alibi_bias.shape[-2] == src_len and alibi_bias.shape[-1] == src_len:
                 # Select the relevant part for the current query length T
                 # This assumes causal mask structure in ALiBi output
                 alibi_bias = alibi_bias[:, :, -tgt_len:, :src_len]
            elif alibi_bias.shape[-2] == tgt_len and alibi_bias.shape[-1] == src_len:
                 # Shape already matches [*, *, T, S]
                 pass
            else:
                 # Attempt broadcast or raise error
                 try:
                     # Check if broadcastable
                     _ = attn_scores + alibi_bias
                 except RuntimeError as e:
                      raise ValueError(f"ALiBi bias shape {alibi_bias.shape} is not compatible or broadcastable with attention scores shape {attn_scores.shape}. Error: {e}")

            attn_scores = attn_scores + alibi_bias

        # === Apply Attention Mask ===
        # Standard masking logic (additive mask)
        if attention_mask is not None:
            # Allow for different mask types, ensure final shape is broadcastable to attn_scores
            if attention_mask.dim() == 2: # [B, S] -> [B, 1, 1, S]
                 attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3: # [B, T, S] -> [B, 1, T, S]
                 attention_mask = attention_mask[:, None, :, :]
            # Check broadcast compatibility
            if attention_mask.shape[-2] != 1 and attention_mask.shape[-2] != tgt_len:
                 raise ValueError(f"Mask T dim {attention_mask.shape[-2]} not compatible with tgt_len {tgt_len}")
            if attention_mask.shape[-1] != 1 and attention_mask.shape[-1] != src_len:
                 raise ValueError(f"Mask S dim {attention_mask.shape[-1]} not compatible with src_len {src_len}")

            # Additive mask assumes 0 for keep, large negative for mask
            attn_scores = attn_scores + attention_mask


        # === Compute attention probabilities ===
        attn_probs = F.softmax(attn_scores, dim=-1)

        # === Apply Dropout ===
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # === (Optional) Apply head mask ===
        if head_mask is not None:
             # Ensure head_mask is broadcastable to [B, H, T, S]
             if head_mask.dim() == 1: # [H] -> [1, H, 1, 1]
                 head_mask = head_mask[None, :, None, None]
             elif head_mask.dim() == 2: # [B, H] -> [B, H, 1, 1]
                 head_mask = head_mask[:, :, None, None]
             attn_probs = attn_probs * head_mask

        # === Compute final output ===
        attn_output = torch.matmul(attn_probs, v) # [B, H, T, D_head]

        # === Reshape and Project Output ===
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # === Return results ===
        if not output_attentions:
             attn_probs = None # Don't return probs if not requested

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
        use_rope: bool = False,        # Flag to enable RoPE
        use_alibi: bool = False,       # Flag to enable ALiBi
        max_position_embeddings: int = 4096, # Max length for RoPE/ALiBi modules
        rope_base: int = 10000,        # Base for RoPE frequencies
        # --- End RoPE/ALiBi ---
        **kwargs # Capture any other arguments passed from config/builder
    ):
        # Initialize the base class first
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout, # Dropout is applied in base forward after softmax
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            bias=bias,
            **kwargs # Pass along any other arguments
        )

        self.use_rope = use_rope
        self.use_alibi = use_alibi

        # --- Conditionally Initialize RoPE ---
        self.rotary_proj = None
        if self.use_rope:
            # Use RotaryProjection from time_attention
            self.rotary_proj = RotaryProjection(
                dim=self.head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_base,
            )

        # --- Conditionally Initialize ALiBi ---
        self.alibi_bias_generator = None
        if self.use_alibi:
            # Use ALiBiPositionalBias from embedding
            self.alibi_bias_generator = ALiBiPositionalBias(
                num_heads=self.num_heads,
                max_seq_len=max_position_embeddings
            )

    # Override the forward method to pass RoPE/ALiBi modules if they exist
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        # Need position_ids for RoPE, especially when using caching
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # Call the base class's forward method, passing the potentially initialized
        # RoPE and ALiBi modules. The base forward now contains the logic to apply them.
        return super().forward(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_ids=position_ids, # Pass along position_ids
            rotary_proj=self.rotary_proj, # Pass the RoPE module instance
            alibi_bias_generator=self.alibi_bias_generator # Pass the ALiBi module instance
        )

    # We don't need to override compute_attention_scores unless FullAttention
    # uses a different scoring mechanism than the base (scaled dot-product).



@register_module("attention", "flash")
class FlashAttention(BaseMultiHeadAttention):
    """
    Attention implementation using flash_attn library (if available).
    Note: FlashAttention often has different calling conventions and limitations
          (e.g., mask support, KV caching, head dim requirements) than standard attention.
          This implementation makes simplifying assumptions.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False, # FlashAttention typically doesn't do cross-attn
        bias: bool = True, # Projection bias
        softmax_scale: Optional[float] = None, # Specific to flash_attn_func
        causal: bool = False, # Use causal masking built into flash_attn_func
        **kwargs
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            bias=bias,
            **kwargs
        )
        self.softmax_scale = softmax_scale
        self.causal = causal or (is_decoder and not is_cross_attention) # Infer causal from is_decoder

        if not _flash_attn_available:
            raise ImportError("FlashAttention requires the flash_attn library to be installed.")
        if is_cross_attention:
            print("Warning: FlashAttention typically does not support cross-attention; using self-attention pattern.")


    # Override forward to use flash_attn_func
    def forward(
        self,
        hidden_states: torch.Tensor,               # [B, T, D]
        key_value_states: Optional[torch.Tensor] = None, # Ignored if cross-attn not truly supported
        past_key_value=None, # Flash Attention typically doesn't support past_key_value caching directly
        attention_mask=None, # Flash Attention often handles masking differently (e.g., seqlen_q/k) or not at all
        head_mask=None, # Not typically supported by flash_attn_func
        output_attentions=False,
        use_cache=False, # Explicitly add use_cache
        # --- RoPE/ALiBi arguments are NOT used by this FlashAttention implementation ---
        position_ids: Optional[torch.LongTensor] = None,
        rotary_proj: Optional[nn.Module] = None,
        alibi_bias_generator: Optional[nn.Module] = None
        # --- End ---
    ):
        if flash_attn_func is None:
            # This check is redundant if __init__ already raised, but belts and suspenders
            raise ImportError("flash_attn_func is not available.")
        if past_key_value is not None or use_cache:
             # Flash attention v1/v2 might not support caching this way.
             # Depending on the version, you might need custom logic or fall back to standard attn.
             # FlashAttention-2 does support MQA/GQA KV caching via specific interfaces not used here.
             print("Warning: This FlashAttention implementation does not support KV caching; ignoring use_cache.")
        if attention_mask is not None:
             # Standard flash_attn_func doesn't take a general mask like this.
             # Causal masking is handled by the `causal` flag. Padding needs seqlen_k.
             # For simplicity, we'll ignore arbitrary masks here. A production system might need more.
             print("Warning: FlashAttention is ignoring the provided attention_mask; relying on 'causal' flag.")
        if head_mask is not None:
             print("Warning: FlashAttention is ignoring the provided head_mask.")
        if output_attentions:
             print("Warning: FlashAttention does not return attention probabilities.")


        B, T, _ = hidden_states.shape
        kv_source = key_value_states if self.is_cross_attention and key_value_states is not None else hidden_states

        # === Project Q, K, V ===
        q = self.q_proj(hidden_states)
        k = self.k_proj(kv_source)
        v = self.v_proj(kv_source)

        # === Reshape for flash_attn_func ===
        # Expected input: [B, SeqLen, NumHeads, HeadDim]
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, -1, self.num_heads, self.head_dim) # Use -1 for seqlen flexibility (S)
        v = v.view(B, -1, self.num_heads, self.head_dim) # Use -1 for seqlen flexibility (S)


        # === Call flash_attn_func ===
        # Note: seqlen_k arguments would be needed for padding. Not handled here.
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.softmax_scale, # Uses 1/sqrt(d_head) if None
            causal=self.causal # Use the causal flag set in __init__
        )

        # === Reshape and Project Output ===
        attn_output = attn_output.view(B, T, self.embed_dim) # [B, T, D]
        attn_output = self.out_proj(attn_output)

        # === Return results (no attn_probs, no past_key_value) ===
        return attn_output, None, None
