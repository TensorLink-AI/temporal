# Modified temporal/modules/attentions/base_attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module

try:
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn_func = None
    print("Warning: flash_attn is not installed. Flash attention will not be available.")

def expand_mask(attention_mask: torch.Tensor, tgt_len: int, dtype: torch.dtype) -> torch.Tensor:
    # ... (no changes) ...
    if attention_mask.dim() != 2:
        raise ValueError(f"Expected 2D mask of shape [B, S], got {attention_mask.shape}.")

    bsz, src_len = attention_mask.shape
    # Convert mask: 0 -> 0.0, 1 -> -inf 
    inverted_mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min
    expanded_mask = inverted_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)

    return expanded_mask # Return the expanded mask directly


class BaseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False, # Flag for causal masking
        is_cross_attention: bool = False, # Flag for cross attention setup
        bias: bool = True,
        # Added `**kwargs` to capture any unexpected args from builder/config
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention # Store this flag

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Modify k_proj and v_proj based on cross_attention flag if needed
        # For standard MHA, the input dim is embed_dim for both self and cross attention context
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _reshape_for_heads(self, x: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        # Reshape to (B, T, H, D_head) -> (B, H, T, D_head)
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Note: Original code reshaped to (B*H, T, D_head), keeping standard (B, H, T, D_head) is often cleaner

    def compute_attention_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # q, k: [B, H, T, D_head]
        # k.transpose: [B, H, D_head, S] -> output: [B, H, T, S]
        return torch.matmul(q, k.transpose(-1, -2))

    def forward(
        self,
        hidden_states: torch.Tensor, # query tensor, shape [B, T, D]
        key_value_states: Optional[torch.Tensor] = None, # optional key/value tensor for cross-attn, shape [B, S, D]
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # for caching in decoding, shapes [(B, H, S_prev, D_head), (B, H, S_prev, D_head)]
        attention_mask: Optional[torch.Tensor] = None, # shape [B, 1, T, S] (additive mask: 0 for keep, -inf for mask)
        head_mask: Optional[torch.Tensor] = None, # shape [H,] or [B, H] (multiplicative mask) - Not typically used here
        output_attentions: bool = False,
        # Added use_cache flag, standard in HF decoders
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        bsz, tgt_len, _ = hidden_states.size()

        # Determine key/value source
        is_cross_attn = key_value_states is not None
        kv_source = key_value_states if is_cross_attn else hidden_states
        src_len = kv_source.size(1)

        # === Q, K, V projection ===
        q = self.q_proj(hidden_states) * self.scaling # Apply scaling to Q
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
                # Prepend past k/v to current k/v: [B, H, S_prev + S_curr, D_head]
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            # Save k/v state for next time step
            present_key_value = (k, v)

        # Update src_len after potentially prepending past keys/values
        src_len = k.size(2)

        # === Compute attention scores [B, H, T, S] ===
        attn_scores = self.compute_attention_scores(q, k)

        # Check attention mask dimensions
        if attention_mask is not None:
            expected_mask_shape = (bsz, 1, tgt_len, src_len)
            if attention_mask.shape != expected_mask_shape:
                 # Try to expand if needed (e.g., if input mask was [B, S])
                 if attention_mask.dim() == 2 and attention_mask.shape == (bsz, src_len):
                      # Common case: padding mask for encoder or cross-attention
                      attention_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
                 elif attention_mask.dim() == 3 and attention_mask.shape == (bsz, tgt_len, src_len):
                      # Might be a combined mask, add head dim
                      attention_mask = attention_mask[:, None, :, :]
                 else:
                      raise ValueError(f"Attention mask shape mismatch. Expected {expected_mask_shape}, got {attention_mask.shape}")
                 
                 # Apply additive mask
                 attn_scores = attn_scores + attention_mask # Assuming mask has -inf for masked positions

        # === Compute attention probabilities ===
        attn_probs = F.softmax(attn_scores, dim=-1) # [B, H, T, S]
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Optional head mask application (multiplicative)
        if head_mask is not None:
             # Ensure head_mask has shape [B, H, 1, 1] or [1, H, 1, 1] for broadcasting
             if head_mask.dim() == 1: # Shape [H]
                 head_mask = head_mask[None, :, None, None]
             elif head_mask.dim() == 2: # Shape [B, H]
                 head_mask = head_mask[:, :, None, None]
             # Add other checks if necessary based on expected head_mask shapes
             attn_probs = attn_probs * head_mask


        # === Compute final output ===
        # attn_probs: [B, H, T, S], v: [B, H, S, D_head] -> output: [B, H, T, D_head]
        attn_output = torch.matmul(attn_probs, v)

        # Reshape back to [B, T, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # Return results
        if not output_attentions:
            attn_probs = None # Don't return attention probabilities if not requested

        # Return tuple: (final_output, attention_probs, present_key_value_state)
        return attn_output, attn_probs, present_key_value


# ------------------------------------------------------------------
# ✅ Implementation 1: Standard Dot Product Attention
# ------------------------------------------------------------------

@register_module("attention", "full")
class FullAttention(BaseMultiHeadAttention):
    # *** MODIFICATION: Explicitly accept arguments ***
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        bias: bool = True,
        **kwargs # Accept extra kwargs
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            bias=bias,
            **kwargs # Pass extra kwargs to base
        )
    # Forward method inherited from BaseMultiHeadAttention is sufficient


@register_module("attention", "flash")
class FlashAttention(BaseMultiHeadAttention):
    # *** MODIFICATION: Explicitly accept arguments (already correct here) ***
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        bias: bool = True,
        # Added **kwargs here too for consistency
        **kwargs
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            is_cross_attention=is_cross_attention,
            bias=bias,
            **kwargs # Pass extra kwargs to base
        )
        if flash_attn_func is None:
             # Raise error during init if flash isn't available but requested
             raise ImportError("Flash Attention was requested, but flash_attn library is not installed.")


    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None, # Flash Attention typically doesn't support past_key_value caching directly
        attention_mask=None, # Flash Attention often handles masking differently (e.g., seqlen_q/k) or not at all
        head_mask=None, # Not typically supported by flash_attn_func
        output_attentions=False,
        use_cache=False, # Explicitly add use_cache
    ):
        if flash_attn_func is None:
            # This check is redundant if __init__ already raised, but belts and suspenders
            raise ImportError("flash_attn_func is not available.")
        if past_key_value is not None:
             # Flash attention v1/v2 might not support caching this way.
             # Depending on the version, you might need custom logic or fall back to standard attn.
             raise NotImplementedError("FlashAttention in this implementation does not support past_key_value caching.")
        if attention_mask is not None:
             # Standard flash_attn_func doesn't take a general mask like this.
             # Causal masking is handled by the `causal` flag. Padding needs seqlen_k.
             # For simplicity, we'll ignore arbitrary masks here. A production system might need more.
             print("Warning: FlashAttention is ignoring the provided attention_mask.")
        if head_mask is not None:
             print("Warning: FlashAttention is ignoring the provided head_mask.")
        if output_attentions:
             print("Warning: FlashAttention does not return attention probabilities.")


        B, T, _ = hidden_states.shape
        is_cross_attn = key_value_states is not None
        kv_source = key_value_states if is_cross_attn else hidden_states
        S = kv_source.size(1)

        q = self.q_proj(hidden_states)
        k = self.k_proj(kv_source)
        v = self.v_proj(kv_source)

        # Reshape for flash_attn: [B, T, H, D_head]
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)

        # Determine causal flag based on whether it's decoder self-attention
        # Assumes self.is_decoder is correctly set during __init__
        # And assumes it's NOT cross-attention (causal mask doesn't apply to cross-attn)
        is_causal = self.is_decoder and not is_cross_attn

        # flash_attn_func expects [B, T, H, D_head]
        # Note: Check flash_attn documentation for exact API version details
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scaling, # Use pre-computed scaling
            causal=is_causal
            # seqlen_q=T, # May need seqlen info if using padding masks
            # seqlen_k=S,
        )

        # Reshape back to [B, T, D]
        attn_output = attn_output.view(B, T, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # Flash attention doesn't return attn_probs or support standard caching
        present_key_value = None

        return attn_output, None, present_key_value
