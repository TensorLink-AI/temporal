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
    rotate_half,               # Import rotate_half for manual RoPE application
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
    Base class for multi-head attention mechanisms.

    This class provides the common structure for query, key, and value
    projections, as well as the output projection. Subclasses should
    implement the core attention logic in the `forward` method.

    Attributes:
        embed_dim (int): The embedding dimension of the model.
        num_heads (int): The number of attention heads.
        dropout (float): The dropout rate.
        head_dim (int): The dimension of each attention head.
        is_decoder (bool): Whether this module is used in a decoder.
        is_cross_attention (bool): Whether this module is used for cross-attention.
        scaling (float): The scaling factor for the attention scores.
        q_proj (nn.Linear): The linear projection for the query.
        k_proj (nn.Linear): The linear projection for the key.
        v_proj (nn.Linear): The linear projection for the value.
        out_proj (nn.Linear): The linear projection for the output.
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
        """
        Initializes the BaseMultiHeadAttention module.

        Args:
            embed_dim (int): The embedding dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            is_decoder (bool): Whether this module is used in a decoder.
            is_cross_attention (bool): Whether this module is used for cross-attention.
            bias (bool): Whether to include a bias in the linear projections.
            **kwargs: Additional keyword arguments for flexibility.
        """
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
        """
        Computes the attention scores.

        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.

        Returns:
            torch.Tensor: The attention scores.
        """
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
        """
        Performs the forward pass of the attention layer.

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
            rotary_proj (Optional[nn.Module]): The RoPE projection module.
                Defaults to None.
            alibi_bias_generator (Optional[nn.Module]): The ALiBi bias generator.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                A tuple containing the attention output, the attention probabilities
                (if output_attentions is True), and the updated key and value states
                (if use_cache is True).
        """

        batch_size, query_length, _ = hidden_states.size()
        is_cross_attn = key_value_states is not None
        kv_source = key_value_states if is_cross_attn else hidden_states

        # Determine the source sequence length
        if is_cross_attn and key_value_states is not None:
            key_length = key_value_states.size(1)
        else:
            key_length = hidden_states.size(1)


        q = self.q_proj(hidden_states)
        k = self.k_proj(kv_source)
        v = self.v_proj(kv_source)

        # Reshape Q, K, V
        q = q.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)

        # When using cache, the key/value states from the source are only for the *current* step.
        # So, their length is `query_length`, not `key_length`.
        kv_len = query_length if past_key_value is not None else key_length
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)


        present_key_value = None
        if use_cache:
            if past_key_value is not None:
                # k and v have sequence length 1 here.
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            present_key_value = (k, v)

        # The source length is now the full length of the key tensor
        key_length = k.size(2)

        # --- 1) Apply RoPE if rotary_proj is provided ---
        if rotary_proj is not None:
            kv_seq_len = k.shape[-2]
            cos, sin = rotary_proj(v, seq_len=kv_seq_len)
            
            # --- Correct RoPE application for KV Caching ---
            # The query (q) has a sequence length of 1 during generation.
            # We must slice the cos/sin cache to get the embedding for the CURRENT token's position,
            # which is at the end of the sequence. `cos[-query_length:, :]` correctly handles this.
            # The key (k) has the full sequence length, so it uses the full cos/sin cache.
            
            # Manually apply RoPE to q
            q_cos = cos[-query_length:, :] # Slice for the new token(s)
            q_sin = sin[-query_length:, :]
            q = (q * q_cos) + (rotate_half(q) * q_sin)
            
            if not is_cross_attn:
                # Manually apply RoPE to k
                k = (k * cos) + (rotate_half(k) * sin)

        # === Compute attention scores [B, H, T, S] ===
        attn_scores = self.compute_attention_scores(q, k)

        # --- 2) Add ALiBi relative-bias if enabled ---
        if alibi_bias_generator is not None:
            # Call the forward of the passed ALiBiPositionalBias instance
            # Assuming it generates bias based on seq_len (T x T or S x S)
            alibi_bias = alibi_bias_generator(batch_size=batch_size, seq_len=key_length)
            # Slice the bias if necessary (e.g., for T x S attention from S x S bias)
            if alibi_bias.shape[-2] == key_length and alibi_bias.shape[-1] == key_length:
                 alibi_bias = alibi_bias[..., -query_length:, :key_length] # Assumes causal slicing
            elif alibi_bias.shape[-2] == query_length and alibi_bias.shape[-1] == key_length:
                 pass # Shape already T x S
            else:
                 # Check broadcast compatibility carefully
                 try: _ = attn_scores + alibi_bias
                 except RuntimeError as e: raise ValueError(f"ALiB_ bias shape {alibi_bias.shape} not compatible with attention scores shape {attn_scores.shape}. Error: {e}")
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
        # attn_output shape: [batch, heads, q_len, head_dim]
        batch_size, num_heads, q_len, head_dim = attn_output.shape

        # Transpose back to [batch, q_len, heads, head_dim]
        attn_output = attn_output.transpose(1, 2) # No need for contiguous here

        # Reshape to the original embedding dimension. This is more robust
        # as subclasses might change head_dim for the value tensor.
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_probs = None
        return attn_output, attn_probs, present_key_value


@register_module("attention", "full")
class FullAttention(BaseMultiHeadAttention):
    """
    Standard multi-head attention with optional RoPE and ALiBi.

    This class implements a standard multi-head attention mechanism using
    scaled dot-product attention. It can be configured to use Rotary
    Positional Embeddings (RoPE) and ALiBi positional embeddings.

    Attributes:
        use_rope (bool): Whether to use RoPE.
        use_alibi (bool): Whether to use ALiBi.
        rotary_proj (Optional[RotaryPositionalEmbedding]): The RoPE module.
        alibi_bias_generator (Optional[ALiBiPositionalBias]): The ALiBi bias
            generator.
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
        """
        Initializes the FullAttention module.

        Args:
            embed_dim (int): The embedding dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            is_decoder (bool): Whether this module is used in a decoder.
            is_cross_attention (bool): Whether this module is used for cross-attention.
            bias (bool): Whether to include a bias in the linear projections.
            use_rope (bool): Whether to use RoPE.
            use_alibi (bool): Whether to use ALiBi.
            max_position_embeddings (int): The maximum sequence length for RoPE and ALiBi.
            rope_base (int): The base for RoPE frequencies.
            **kwargs: Additional keyword arguments for the parent class.
        """
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
        """
        Performs the forward pass of the attention layer.

        This method calls the `forward` method of the `BaseMultiHeadAttention`
        class, passing the RoPE and ALiBi modules if they are enabled.

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
    """
    Flash attention implementation.

    This class implements multi-head attention using the `flash_attn` library,
    which provides a more efficient implementation of attention.

    Attributes:
        softmax_scale (Optional[float]): The softmax scale.
        causal (bool): Whether to use causal attention.
    """
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
        """
        Initializes the FlashAttention module.

        Args:
            embed_dim (int): The embedding dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            is_decoder (bool): Whether this module is used in a decoder.
            is_cross_attention (bool): Whether this module is used for cross-attention.
            bias (bool): Whether to include a bias in the linear projections.
            softmax_scale (Optional[float]): The softmax scale.
            causal (bool): Whether to use causal attention.
            **kwargs: Additional keyword arguments for the parent class.
        """
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
        """
        Performs the forward pass of the attention layer.

        This method uses the `flash_attn_func` from the `flash_attn` library
        to perform the attention calculation. It does not support all the
        features of the `BaseMultiHeadAttention` class, such as RoPE and ALiBi.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            key_value_states (Optional[torch.Tensor]): The key and value states for
                cross-attention. Defaults to None.
            past_key_value: Ignored.
            attention_mask: Ignored.
            head_mask: Ignored.
            output_attentions: Ignored.
            use_cache: Ignored.
            position_ids: Ignored.
            rotary_proj: Ignored.
            alibi_bias_generator: Ignored.

        Returns:
            Tuple[torch.Tensor, None, None]: A tuple containing the attention
                output, None for the attention probabilities, and None for the
                key and value states.
        """
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
