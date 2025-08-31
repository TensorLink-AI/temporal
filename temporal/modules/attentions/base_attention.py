# temporal/modules/attentions/base_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from temporal.registry.core import register_module
from temporal.modules.embedders.embedding import (
    RotaryPositionalEmbedding,
    ALiBiPositionalBias,
    apply_rotary_pos_emb,
    rotate_half,
)
from temporal.modules.attentions.destationary import Projector
from temporal.configs.attention_config import DestationaryProjectorConfig

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
        destationary_projector: Optional[DestationaryProjectorConfig] = None,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if self.head_dim % 2 != 0:
            print(f"Warning: head_dim ({self.head_dim}) not even—RoPE may break.")
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # --- query/key LayerNorm ---
        self.use_qk_layernorm = kwargs.get('qk_layernorm', False)
        if self.use_qk_layernorm:
            # normalize each head's vector of size head_dim
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        self.destationary_projector = destationary_projector
        if self.destationary_projector is not None:
            enc_in = kwargs.get('enc_in')
            seq_len = kwargs.get('seq_len')
            if enc_in is None or seq_len is None:
                raise ValueError("Destationary projector requires 'enc_in' and 'seq_len' in kwargs.")
            self.tau_learner   = Projector(enc_in=enc_in, seq_len=seq_len,
                                           hidden_dims=self.destationary_projector.hidden_dims,
                                           hidden_layers=self.destationary_projector.hidden_layers,
                                           output_dim=1)
            self.delta_learner = Projector(enc_in=enc_in, seq_len=seq_len,
                                           hidden_dims=self.destationary_projector.hidden_dims,
                                           hidden_layers=self.destationary_projector.hidden_layers,
                                           output_dim=seq_len)

        self.dropout = dropout
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention

    def _compute_attn_probs(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Computes attention probabilities from scores.
        Standard implementation uses softmax.
        """
        return F.softmax(scores, dim=-1)

    def compute_attention_scores(self, q: torch.Tensor, k: torch.Tensor, x_raw: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the attention scores.

        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.

        Returns:
            torch.Tensor: The attention scores.
        """
        scores = torch.matmul(q * self.scaling, k.transpose(-1, -2))
        if self.destationary_projector is not None and x_raw is not None:
            mean_enc = x_raw.mean(1, keepdim=True).detach()
            std_enc = torch.sqrt(torch.var(x_raw, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            tau = self.tau_learner(x_raw, std_enc)
            delta = self.delta_learner(x_raw, mean_enc)
            # broadcast
            tau = torch.exp(tau).unsqueeze(1).unsqueeze(1) if tau is not None else 1.0  # B x 1 x 1 x 1
            delta = delta.unsqueeze(1).unsqueeze(1) if delta is not None else 0.0       # B x 1 x 1 x L
            scores = scores * tau + delta[..., : scores.size(-1)]
        return scores

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
            rotary_proj: Optional[nn.Module] = None,
            alibi_bias_generator: Optional[nn.Module] = None,
            x_raw: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs the forward pass of the attention layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, T, E].
            key_value_states (Optional[torch.Tensor]): For cross-attention, encoder states [B, S, E].
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Cached (k, v) of shape
                [B, H, K, Dh], [B, H, K, Dh] for KV caching during decoding.
            attention_mask (Optional[torch.Tensor]): Additive attention mask broadcastable to [B, 1, T_q, T_k].
            head_mask (Optional[torch.Tensor]): Optional per-head mask.
            output_attentions (bool): Whether to return attention probabilities.
            use_cache (bool): Whether to return present key/value tensors for caching.
            position_ids (Optional[torch.LongTensor]): Absolute positions [B, T_q] for RoPE.
            rotary_proj (Optional[nn.Module]): RoPE module (RotaryPositionalEmbedding).
            alibi_bias_generator (Optional[nn.Module]): ALiBi bias generator.
            x_raw (Optional[torch.Tensor]): Raw encoder inputs for de-stationary projector.

        Returns:
            Tuple containing:
                - Output tensor [B, T, E]
                - Optional attention probabilities [B, H, T_q, T_k] if output_attentions is True
                - Optional present key/value tensors for caching if use_cache is True.
        """
        B, T, _ = hidden_states.size()
        is_cross = key_value_states is not None
        kv_source = key_value_states if is_cross else hidden_states

        # Project and reshape Q, K, V
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_source).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_source).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_layernorm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # --- BUGFIX: Explicit RoPE application before caching ---
        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value[0].shape[-2]

        if rotary_proj is not None:
            # Path A: use explicit absolute positions when provided
            if position_ids is not None:
                needed = int(position_ids.max().item()) + 1
                cos_all, sin_all = rotary_proj(q, seq_len=needed)  # [needed, Dh], correct device/dtype
                # apply_rotary_pos_emb will gather by position_ids (broadcasts to [B, 1, T, Dh])
                pos_slice = position_ids[..., -T:]  # [B, T]
                q_rot, k_rot = apply_rotary_pos_emb(q, k, cos_all, sin_all, position_ids=pos_slice)
                q = q_rot
                if not is_cross:
                    k = k_rot
            else:
                # Path B: derive absolute offset from cache length
                total_len = past_len + T
                cos, sin = rotary_proj(q, seq_len=total_len)  # [total_len, Dh] on q's device/dtype
                # Slice only the NEW tokens (past_len : past_len + T) and broadcast to [B, H, T, Dh]
                cos_slice = cos[past_len:total_len][None, None, :, :]
                sin_slice = sin[past_len:total_len][None, None, :, :]
                q = (q * cos_slice) + (rotate_half(q) * sin_slice)
                if not is_cross:
                    k = (k * cos_slice) + (rotate_half(k) * sin_slice)

        # Handle caching AFTER RoPE has been applied to the new k
        present = None
        if use_cache and not is_cross:
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            present = (k, v)

        # --- The rest of the logic remains the same ---
        scores = self.compute_attention_scores(q, k, x_raw=x_raw)

        if alibi_bias_generator is not None:
            bias = alibi_bias_generator(batch_size=B, seq_len=k.size(-2))
            scores = scores + bias.to(dtype=scores.dtype, device=scores.device)

        if attention_mask is not None:
            # Accept bool masks by converting to additive (large negative) values
            if attention_mask.dtype == torch.bool:
                neg_inf = torch.finfo(scores.dtype).min
                attention_mask = attention_mask.masked_fill(attention_mask, neg_inf)

            # --- FIX START: Robustly handle attention mask shapes ---
            # Normalize to [B, 1, T_q, T_k]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

            Tq = q.size(-2)
            Tk = k.size(-2)
            mq = attention_mask.size(-2)
            mk = attention_mask.size(-1)
            start_q = max(0, mq - Tq)
            end_q   = mq
            end_k   = min(mk, Tk)
            attention_mask = attention_mask[:, :, start_q:end_q, :end_k]
            attention_mask = attention_mask.to(dtype=scores.dtype, device=scores.device)
            # --- FIX END ---
            scores = scores + attention_mask

        probs = self._compute_attn_probs(scores)
        probs = F.dropout(probs, p=self.dropout, training=self.training)

        if head_mask is not None:
            probs = probs * head_mask.view(1, -1, 1, 1)

        out = torch.matmul(probs, v.to(probs.dtype))
        out = out.transpose(1, 2).reshape(B, T, -1)
        out = self.out_proj(out)

        return (out, probs if output_attentions else None, present)


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
        **kwargs
    ):
        """
        Initializes the FullAttention module.
        """
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, **kwargs)

        # Retrieve optional parameters in a backward-compatible way
        self.use_rope = kwargs.get('use_rope', False)
        self.use_alibi = kwargs.get('use_alibi', False)
        max_position_embeddings = kwargs.get('max_position_embeddings', 4096)
        rope_base = kwargs.get('rope_base', 10000)

        self.rotary_proj = None
        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError("RoPE requires head_dim to be even.")
            self.rotary_proj = RotaryPositionalEmbedding(
                d_model=self.head_dim,
                max_seq_len=max_position_embeddings,
                base=rope_base,
            )

        self.alibi_bias_generator = None
        if self.use_alibi:
            self.alibi_bias_generator = ALiBiPositionalBias(
                num_heads=self.num_heads,
                max_seq_len=max_position_embeddings,
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
        x_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs the forward pass of the attention layer.

        This method calls the `forward` method of the `BaseMultiHeadAttention`
        class, passing the RoPE and ALiBi modules if they are enabled.
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
            alibi_bias_generator=self.alibi_bias_generator,
            x_raw=x_raw,
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
        **kwargs
    ):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, **kwargs)
        if not _flash_attn_available:
            raise ImportError("FlashAttention requires flash_attn.")

        self.softmax_scale = kwargs.get('softmax_scale', None)
        self.causal = kwargs.get('causal', self.is_decoder and not self.is_cross_attention)

        if self.is_cross_attention:
            print("Warning: FlashAttention cross-attention not fully supported.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        use_cache=False,
        position_ids=None,
        rotary_proj=None,
        alibi_bias_generator=None,
        x_raw: Optional[torch.Tensor] = None,
    ):
        """
        Performs the forward pass of the attention layer using flash_attn_func.
        """
        if flash_attn_func is None:
            raise ImportError("flash_attn_func is not available.")
        if use_cache or past_key_value is not None:
            print("Warning: FlashAttention KV caching is not implemented.")

        B, T, _ = hidden_states.shape
        kv_source = key_value_states if self.is_cross_attention and key_value_states is not None else hidden_states
        q, k, v = self.q_proj(hidden_states), self.k_proj(kv_source), self.v_proj(kv_source)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, -1, self.num_heads, self.head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim)

        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=self.causal
        )
        attn_output = attn_output.view(B, T, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, None
