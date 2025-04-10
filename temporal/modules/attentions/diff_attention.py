import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention


def lambda_init_fn(depth: int):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    If x => [B, n_kv_heads, seq_len, head_dim],
    repeating n_rep => [B, n_kv_heads*n_rep, seq_len, head_dim].
    """
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = True):
    """
    Example rotary embedding applying to (q, k) or similar. 
    x => [B, heads, T, dim], split into half-dims => rotate => rejoin.
    """
    # We'll assume x.shape[-1] is even
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    # cos, sin => shapes broadcastable to x1
    x1_rot = x1 * cos[..., :half] - x2 * sin[..., :half]
    x2_rot = x2 * cos[..., :half] + x1 * sin[..., :half]

    if interleaved:
        return torch.cat([x1_rot, x2_rot], dim=-1)
    else:
        return torch.cat([x1_rot, x2_rot], dim=-1)


# ----------------------------------------------------------------------
# DiffWistAttention
# ----------------------------------------------------------------------
class DiffWistAttentionWithCache(BaseMultiHeadAttention):
    """
    Same DIFF 'wistring' logic, but demonstrates how to handle
    `past_key_value` for decoder caching.
    """

    def __init__(
        self,
        config,
        depth: int,
        num_kv_heads: int = None,
    ):
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=config.is_decoder,
            bias=False
        )
        # 1) Setup: same as before
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else self.num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = self.embed_dim // self.num_heads // 2
        # Rebuild linear layers for Q, K, V
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Gating
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(
        self,
        hidden_states: torch.Tensor,                # [B, T, D]
        rel_pos: Tuple[torch.Tensor, torch.Tensor], # (cos, sin)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
          attn_output: [B, T, D]
          attn_probs:  [B, num_heads, T, T] (optional)
          present_key_value: (k, v) for caching
        """
        bsz, tgt_len, _ = hidden_states.size()
        src_len = tgt_len

        # 1) Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2) Reshape the new chunk
        # Q => [B, T, 2*num_heads, head_dim]
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        # K => [B, T, 2*num_kv_heads, head_dim]
        k = k.view(bsz, tgt_len, 2 * self.num_kv_heads, self.head_dim)
        # V => [B, T, num_kv_heads, 2*head_dim]
        v = v.view(bsz, tgt_len, self.num_kv_heads, 2 * self.head_dim)

        # 3) Rotary
        cos, sin = rel_pos
        k = apply_rotary_emb(k, cos, sin, interleaved=True)
        q = apply_rotary_emb(q, cos, sin, interleaved=True)

        # 4) Move heads to dimension 1 => [B, 2*num_heads, T, head_dim]
        q = q.transpose(1, 2)  # => [B, 2*num_heads, T, head_dim]
        k = k.transpose(1, 2)  # => [B, 2*num_kv_heads, T, head_dim]
        k = repeat_kv(k, self.n_rep)  # => [B, 2*num_heads, T, head_dim]

        v = v.transpose(1, 2)  # => [B, num_kv_heads, T, 2*head_dim]
        v = repeat_kv(v, self.n_rep)  # => [B, 2*num_heads, T, 2*head_dim]

        # 5) If we have past_key_value => shape is same as final K, V
        #    i.e. k => [B, 2*num_heads, old_len, head_dim],
        #             v => [B, 2*num_heads, old_len, 2*head_dim]
        # We cat along the seq_len dimension = 2
        if past_key_value is not None:
            pk, pv = past_key_value
            # pk => [B, 2*num_heads, old_len, head_dim]
            # pv => [B, 2*num_heads, old_len, 2*head_dim]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            src_len = k.size(2)  # now total length = old_len + new_len

        # Prepare present_key_value for next time
        present_key_value = (k, v)

        # 6) Scale Q
        q = q * (self.head_dim**-0.5)

        # 7) Compute attention logits => [B, 2*num_heads, T, src_len]
        attn_weights = torch.matmul(q, k.transpose(-1, -2))

        # 8) If no mask => causal. Otherwise, we add attention_mask
        if attention_mask is not None:
            # shape => [B, 1, T, src_len] or broadcastable
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(q.dtype)

        # 9) DIFF gating => [B, num_heads, 2, T, src_len]
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        # => shape [B, num_heads, T, src_len]

        # 10) Weighted sum => V
        # V => [B, 2*num_heads, src_len, 2*head_dim]
        # We'll reshape => [B, num_heads, 2, src_len, head_dim] for the einsum
        v = v.view(bsz, 2*self.num_heads, src_len, 2*self.head_dim)
        v = v.view(bsz, self.num_heads, 2, src_len, self.head_dim)

        attn_output = torch.einsum('bhts,bhstd->bhtd', attn_weights, v)

        # 11) RMSNorm => [B, num_heads, T, 2*head_dim]
        attn_output = self.subln(attn_output)

        # 12) Multiply by (1 - lambda_init)
        attn_output = attn_output * (1 - self.lambda_init)

        # 13) Reshape => [B, T, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # 14) Return attn_probs if requested
        attn_probs = None
        if output_attentions:
            # shape => [B, num_heads, T, src_len]
            attn_probs = attn_weights

        return attn_output, attn_probs, present_key_value


