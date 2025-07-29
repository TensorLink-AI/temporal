
import torch
from torch import nn
from temporal.modules.attentions.base_attention import BaseMultiHeadAttention
from temporal.registry import register_module

@register_module("attention", "lse")
class LSEAttention(BaseMultiHeadAttention):
    """
    LSEAttention (Log-Sum-Exp Attention) is a numerically stable self-attention mechanism
    that replaces the standard softmax with a combination of the Log-Sum-Exp trick and
    GELU activation. This approach aims to address issues like numerical instability
    and entropy collapse often observed in softmax-based attention, especially in
    long-term multivariate forecasting.

    The attention mechanism follows these steps:
    1. Compute the Log-Sum-Exp (LSE) of the scores for numerical stability.
    2. Apply the GELU non-linearity to the LSE result.
    3. Re-normalize the scores similar to softmax to obtain probabilities.
    4. Apply attention mask and dropout.
    5. Compute the weighted sum of values.

    This attention mechanism is registered under the "attention" registry with the key "lse".
    """
    def forward(self, q, k, v, attn_mask=None, head_mask=None, output_attentions=False):
        """
        Performs the forward pass of the LSE (Log-Sum-Exp) Attention mechanism.

        Args:
            q (torch.Tensor): Query tensor of shape [B, H, T, D_head].
            k (torch.Tensor): Key tensor of shape [B, H, T, D_head].
            v (torch.Tensor): Value tensor of shape [B, H, T, D_head].
            attn_mask (torch.Tensor, optional): An attention mask tensor.
                If 2D (batch_size, sequence_length), it will be expanded to [B, 1, 1, T].
                Elements with value 0 will be masked out. Defaults to None.
            head_mask (torch.Tensor, optional): An optional mask to nullify selected heads.
                Shape can be [H,] or [B, H, 1, 1]. Defaults to None.
            output_attentions (bool, optional): If set to True, the attention probabilities
                are returned in addition to the output. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - out (torch.Tensor): The output tensor after attention, shape [B, H, T, D_head].
                - attn_weights (torch.Tensor or None): The attention probabilities if `output_attentions` is True,
                  otherwise None. Shape [B, H, T, T].
                - present_key_value (None): This attention type does not produce present_key_value, always None.
        """
        # q,k,v: [B, H, T, D_head]
        dk = q.shape[-1]
        scores = torch.einsum("bhqd, bhkd -> bhqk", q, k) / dk**0.5

        # --- ① Numerical-stable LSE ---
        a = scores.max(dim=-1, keepdim=True).values      # “a” in the paper
        lse = a + torch.log(torch.exp(scores - a).sum(dim=-1, keepdim=True))

        # --- ② GELU non-linearity ---
        lse = nn.functional.gelu(lse)

        # --- ③ Re-normalise like softmax ---
        probs = torch.exp(scores - lse)

        # mask & dropout
        if attn_mask is not None:
            # expand attention mask to fit the shape of scores
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, T]
            probs = probs.masked_fill(attn_mask == 0, 0.0)
        
        # Apply head mask if provided
        if head_mask is not None:
            probs = probs * head_mask

        # Renormalize after masking if needed, based on the original paper this is done after masking
        probs = self.dropout(probs / probs.sum(dim=-1, keepdim=True))

        out = torch.einsum("bhqk, bhkd -> bhqd", probs, v)
        
        # We need to return attention weights if output_attentions is True
        attn_weights = probs if output_attentions else None
        
        return out, attn_weights, None # present_key_value is not used in this attention type
