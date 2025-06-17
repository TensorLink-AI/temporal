import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def expand_mask(
    attention_mask: torch.Tensor, tgt_len: int, dtype: torch.dtype
) -> torch.Tensor:
    """
    Expands a 2D attention mask to a 4D attention mask for self-attention.

    This utility function converts a 2D attention mask of shape `[batch_size, seq_len]`
    into a 4D mask of shape `[batch_size, 1, tgt_len, src_len]`, which is the format
    expected by many multi-head attention implementations.

    The conversion involves:
    1.  Expanding the dimensions to `[B, 1, 1, S]`.
    2.  Expanding the mask to cover the target sequence length, resulting in `[B, 1, T, S]`.
    3.  Inverting the mask values (0 becomes a large negative number, 1 becomes 0.0),
        as attention mechanisms typically use an additive mask.

    Args:
        attention_mask (torch.Tensor): The 2D input attention mask of shape
            `[batch_size, src_len]`.
        tgt_len (int): The length of the target sequence.
        dtype (torch.dtype): The target data type for the output mask.

    Returns:
        torch.Tensor: The expanded 4D attention mask.
    """
    if attention_mask.dim() != 2:
        raise ValueError(
            f"Expected a 2D attention mask of shape [batch_size, src_len], "
            f"but got a tensor with shape {attention_mask.shape}."
        )

    batch_size, src_len = attention_mask.shape
    # Expand to a 4D tensor: [B, S] -> [B, 1, 1, S]
    expanded_mask = attention_mask[:, None, None, :]
    # Expand along the target sequence length dimension: [B, 1, 1, S] -> [B, 1, T, S]
    expanded_mask = expanded_mask.expand(batch_size, 1, tgt_len, src_len)
    
    # Invert the mask: 1s become 0s, and 0s become a large negative number.
    # This is because the mask is added to the attention scores.
    inverted_mask = (1.0 - expanded_mask.to(dtype)) * torch.finfo(dtype).min
    
    return inverted_mask
