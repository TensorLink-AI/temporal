import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def expand_mask(attention_mask: torch.Tensor, tgt_len: int, dtype: torch.dtype) -> torch.Tensor:
    if attention_mask.dim() != 2:
        raise ValueError(f"Expected 2D mask of shape [B, S], got {attention_mask.shape}.")

    bsz, src_len = attention_mask.shape
    expanded = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
    expanded = (1.0 - expanded.to(dtype)) * torch.finfo(dtype).min
    return expanded