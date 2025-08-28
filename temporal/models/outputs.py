from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class EncoderLayerOutput:
    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None

@dataclass
class DecoderLayerOutput:
    hidden_states: torch.Tensor
    self_attention_weights: Optional[torch.Tensor] = None
    cross_attention_weights: Optional[torch.Tensor] = None
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    aux_loss: Optional[torch.Tensor] = None
