# temporal/modules/heads/forecast_types.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
import torch

Tensor = torch.Tensor
Params = Union[Tensor, Dict[str, Tensor]]  # tensor for Gaussian/StudentT/DistPred paths; dict for MDN

@dataclass(frozen=True)
class HeadExtras:
    # Optional head-specific byproducts you might want to surface
    paths: Optional[Tensor] = None          # [B,T,F,K] or [B,T,K]  (DistPred)
    path_logits: Optional[Tensor] = None    # [B,T,K]               (DistPred)
    components: Optional[List[str]] = None  # mixture component names (MDN)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ForecastBundle:
    """
    A unified container for anything a caller might reasonably want.
    Shapes:
      point:      [B,T,F]
      quantiles:  [B,T,F,Q] or None
      params:     Tensor or Dict[str,Tensor] you actually predicted from
    """
    point: Tensor
    quantiles: Optional[Tensor]
    params: Optional[Params]
    extras: HeadExtras = field(default_factory=HeadExtras)
