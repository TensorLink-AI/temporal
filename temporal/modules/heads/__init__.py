# ---- Output heads --------------------------------------------------
from .output_heads import (
    LinearOutputHead,
    GaussianHead,
    TDistributionHead,
    MultiQuantileHead,
)

__all__ = [
    "LinearOutputHead",
    "GaussianHead",
    "TDistributionHead",
    "MultiQuantileHead",
]
