# ---- Output heads --------------------------------------------------
from .output_heads import (
    LinearOutputHead,
    GaussianHead,
    # TDistributionHead, # Removed - Was commented out in output_heads.py
    QuantileRegressionOutputHead, # Updated name
    DistPredHead # Added new head
)

__all__ = [
    "LinearOutputHead",
    "GaussianHead",
    # "TDistributionHead", # Removed
    "QuantileRegressionOutputHead", # Updated name
    "DistPredHead", # Added new head
]
