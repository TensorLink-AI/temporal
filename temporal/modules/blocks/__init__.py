# ---- Transformer blocks -------------------------------------------
from .block_multiscale import MultiScaleTransformerBlock
from .effitime_block   import EfficientTimeBlock

# 👉 register your "standard" block here too, if that’s the one the config expects

__all__ = [
    "MultiScaleTransformerBlock",
    "EfficientTimeBlock",
]
