# ---- Transformer blocks -------------------------------------------
from .block_multiscale import MultiScaleBlock
from .effitime_block   import EffiTimeBlockHybridConvFirst

# 👉 register your "standard" block here too, if that’s the one the config expects

__all__ = [
    "MultiScaleBlock",
    "EffiTimeBlockHybridConvFirst",
]
