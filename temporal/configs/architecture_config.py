from dataclasses import dataclass, field
from typing import Dict, Any
from temporal.configs.base_config import BaseConfig, register_config_type

@register_config_type("transformer_architecture")
@dataclass(frozen=True, kw_only=True)
class TransformerArchitectureConfig(BaseConfig):
    """
    Configuration for transformer architecture specifying layer layout and weight sharing.
    """
    # All fields are keyword-only, so order doesn't strictly matter, but good practice
    # to put required fields first if there were any in the subclass.
    layout: str = field(default="encoder-decoder")
    num_encoder_layers: int = field(default=4)
    num_decoder_layers: int = field(default=2)
    share_weights: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__() # Call base class validation
        if self.layout not in ("encoder", "decoder", "encoder-decoder"):
            raise ValueError(f"layout must be 'encoder', 'decoder', or 'encoder-decoder', got {self.layout}")
        if self.num_encoder_layers <= 0 and self.layout in ("encoder", "encoder-decoder"):
            raise ValueError("num_encoder_layers must be > 0 for encoder or encoder-decoder layout.")
        if self.num_decoder_layers <= 0 and self.layout in ("decoder", "encoder-decoder"):
            raise ValueError("num_decoder_layers must be > 0 for decoder or encoder-decoder layout.")
