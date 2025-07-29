from dataclasses import dataclass, asdict
from typing import Dict, Any
from temporal.configs.base_config import BaseConfig, register_config_type

@register_config_type("transformer_architecture")
@dataclass(frozen=True)
class TransformerArchitectureConfig(BaseConfig):
    """
    Configuration for transformer architecture specifying layer layout and weight sharing.
    """
    layout: str = "encoder-decoder"
    num_encoder_layers: int = 4
    num_decoder_layers: int = 2
    share_weights: bool = False

    def __post_init__(self):
        if self.layout not in ("encoder", "decoder", "encoder-decoder"):
            raise ValueError(f"layout must be 'encoder', 'decoder', or 'encoder-decoder', got {self.layout}")
        if self.num_encoder_layers <= 0 and self.layout in ("encoder", "encoder-decoder"):
            raise ValueError("num_encoder_layers must be > 0 for encoder or encoder-decoder layout.")
        if self.num_decoder_layers <= 0 and self.layout in ("decoder", "encoder-decoder"):
            raise ValueError("num_decoder_layers must be > 0 for decoder or encoder-decoder layout.")
