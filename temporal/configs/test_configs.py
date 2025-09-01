from dataclasses import dataclass

@dataclass
class LayerNormConfig:
    type: str = "layer"
