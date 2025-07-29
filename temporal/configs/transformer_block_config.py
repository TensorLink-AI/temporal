from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY
from temporal.configs.attention_config import AttentionConfig, attention_config_from_dict
from temporal.configs.feedforward_config import FeedForwardConfig, feedforward_config_from_dict
from temporal.configs.normalization_config import NormalizationConfig, normalization_config_from_dict


@dataclass(frozen=True)
class TransformerBlockConfig(BaseConfig):
    """
    Base configuration for a generic transformer block.
    Specific block types should inherit from this class.
    """
    attention_config: AttentionConfig = field(default_factory=lambda: attention_config_from_dict({"type": "full"}))
    ffn_config: FeedForwardConfig = field(default_factory=lambda: feedforward_config_from_dict({"type": "standard"}))
    norm_config: NormalizationConfig = field(default_factory=lambda: normalization_config_from_dict({"type": "layer"}))
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.attention_config, AttentionConfig):
            raise ValueError("attention_config must be an AttentionConfig instance.")
        if not isinstance(self.ffn_config, FeedForwardConfig):
            raise ValueError("ffn_config must be a FeedForwardConfig instance.")
        if not isinstance(self.norm_config, NormalizationConfig):
            raise ValueError("norm_config must be a NormalizationConfig instance.")


@register_config_type("encoder_block")
@dataclass(frozen=True)
class EncoderBlockConfig(TransformerBlockConfig):
    """
    Configuration for a transformer encoder block.
    """
    type: str = "default_encoder"


@register_config_type("decoder_block")
@dataclass(frozen=True)
class DecoderBlockConfig(TransformerBlockConfig):
    """
    Configuration for a transformer decoder block.
    """
    type: str = "default_decoder"
    cross_attention_config: Optional[AttentionConfig] = None

    def __post_init__(self):
        super().__post_init__()
        if self.cross_attention_config is not None and not isinstance(self.cross_attention_config, AttentionConfig):
            raise ValueError("cross_attention_config must be an AttentionConfig instance or None.")


@register_config_type("adaptive_patch_transformer_block")
@dataclass(frozen=True)
class AdaptivePatchTransformerBlockConfig(TransformerBlockConfig):
    """
    Configuration for an adaptive patch transformer block.
    """
    type: str = "adaptive_patch_transformer"
    expansion_factor: int
    wrapped_block_type: str # e.g., "default_encoder" or "default_decoder"

    def __post_init__(self):
        super().__post_init__()
        if self.expansion_factor <= 0:
            raise ValueError("expansion_factor must be a positive integer.")
        if self.wrapped_block_type not in CONFIG_REGISTRY or \
           not issubclass(CONFIG_REGISTRY[self.wrapped_block_type], TransformerBlockConfig):
            raise ValueError(f"wrapped_block_type '{self.wrapped_block_type}' not found in registry or not a valid TransformerBlockConfig type.")


# Helper function for polymorphic creation
def transformer_block_config_from_dict(data: Dict[str, Any]) -> TransformerBlockConfig:
    block_type = data.get("type", "default_encoder") # Default to encoder if type not specified
    # Map config type names to registry keys if they differ
    type_to_registry_key = {
        "default_encoder": "encoder_block",
        "default_decoder": "decoder_block",
        "adaptive_patch_transformer": "adaptive_patch_transformer_block",
    }
    registry_key = type_to_registry_key.get(block_type, block_type)

    config_class = CONFIG_REGISTRY.get(registry_key)

    if not config_class or not issubclass(config_class, TransformerBlockConfig):
        raise ValueError(f"Unknown or invalid transformer block type: {block_type} (mapped to registry key: {registry_key})")

    # Manually handle nested configs if they are passed as dictionaries
    if "attention_config" in data and isinstance(data["attention_config"], dict):
        data["attention_config"] = attention_config_from_dict(data["attention_config"])
    if "cross_attention_config" in data and isinstance(data["cross_attention_config"], dict):
        data["cross_attention_config"] = attention_config_from_dict(data["cross_attention_config"])
    if "ffn_config" in data and isinstance(data["ffn_config"], dict):
        data["ffn_config"] = feedforward_config_from_dict(data["ffn_config"])
    if "norm_config" in data and isinstance(data["norm_config"], dict):
        data["norm_config"] = normalization_config_from_dict(data["norm_config"])
    
    # For AdaptivePatchTransformerBlockConfig, ensure the wrapped_block_type is resolved if it's a dict
    if registry_key == "adaptive_patch_transformer_block" and "wrapped_block_type" in data and isinstance(data["wrapped_block_type"], dict):
        data["wrapped_block_type"] = transformer_block_config_from_dict(data["wrapped_block_type"])

    return config_class.from_dict(data)
