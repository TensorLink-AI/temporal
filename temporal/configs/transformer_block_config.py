from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY
from temporal.configs.attention_config import AttentionConfig, attention_config_from_dict
from temporal.configs.feedforward_config import FeedForwardConfig, feedforward_config_from_dict
from temporal.configs.normalization_config import NormalizationConfig, normalization_config_from_dict


@dataclass(frozen=True, kw_only=True)
class TransformerBlockConfig(BaseConfig):
    """
    Base configuration for a generic transformer block.
    Specific block types should inherit from this class.
    """
    attention_config: AttentionConfig = field(default_factory=lambda: attention_config_from_dict({"type": "full"}))
    ffn_config: FeedForwardConfig = field(default_factory=lambda: feedforward_config_from_dict({"type": "standard"}))
    normalization_config: NormalizationConfig = field(default_factory=lambda: normalization_config_from_dict({"type": "layer"}))
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.attention_config, AttentionConfig):
            raise ValueError("attention_config must be an AttentionConfig instance.")
        if not isinstance(self.ffn_config, FeedForwardConfig):
            raise ValueError("ffn_config must be a FeedForwardConfig instance.")
        if not isinstance(self.normalization_config, NormalizationConfig):
            raise ValueError("normalization_config must be a NormalizationConfig instance.")


@register_config_type("encoder_block")
@dataclass(frozen=True, kw_only=True)
class EncoderBlockConfig(TransformerBlockConfig):
    """
    Configuration for a transformer encoder block.
    """
    type: str = field(default="default_encoder") # Override base type and make it kw_only


@register_config_type("decoder_block")
@dataclass(frozen=True, kw_only=True)
class DecoderBlockConfig(TransformerBlockConfig):
    """
    Configuration for a transformer decoder block.
    """
    type: str = field(default="default_decoder") # Override base type and make it kw_only
    cross_attention_config: Optional[AttentionConfig] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.cross_attention_config is not None and not isinstance(self.cross_attention_config, AttentionConfig):
            raise ValueError("cross_attention_config must be an AttentionConfig instance or None.")


@register_config_type("adaptive_patch_transformer_block")
@dataclass(frozen=True, kw_only=True)
class AdaptivePatchTransformerBlockConfig(TransformerBlockConfig):
    """
    Configuration for an adaptive patch transformer block.
    """
    expansion_factor: int # Required kw-only field
    wrapped_block_type: str # Required kw-only field (e.g., "default_encoder" or "default_decoder")
    order: str = field(default='split_first')

    type: str = field(default="adaptive_patch_transformer") # Override base type and make it kw_only

    def __post_init__(self):
        super().__post_init__()
        if self.expansion_factor <= 0:
            raise ValueError("expansion_factor must be a positive integer.")
        if self.order not in ['split_first', 'merge_first']:
            raise ValueError(f"order must be one of 'split_first' or 'merge_first', but got {self.order}")
        # wrapped_block_type can be a string or a config object after from_dict parsing.
        # If it's a string, look it up in the registry.
        if isinstance(self.wrapped_block_type, str):
            if self.wrapped_block_type not in CONFIG_REGISTRY or \
               not issubclass(CONFIG_REGISTRY[self.wrapped_block_type], TransformerBlockConfig):
                raise ValueError(f"wrapped_block_type '{self.wrapped_block_type}' not found in registry or not a valid TransformerBlockConfig type.")
        elif not isinstance(self.wrapped_block_type, TransformerBlockConfig):
            raise ValueError(f"wrapped_block_type must be a string or a TransformerBlockConfig instance, got {type(self.wrapped_block_type).__name__}.")


# Helper function for polymorphic creation
def transformer_block_config_from_dict(data: Dict[str, Any]) -> TransformerBlockConfig:
    block_type = data.get("type", "default_encoder") # Default to encoder if type not specified
    # Map config type names to registry keys if they differ
    type_to_registry_key = {
        "default_encoder": "encoder_block",
        "default_decoder": "decoder_block",
        "encoder": "encoder_block",   # alias for quick API
        "decoder": "decoder_block",   # alias for quick API
        "adaptive_patch_transformer": "adaptive_patch_transformer_block",
    }
    registry_key = type_to_registry_key.get(block_type, block_type)

    config_class = CONFIG_REGISTRY.get(registry_key)

    if not config_class or not issubclass(config_class, TransformerBlockConfig):
        raise ValueError(f"Unknown or invalid transformer block type: {block_type} (mapped to registry key: {registry_key})")

    # Manually handle nested configs if they are passed as dictionaries
    # These will be passed to from_dict methods of their respective config types
    if "attention_config" in data and isinstance(data["attention_config"], dict):
        data["attention_config"] = attention_config_from_dict(data["attention_config"])
    if "cross_attention_config" in data and isinstance(data["cross_attention_config"], dict):
        data["cross_attention_config"] = attention_config_from_dict(data["cross_attention_config"])
    if "ffn_config" in data and isinstance(data["ffn_config"], dict):
        data["ffn_config"] = feedforward_config_from_dict(data["ffn_config"])
    if "normalization_config" in data and isinstance(data["normalization_config"], dict):
        data["normalization_config"] = normalization_config_from_dict(data["normalization_config"])
    
    # For AdaptivePatchTransformerBlockConfig, ensure the wrapped_block_type is resolved if it's a dict
    if registry_key == "adaptive_patch_transformer_block" and "wrapped_block_type" in data and isinstance(data["wrapped_block_type"], dict):
        data["wrapped_block_type"] = transformer_block_config_from_dict(data["wrapped_block_type"])

    return config_class.from_dict(data)
