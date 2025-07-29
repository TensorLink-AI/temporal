from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from temporal.configs.base_config import BaseConfig, register_config_type, CONFIG_REGISTRY


@dataclass(frozen=True, kw_only=True)
class AttentionConfig(BaseConfig):
    """
    Base configuration for a transformer attention mechanism.
    Specific attention types should inherit from this class.
    """
    num_heads: int = field(default=4)
    dropout: float = field(default=0.1)
    bias: bool = field(default=True)
    qk_layernorm: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__() # Call base class validation
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")


@register_config_type("full_attention")
@dataclass(frozen=True, kw_only=True)
class FullAttentionConfig(AttentionConfig):
    """
    Configuration for standard multi-head attention.
    """
    type: str = field(default="full") # Override base type and make it kw_only
    use_rope: bool = field(default=False)
    use_alibi: bool = field(default=False)
    rope_base: int = field(default=10000)
    max_position_embeddings: int = field(default=4096) # Required for RoPE/ALiBi in FullAttention

    def __post_init__(self):
        super().__post_init__() # Call base class validation
        if self.use_rope and self.use_alibi:
            # This is a warning, not an error, as some models might conceptually combine them
            print("Warning: Both use_rope and use_alibi are set to True. Behavior might be undefined.")


@register_config_type("flash_attention")
@dataclass(frozen=True, kw_only=True)
class FlashAttentionConfig(AttentionConfig):
    """
    Configuration for Flash Attention.
    """
    type: str = field(default="flash") # Override base type and make it kw_only
    softmax_scale: Optional[float] = field(default=None)
    causal: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__() # Call base class validation
        # Add any FlashAttention specific validation here if necessary

@register_config_type("lse_attention")
@dataclass(frozen=True, kw_only=True)
class LSEAttentionConfig(AttentionConfig):
    """
    Configuration for LSE Attention.
    """
    type: str = field(default="lse") # Override base type and make it kw_only
    use_rope: bool = field(default=False)
    use_alibi: bool = field(default=False)
    rope_base: int = field(default=10000)
    max_position_embeddings: int = field(default=4096) # Required for RoPE/ALiBi in LSEAttention

    def __post_init__(self):
        super().__post_init__() # Call base class validation
        if self.use_rope and self.use_alibi:
            print("Warning: Both use_rope and use_alibi are set to True. Behavior might be undefined for LSEAttention.")

# Update the from_dict to handle polymorphic creation
def attention_config_from_dict(data: Dict[str, Any]) -> AttentionConfig:
    attention_type = data.get("type", "full") # Default to 'full' if type not specified
    config_class = CONFIG_REGISTRY.get(attention_type + "_attention") # e.g., "full_attention"
    if not config_class or not issubclass(config_class, AttentionConfig):
        raise ValueError(f"Unknown or invalid attention_type: {attention_type}")
    return config_class.from_dict(data)
