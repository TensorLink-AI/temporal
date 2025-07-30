from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

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
    kwargs: Dict[str, Any] = field(default_factory=dict) # Added kwargs field

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

# --- NEW: Differential Attention Config ---
@register_config_type("diffwist_attention")
@dataclass(frozen=True, kw_only=True)
class DiffWistAttentionConfig(AttentionConfig):
    """
    Configuration for Differential-Wist (DiffWist) attention.
    """
    type: str = field(default="diffwist")
    depth: int = field(default=1) # Needed for lambda_init_fn
    num_kv_heads: Optional[int] = field(default=None) # For grouped-query attention
    use_rope: bool = field(default=True) # DiffWist specific RoPE flag
    rope_base: int = field(default=10000)
    max_position_embeddings: int = field(default=4096)

    def __post_init__(self):
        super().__post_init__()
        if self.depth <= 0:
            raise ValueError("depth must be a positive integer for DiffWistAttention.")
        if self.num_kv_heads is not None and self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be a positive integer or None.")
        if self.num_kv_heads is not None and self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for DiffWistAttention.")
        if self.use_rope and self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be > 0 if use_rope is True.")
        if self.use_rope and self.rope_base <= 0:
            raise ValueError("rope_base must be > 0 if use_rope is True.")

# --- NEW: Hybrid Attention Config ---
@register_config_type("hybrid_attention")
@dataclass(frozen=True, kw_only=True)
class HybridAttentionConfig(AttentionConfig):
    """
    Configuration for Hybrid Attention, allowing multiple attention kernel types.
    """
    type: str = field(default="hybrid")
    head_splits: List[int] = field(default_factory=list) # e.g., [2, 2] for 4 heads
    head_types: List[str] = field(default_factory=list) # e.g., ["full", "flash"]
    head_agg: str = field(default="concat") # "concat" or name of registered head_agg
    head_agg_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.head_splits or not self.head_types:
            raise ValueError("head_splits and head_types are required for HybridAttentionConfig.")
        if sum(self.head_splits) != self.num_heads:
            raise ValueError(f"Sum of head_splits ({sum(self.head_splits)}) must equal num_heads ({self.num_heads}).")
        if len(self.head_splits) != len(self.head_types):
            raise ValueError("Length of head_splits must match length of head_types.")
        if self.head_agg not in ["concat", "mean", "gated", "weighted_mean", "se", "moe", "head2head", "low_rank", "small_mlp"] and self.head_agg not in CONFIG_REGISTRY: # Example known aggregators
            print(f"Warning: Unknown head_agg type '{self.head_agg}'. Ensure it is registered as a 'head_agg' module.")

# Helper function for polymorphic creation
def attention_config_from_dict(data: Dict[str, Any]) -> AttentionConfig:
    attention_type = data.get("type", "full") # Default to 'full' if type not specified
    # Map config type names to registry keys if they differ (e.g., "full" -> "full_attention")
    type_to_registry_key = {
        "full": "full_attention",
        "flash": "flash_attention",
        "lse": "lse_attention",
        "diffwist": "diffwist_attention", # ADDED
        "hybrid": "hybrid_attention",     # ADDED
    }
    registry_key = type_to_registry_key.get(attention_type, attention_type + "_attention") # Fallback to type + _attention if not in map

    config_class = CONFIG_REGISTRY.get(registry_key)

    if not config_class or not issubclass(config_class, AttentionConfig):
        raise ValueError(f"Unknown or invalid attention_type: {attention_type} (mapped to registry key: {registry_key})")
    
    # Special handling for nested configs within HybridAttentionConfig
    if registry_key == "hybrid_attention" and "head_types" in data and isinstance(data["head_types"], list):
        # Recursively parse nested attention configs if they are dicts
        # This assumes head_types could contain full nested configs, not just strings
        data["head_types"] = [\
            attention_config_from_dict(cfg) if isinstance(cfg, dict) else cfg
            for cfg in data["head_types"]\
        ]

    return config_class.from_dict(data)
