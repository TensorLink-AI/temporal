# temporal/configs/transformer_config.py

# Preserve original imports
from temporal.configs.basetimeseriesconfig import BaseTimeSeriesConfig
from typing import Optional, List, Dict, Any # Ensure necessary types are imported
import math # Keep for potential use
import numpy as np # Keep for potential use

# Preserve original classes before AttentionConfig
class TransformerArchitectureConfig:
    """
    Configuration for transformer architecture specifying layer layout and weight sharing.

    Args:
        layout (str): One of 'encoder', 'decoder', or 'encoder-decoder'.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        share_weights (bool): Whether to share weights between encoder and decoder.
    """
    def __init__(self, layout="encoder-decoder", num_encoder_layers=4, num_decoder_layers=2, share_weights=False):
        assert layout in ("encoder", "decoder", "encoder-decoder"), \
            f"layout must be 'encoder', 'decoder', or 'encoder-decoder', got {layout}"
        self.layout = layout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.share_weights = share_weights

    def to_dict(self):
        """
        Convert architecture configuration to a dictionary.

        Returns:
            dict: Dictionary containing architecture settings.
        """
        return {
            "layout": self.layout,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "share_weights": self.share_weights
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create a TransformerArchitectureConfig from a dictionary.

        Args:
            d (dict): Dictionary with keys matching __init__ parameters.

        Returns:
            TransformerArchitectureConfig: New instance.
        """
        # Use .get for defaults
        return cls(
            layout=d.get("layout", "encoder-decoder"),
            num_encoder_layers=d.get("num_encoder_layers", 4),
            num_decoder_layers=d.get("num_decoder_layers", 2),
            share_weights=d.get("share_weights", False)
        )

# --- Modified AttentionConfig --- 
class AttentionConfig:
    """
    Configuration for a transformer attention mechanism.

    Args:
        attention_type (str): Type of attention (e.g., 'full', 'local', 'flash', 'hybrid').
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability for attention weights.
        bias (bool): Whether projection layers (QKV, Out) should have bias.
        use_rope (bool): If True, enables Rotary Positional Embeddings.
        use_alibi (bool): If True, enables Attention with Linear Biases.
        rope_base (int): Base frequency for RoPE calculations if use_rope is True.
        kwargs (dict): Additional keyword arguments specific to the attention_type implementation.
    """
    def __init__(self, 
                 attention_type="full", 
                 num_heads=4, 
                 dropout=0.1, 
                 bias=True, # Added bias argument
                 use_rope=False, # Added use_rope flag
                 use_alibi=False, # Added use_alibi flag
                 rope_base=10000, # Added rope_base parameter
                 kwargs=None):
        # Preserve original dropout validation
        assert isinstance(dropout, (float, int)) and 0.0 <= dropout <= 1.0, \
            f"dropout must be in [0, 1], got {dropout}"
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.dropout = float(dropout)
        # Store new fields
        self.bias = bias
        self.use_rope = use_rope
        self.use_alibi = use_alibi
        self.rope_base = rope_base
        self.kwargs = kwargs or {}

    def to_dict(self):
        """
        Convert attention configuration to a dictionary.
        """
        # Include new fields in the dictionary
        return {
            "attention_type": self.attention_type,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "bias": self.bias,
            "use_rope": self.use_rope,
            "use_alibi": self.use_alibi,
            "rope_base": self.rope_base,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create an AttentionConfig from a dictionary.
        """
        # Include new fields with defaults when creating from dict
        return cls(
            attention_type=d.get("attention_type", "full"),
            num_heads=d.get("num_heads", 4),
            dropout=d.get("dropout", 0.1),
            bias=d.get("bias", True),
            use_rope=d.get("use_rope", False),
            use_alibi=d.get("use_alibi", False),
            rope_base=d.get("rope_base", 10000),
            kwargs=d.get("kwargs", {})
        )

    def validate(self):
        """
        Validate the attention configuration for correctness.
        """
        # Preserve original validation
        # assert self.attention_type in {"full", "local", "flash", "diffwist", "hybrid"}, \
        #     f"Unknown attention_type: {self.attention_type}" # Allow flexibility for now
        assert self.num_heads > 0, "num_heads must be > 0"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0, 1]"
        # Add validation for mutually exclusive flags if needed
        if self.use_rope and self.use_alibi:
            print("Warning: Both use_rope and use_alibi are set to True in AttentionConfig. Behavior might be undefined depending on implementation.")

# --- End Modified AttentionConfig --- 

# Preserve original classes after AttentionConfig
class OutputHeadConfig:
    """
    Configuration for the output head of the transformer model.

    Args:
        type (str): Type of output head (e.g., 'linear').
        output_size (int): Dimensionality of the head output.
        kwargs (dict): Additional keyword arguments for the output head.
    """
    def __init__(self, type="linear", output_size=None, kwargs=None):
        self.type = type
        self.output_size = output_size
        self.kwargs = kwargs or {}

    def to_dict(self):
        """
        Convert output head configuration to a dictionary.

        Returns:
            dict: Dictionary of output head settings.
        """
        return {
            "type": self.type,
            "output_size": self.output_size,
            "kwargs": self.kwargs,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create an OutputHeadConfig from a dictionary.

        Args:
            d (dict): Dictionary with keys matching __init__ parameters.

        Returns:
            OutputHeadConfig: New instance.
        """
        return cls(
            type=d.get("type", "linear"),
            output_size=d.get("output_size"),
            kwargs=d.get("kwargs", {})
        )


class TransformerAttentionBlockConfig: # Keep this class definition
    """
    Configuration grouping attention configs for encoder and decoder blocks.

    Args:
        encoder_attention (AttentionConfig): Config for encoder self-attention.
        decoder_attention (AttentionConfig): Config for decoder self-attention.
        decoder_cross_attention (AttentionConfig): Config for decoder cross-attention.
    """
    def __init__(self, encoder_attention=None, decoder_attention=None, decoder_cross_attention=None):
        # Use AttentionConfig() to get defaults correctly
        self.encoder_attention = encoder_attention or AttentionConfig()
        self.decoder_attention = decoder_attention or AttentionConfig()
        self.decoder_cross_attention = decoder_cross_attention or AttentionConfig()

    def to_dict(self):
        """
        Convert attention block configuration to a dictionary.

        Returns:
            dict: Dictionary of nested attention block settings.
        """
        return {
            "encoder_attention": self.encoder_attention.to_dict(),
            "decoder_attention": self.decoder_attention.to_dict(),
            "decoder_cross_attention": self.decoder_cross_attention.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create a TransformerAttentionBlockConfig from a dictionary.

        Args:
            d (dict): Dictionary with nested attention configurations.

        Returns:
            TransformerAttentionBlockConfig: New instance.
        """
        return cls(
            encoder_attention=AttentionConfig.from_dict(d.get("encoder_attention", {})),
            decoder_attention=AttentionConfig.from_dict(d.get("decoder_attention", {})),
            decoder_cross_attention=AttentionConfig.from_dict(d.get("decoder_cross_attention", {}))
        )

class FeedForwardConfig: # Keep this class definition
    """
    Configuration for the feed-forward network sublayer in transformer blocks.
    Updated to support MoE parameters.
    """
    def __init__(self,
                 type="standard",
                 intermediate_size=256,
                 activation="gelu",
                 dropout=0.1,
                 bias=True, # Added bias
                 # MoE specific args
                 num_experts: Optional[int] = None,
                 top_k: Optional[int] = None,
                 expert_intermediate_size: Optional[int] = None,
                 load_balancing_coef: float = 0.01,
                 kwargs=None):
        self.type = type
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.dropout = dropout
        self.bias = bias
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_intermediate_size = expert_intermediate_size
        self.load_balancing_coef = load_balancing_coef
        self.kwargs = kwargs or {}

    def to_dict(self):
        return {
            "type": self.type,
            "intermediate_size": self.intermediate_size,
            "activation": self.activation,
            "dropout": self.dropout,
            "bias": self.bias,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_intermediate_size": self.expert_intermediate_size,
            "load_balancing_coef": self.load_balancing_coef,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        # Handle potential missing MoE keys with defaults
        return cls(
            type=d.get("type", "standard"),
            intermediate_size=d.get("intermediate_size", 256),
            activation=d.get("activation", "gelu"),
            dropout=d.get("dropout", 0.1),
            bias=d.get("bias", True),
            num_experts=d.get("num_experts"),
            top_k=d.get("top_k"),
            expert_intermediate_size=d.get("expert_intermediate_size"),
            load_balancing_coef=d.get("load_balancing_coef", 0.01),
            kwargs=d.get("kwargs", {})
        )

    def validate(self):
        # Add validation for MoE specific fields if type is 'moe'
        if self.type == "moe":
            assert self.num_experts is not None and self.num_experts > 0, "num_experts must be > 0 for MoE"
            assert self.top_k is not None and 0 < self.top_k <= self.num_experts, "top_k must be > 0 and <= num_experts"
        # Add other validations as needed
        pass

class EmbeddingConfig: # Keep this class definition
    """
    Configuration for embedding layers in the transformer.

    Args:
        type (str): Type of embedding ('value', 'patch', 'sinusoidal', 'learned', etc.).
        dropout (float): Dropout probability applied to embeddings.
        embedding_dim (int): Dimensionality of the embedding output (often d_model).
        kwargs (dict): Additional arguments for embedding implementation.
    """
    def __init__(self, type="value", dropout=0.1, embedding_dim=None, kwargs=None):
        self.type = type
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.kwargs = kwargs or {}

    def to_dict(self):
        """
        Convert embedding configuration to a dictionary.
        """
        return {
            "type": self.type,
            "dropout": self.dropout,
            "embedding_dim": self.embedding_dim,
            "kwargs": self.kwargs,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create an EmbeddingConfig from a dictionary.
        """
        return cls(
            type=d.get("type", "value"),
            dropout=d.get("dropout", 0.1),
            embedding_dim=d.get("embedding_dim", None),
            kwargs=d.get("kwargs", {})
        )

    def validate(self):
        """
        Validate the embedding configuration.
        """
        # Example validation - needs expansion based on actual registered types
        # assert self.type in {"value", "patch", "sinusoidal", "learned_abs", "rotary", ...}, \
        #     f"Invalid embedding type: {self.type}"
        pass

class HeadAggregationConfig: # Keep this class definition
    """
    Configuration for combining multiple output heads.
    """
    def __init__(self, type="mean", kwargs=None):
        self.type = type
        self.kwargs = kwargs or {}

    def to_dict(self):
        return {
            "type": self.type,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

class NormalizationConfig: # Keep this class definition
    """
    Configuration for normalization layers.
    """
    def __init__(self, norm_type="layer", eps=1e-5, kwargs: Optional[Dict[str, Any]] = None):
        self.norm_type = norm_type
        self.eps = eps
        self.kwargs = kwargs or {}

    def to_dict(self):
        return {
            "norm_type": self.norm_type,
            "eps": self.eps,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

class QuantizerConfig: # Keep this class definition
    """
    Configuration for time series quantization.
    """
    def __init__(self, quantization_type="mean_std_bins", vocab_size=4096, num_features=1, **kwargs):
        self.quantization_type = quantization_type
        self.vocab_size = vocab_size
        self.num_features = num_features
        self.kwargs = kwargs

    def to_dict(self):
        return {
            "quantization_type": self.quantization_type,
            "vocab_size": self.vocab_size,
            "num_features": self.num_features,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

class TransformerBlockConfig: # Keep this class definition
    """
    Configuration for a generic transformer block.
    """
    def __init__(self, block_type="default_encoder", attention_config=None, ffn_config=None, norm_config=None, kwargs=None):
        self.block_type = block_type
        # Ensure AttentionConfig, FeedForwardConfig, NormalizationConfig are instantiated with defaults
        self.attention_config = attention_config or AttentionConfig()
        self.ffn_config = ffn_config or FeedForwardConfig()
        self.norm_config = norm_config or NormalizationConfig()
        self.kwargs = kwargs or {}

    def to_dict(self):
        return {
            "block_type": self.block_type,
            "attention_config": self.attention_config.to_dict(), # No longer optional here
            "ffn_config": self.ffn_config.to_dict(),             # No longer optional here
            "norm_config": self.norm_config.to_dict(),             # Added norm_config
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        # Pass nested dicts to sub-config from_dict methods
        return cls(
            block_type=d.get("block_type", "default_encoder"),
            attention_config=AttentionConfig.from_dict(d.get("attention_config", {})),
            ffn_config=FeedForwardConfig.from_dict(d.get("ffn_config", {})),
            norm_config=NormalizationConfig.from_dict(d.get("norm_config", {})),
            kwargs=d.get("kwargs", {})
        )

class TransformerTimeSeriesConfig(BaseTimeSeriesConfig): # Keep this class definition
    """
    Configuration for a transformer-based time-series forecasting model.
    Extends ``BaseTimeSeriesConfig`` with transformer-specific options.
    """

    attribute_map = BaseTimeSeriesConfig.attribute_map.copy() # Start with base map

    def __init__(
        self,
        # --- Base --- 
        input_dim:                 int,
        context_length:            int,
        prediction_length:         int,
        # --- Optional Base --- 
        output_dim:                Optional[int] = None,
        static_dim:                int = 0,
        dynamic_dim:               int = 0,
        past_dynamic_dim:          int = 0,
        static_cardinalities:      Optional[List[int]] = None,
        dynamic_cardinalities:     Optional[List[int]] = None,
        past_dynamic_cardinalities:Optional[List[int]] = None,
        static_embedding_dim:      Optional[List[int]] = None,
        dynamic_embedding_dim:     Optional[List[int]] = None,
        past_dynamic_embedding_dim:Optional[List[int]] = None,
        time_features:             Optional[List[str]] = None,
        loss_config:               Dict = {"type": "mse"},
        scaling:                   bool = True,
        # --- Transformer Specific ---
        model_type:                str = "transformer",
        d_model:                   int = 64,
        hidden_dropout_prob:       float = 0.1,
        max_position_embeddings:   int = 4096,
        architecture:              Optional[TransformerArchitectureConfig] = None,
        value_embedding_config:    Optional[EmbeddingConfig] = None,
        positional_embedding_config: Optional[EmbeddingConfig] = None,
        encoder_blocks:            Optional[List[TransformerBlockConfig]] = None,
        decoder_blocks:            Optional[List[TransformerBlockConfig]] = None,
        output_head_config:        Optional[OutputHeadConfig] = None,
        norm_config:               Optional[NormalizationConfig] = None,
        head_agg_config:           Optional[HeadAggregationConfig] = None,
        # --- Tokenization ---
        quantizer_config:          Optional[QuantizerConfig] = None,
        vocab_size:                Optional[int] = None,
        decoder_start_token_id:    Optional[int] = None,
        # --- Legacy/Control --- 
        num_quantiles:             Optional[int] = None,
        quantiles:                 Optional[List[float]] = None,
        output_attentions:         bool = False,
        output_hidden_states:      bool = False,
        use_teacher_forcing:       bool = True,
        # --- Deprecated/Legacy --- 
        attention_blocks:          Optional[Any] = None, # Mark as deprecated
        feedforward_config:        Optional[Any] = None, # Mark as deprecated
        # --- BaseTimeSeries Compatibility ---
        feature_size:              Optional[int] = None,
        autoregressive:            Optional[bool] = None,
        is_decoder:                Optional[bool] = None,
        # --- Catch all ---
        **kwargs: Any,
    ):
        # --- Handle Deprecated Args --- 
        if attention_blocks is not None:
            print("Warning: `attention_blocks` config key is deprecated. Configure attention within `encoder_blocks`/`decoder_blocks`.")
        if feedforward_config is not None:
            print("Warning: `feedforward_config` key is deprecated. Configure FFN within `encoder_blocks`/`decoder_blocks`.")

        # --- Map/Default Transformer Specific Settings --- 
        self.model_type = model_type
        self.d_model = d_model
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.architecture = architecture or TransformerArchitectureConfig()
        self.value_embedding_config = value_embedding_config or EmbeddingConfig(type="value", kwargs={"feature_size": input_dim, "d_model": d_model})
        self.positional_embedding_config = positional_embedding_config or EmbeddingConfig(type="sinusoidal", kwargs={"max_seq_len": max_position_embeddings, "d_model": d_model})
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        _output_dim = output_dim if output_dim is not None else input_dim
        self.output_head_config = output_head_config or OutputHeadConfig(type="linear", output_size=_output_dim)
        self.norm_config = norm_config or NormalizationConfig()
        self.head_agg_config = head_agg_config or HeadAggregationConfig()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_teacher_forcing = use_teacher_forcing

        # --- Tokenization --- 
        self.quantizer_config = quantizer_config
        self.vocab_size = vocab_size
        self.decoder_start_token_id = decoder_start_token_id

        # --- Set Base Settings --- 
        _target_dim = output_dim if output_dim is not None else input_dim
        _autoregressive = autoregressive if autoregressive is not None else (self.architecture.layout != "encoder-only")
        _is_decoder = is_decoder if is_decoder is not None else (self.architecture.layout != "encoder-only")

        _quantiles = quantiles
        if num_quantiles is not None and quantiles is None:
            _quantiles = np.linspace(0.5 / num_quantiles, 1 - 0.5 / num_quantiles, num_quantiles).tolist()
        elif quantiles is not None:
             if num_quantiles is not None and num_quantiles != len(quantiles):
                 raise ValueError(f"num_quantiles ({num_quantiles}) does not match len(quantiles) ({len(quantiles)}). Set one or the other.")
             num_quantiles = len(quantiles)
        self.num_quantiles = num_quantiles

        super().__init__(
            feature_size=input_dim,
            target_dim=_target_dim,
            context_length=context_length,
            prediction_length=prediction_length,
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            past_dynamic_dim=past_dynamic_dim,
            static_cardinalities=static_cardinalities,
            dynamic_cardinalities=dynamic_cardinalities,
            past_dynamic_cardinalities=past_dynamic_cardinalities,
            static_embedding_dim=static_embedding_dim,
            dynamic_embedding_dim=dynamic_embedding_dim,
            past_dynamic_embedding_dim=past_dynamic_embedding_dim,
            time_features=time_features,
            loss_config=loss_config,
            scaling=scaling,
            quantiles=_quantiles,
            autoregressive=_autoregressive,
            is_decoder=_is_decoder,
            **kwargs,
        )
        # Defer validation until all attributes are set
        # self.validate_config() # Call validate explicitly after init if needed

    def to_flat_dict(self) -> dict:
        return self.to_dict()

    def to_dict(self) -> dict:
        base = super().to_dict()
        hf_base_fields = {k: base.get(k) for k in ['_name_or_path', 'model_type', 'architectures'] if k in base}
        own = {
            # Use self.model_type defined in this class
            "model_type":                  self.model_type,
            "d_model":                     self.d_model,
            "hidden_dropout_prob":         self.hidden_dropout_prob,
            "max_position_embeddings":     self.max_position_embeddings,
            "architecture":                self.architecture.to_dict(),
            "value_embedding_config":      self.value_embedding_config.to_dict(),
            "positional_embedding_config": self.positional_embedding_config.to_dict(),
            "encoder_blocks":              [b.to_dict() for b in self.encoder_blocks] if self.encoder_blocks is not None else None,
            "decoder_blocks":              [b.to_dict() for b in self.decoder_blocks] if self.decoder_blocks is not None else None,
            "output_head_config":          self.output_head_config.to_dict(),
            "norm_config":                 self.norm_config.to_dict(),
            "head_agg_config":             self.head_agg_config.to_dict(),
            "output_attentions":           self.output_attentions,
            "output_hidden_states":        self.output_hidden_states,
            "use_teacher_forcing":         self.use_teacher_forcing,
            "quantizer_config":            self.quantizer_config.to_dict() if self.quantizer_config else None,
            "vocab_size":                  self.vocab_size,
            "decoder_start_token_id":      self.decoder_start_token_id,
            "num_quantiles":               self.num_quantiles,
        }
        # Merge base config, own config, ensuring HF base fields are prioritized if present
        # Also, remove keys from `base` that are redefined in `own` to avoid duplication if names differ slightly
        base_filtered = {k: v for k, v in base.items() if k not in own}
        final_dict = {**base_filtered, **own, **hf_base_fields}
        return final_dict

    def validate_config(self):
        super().validate_config()
        all_blocks = (self.encoder_blocks or []) + (self.decoder_blocks or [])
        for i, block in enumerate(all_blocks):
             if block.attention_config:
                  heads = block.attention_config.num_heads
                  assert self.d_model % heads == 0, \
                      f"d_model ({self.d_model}) must be divisible by num_heads ({heads}) in block {i}"
        if self.quantizer_config and not self.vocab_size:
             print("Warning: Quantizer configured but vocab_size not set.")
        if self.architecture.layout != "encoder-only" and self.decoder_start_token_id is None and self.vocab_size is not None:
             print("Warning: Decoder model needs decoder_start_token_id if tokenized.")
        if self.num_quantiles is not None and len(self.quantiles) != self.num_quantiles:
             raise ValueError(f"Length of quantiles ({len(self.quantiles)}) must match num_quantiles ({self.num_quantiles}).")


    @classmethod
    def from_dict(cls, d: dict):
        # Ensure nested dictionaries are converted to config objects
        # Use .get to handle potentially missing keys gracefully
        config_map = {
            "architecture": TransformerArchitectureConfig,
            "value_embedding_config": EmbeddingConfig,
            "positional_embedding_config": EmbeddingConfig,
            "output_head_config": OutputHeadConfig,
            "norm_config": NormalizationConfig,
            "head_agg_config": HeadAggregationConfig,
            "quantizer_config": QuantizerConfig,
        }
        for key, config_cls in config_map.items():
            config_dict = d.get(key)
            if isinstance(config_dict, dict):
                d[key] = config_cls.from_dict(config_dict)
            elif config_dict is None:
                 d[key] = None # Allow None values
            # Else: assume it's already the correct type or handled elsewhere

        # Handle lists of block configs
        for block_key in ["encoder_blocks", "decoder_blocks"]:
            block_list = d.get(block_key)
            if isinstance(block_list, list):
                d[block_key] = [TransformerBlockConfig.from_dict(item) if isinstance(item, dict) else item for item in block_list]

        # Handle potential legacy fields or mappings
        if "hidden_size" in d and "d_model" not in d:
            d["d_model"] = d.pop("hidden_size")
        if "feature_size" in d and "input_dim" not in d:
             d["input_dim"] = d["feature_size"]
        if "loss_type" in d and "loss_config" not in d:
             d["loss_config"] = {"type": d.pop("loss_type")}

        # Remove deprecated top-level keys before passing to __init__
        d.pop("attention_blocks", None)
        d.pop("feedforward_config", None)

        # Map feature_size from base config if input_dim is missing
        if "input_dim" not in d and "feature_size" in d:
            d["input_dim"] = d["feature_size"]
        elif "input_dim" not in d:
             raise ValueError("Missing required argument: input_dim (or feature_size for backward compatibility)")

        return cls(**d)

# Update attribute map dynamically (if needed for HF integration)
# Be cautious with modifying class attributes directly
# try:
#     _dummy_kwargs = {
#         "input_dim": 1, "context_length": 1, "prediction_length": 1,
#         "quantiles": [0.5], "num_quantiles": 1 # Ensure quantiles match num_quantiles
#     }
#     _dummy = TransformerTimeSeriesConfig(**_dummy_kwargs)
#     TransformerTimeSeriesConfig.attribute_map.update({k: k for k in _dummy.to_dict().keys()})
#     del _dummy, _dummy_kwargs
# except Exception as e:
#     print(f"Could not auto-update attribute_map: {e}")
