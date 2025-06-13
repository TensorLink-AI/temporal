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
                 bias=True,
                 use_rope=False, 
                 use_alibi=False, 
                 rope_base=10000, 
                 kwargs=None):
        assert isinstance(dropout, (float, int)) and 0.0 <= dropout <= 1.0, \
            f"dropout must be in [0, 1], got {dropout}"
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.bias = bias
        self.use_rope = use_rope
        self.use_alibi = use_alibi
        self.rope_base = rope_base
        self.kwargs = kwargs or {}

    def to_dict(self):
        """
        Convert attention configuration to a dictionary.
        """
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
        assert self.num_heads > 0, "num_heads must be > 0"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0, 1]"
        if self.use_rope and self.use_alibi:
            print("Warning: Both use_rope and use_alibi are set to True in AttentionConfig. Behavior might be undefined depending on implementation.")


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
        """
        return cls(
            type=d.get("type", "linear"),
            output_size=d.get("output_size"),
            kwargs=d.get("kwargs", {})
        )


class TransformerAttentionBlockConfig:
    """
    Configuration grouping attention configs for encoder and decoder blocks.

    Args:
        encoder_attention (AttentionConfig): Config for encoder self-attention.
        decoder_attention (AttentionConfig): Config for decoder self-attention.
        decoder_cross_attention (AttentionConfig): Config for decoder cross-attention.
    """
    def __init__(self, 
                 encoder_attention: Optional[AttentionConfig]=None, 
                 decoder_attention: Optional[AttentionConfig]=None, 
                 decoder_cross_attention: Optional[AttentionConfig]=None):
        self.encoder_attention = encoder_attention or AttentionConfig()
        self.decoder_attention = decoder_attention or AttentionConfig()
        self.decoder_cross_attention = decoder_cross_attention or AttentionConfig()

    def to_dict(self):
        """
        Convert attention block configuration to a dictionary.
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
        """
        return cls(
            encoder_attention=AttentionConfig.from_dict(d.get("encoder_attention", {})),
            decoder_attention=AttentionConfig.from_dict(d.get("decoder_attention", {})),
            decoder_cross_attention=AttentionConfig.from_dict(d.get("decoder_cross_attention", {}))
        )

class FeedForwardConfig:
    """
    Configuration for the feed-forward network sublayer in transformer blocks.
    Updated to support MoE parameters.
    """
    def __init__(self,
                 type="standard",
                 intermediate_size=256,
                 activation="gelu",
                 dropout=0.1,
                 bias=True, 
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
        """
        Convert feed-forward configuration to a dictionary.
        """
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
        """
        Create a FeedForwardConfig from a dictionary.
        """
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
        """
        Validate the feed-forward configuration.
        """
        if self.type == "moe":
            assert self.num_experts is not None and self.num_experts > 0, "num_experts must be > 0 for MoE"
            assert self.top_k is not None and 0 < self.top_k <= self.num_experts, "top_k must be > 0 and <= num_experts"
        pass

class EmbeddingConfig:
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
        pass

class HeadAggregationConfig:
    """
    Configuration for combining multiple output heads.
    """
    def __init__(self, type="mean", kwargs=None):
        self.type = type
        self.kwargs = kwargs or {}

    def to_dict(self):
        """
        Convert head aggregation configuration to a dictionary.
        """
        return {
            "type": self.type,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create a HeadAggregationConfig from a dictionary.
        """
        return cls(**d)

class NormalizationConfig:
    """
    Configuration for normalization layers.
    """
    def __init__(self, norm_type="layer", eps=1e-5, kwargs: Optional[Dict[str, Any]] = None):
        self.norm_type = norm_type
        self.eps = eps
        self.kwargs = kwargs or {}

    def to_dict(self):
        """
        Convert normalization configuration to a dictionary.
        """
        return {
            "norm_type": self.norm_type,
            "eps": self.eps,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create a NormalizationConfig from a dictionary.
        """
        return cls(**d)

class QuantizerConfig:
    """
    Configuration for time series quantization.
    """
    def __init__(self, quantization_type="mean_std_bins", vocab_size=4096, num_features=1, **kwargs):
        self.quantization_type = quantization_type
        self.vocab_size = vocab_size
        self.num_features = num_features
        self.kwargs = kwargs

    def to_dict(self):
        """
        Convert quantizer configuration to a dictionary.
        """
        return {
            "quantization_type": self.quantization_type,
            "vocab_size": self.vocab_size,
            "num_features": self.num_features,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create a QuantizerConfig from a dictionary.
        """
        return cls(**d)

class TransformerBlockConfig:
    """
    Configuration for a generic transformer block.

    Args:
        block_type (str): The type of transformer block (e.g., 'default_encoder', 'default_decoder').
        attention_config (Optional[AttentionConfig]): Configuration for the self-attention mechanism.
                                                    If None, defaults to AttentionConfig().
        cross_attention_config (Optional[AttentionConfig]): Configuration for the cross-attention mechanism.
                                                         Only used if the block_type supports cross-attention
                                                         (e.g., in decoder blocks of an encoder-decoder architecture).
                                                         If None, and cross-attention is needed, it may fall back
                                                         to attention_config. Defaults to None.
        ffn_config (Optional[FeedForwardConfig]): Configuration for the feed-forward network.
                                               If None, defaults to FeedForwardConfig().
        norm_config (Optional[NormalizationConfig]): Configuration for normalization layers.
                                                  If None, defaults to NormalizationConfig().
        kwargs (Optional[dict]): Additional keyword arguments for the block implementation.
    """
    def __init__(self, 
                 block_type: str ="default_encoder", 
                 attention_config: Optional[AttentionConfig]=None, 
                 cross_attention_config: Optional[AttentionConfig]=None, # ADDED
                 ffn_config: Optional[FeedForwardConfig]=None, 
                 norm_config: Optional[NormalizationConfig]=None, 
                 kwargs: Optional[Dict[str, Any]]=None):
        self.block_type = block_type
        self.attention_config = attention_config or AttentionConfig()
        self.cross_attention_config = cross_attention_config # Defaults to None, decoder layer will handle fallback
        self.ffn_config = ffn_config or FeedForwardConfig()
        self.norm_config = norm_config or NormalizationConfig()
        self.kwargs = kwargs or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert block configuration to a dictionary.
        """
        return {
            "block_type": self.block_type,
            "attention_config": self.attention_config.to_dict(),
            "cross_attention_config": self.cross_attention_config.to_dict() if self.cross_attention_config else None,
            "ffn_config": self.ffn_config.to_dict(),
            "norm_config": self.norm_config.to_dict(),
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TransformerBlockConfig':
        """
        Create a TransformerBlockConfig from a dictionary.
        """
        cross_attn_cfg_dict = d.get("cross_attention_config")
        return cls(
            block_type=d.get("block_type", "default_encoder"),
            attention_config=AttentionConfig.from_dict(d.get("attention_config", {})),
            cross_attention_config=AttentionConfig.from_dict(cross_attn_cfg_dict) if cross_attn_cfg_dict else None,
            ffn_config=FeedForwardConfig.from_dict(d.get("ffn_config", {})),
            norm_config=NormalizationConfig.from_dict(d.get("norm_config", {})),
            kwargs=d.get("kwargs", {})
        )

class LossConfig:
    """
    Configuration for the loss function.

    Args:
        type (str): Type of loss function (e.g., 'mse', 'crps').
        kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the loss function.
    """
    def __init__(self, type: str = "mse", kwargs: Optional[Dict[str, Any]] = None):
        self.type = type
        self.kwargs = kwargs or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert loss configuration to a dictionary.
        """
        return {
            "type": self.type,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LossConfig':
        """
        Create a LossConfig from a dictionary.
        """
        return cls(
            type=d.get("type", "mse"),
            kwargs=d.get("kwargs", {})
        )

class TransformerTimeSeriesConfig(BaseTimeSeriesConfig):
    """
    Configuration for a transformer-based time-series forecasting model.
    Extends ``BaseTimeSeriesConfig`` with transformer-specific options.
    """

    attribute_map = BaseTimeSeriesConfig.attribute_map.copy()

    def __init__(
        self,
        input_dim: int,
        context_length: int,
        prediction_length: int,
        output_dim: Optional[int] = None,
        static_dim: int = 0,
        dynamic_dim: int = 0,
        past_dynamic_dim: int = 0,
        static_cardinalities: Optional[List[int]] = None,
        dynamic_cardinalities: Optional[List[int]] = None,
        past_dynamic_cardinalities: Optional[List[int]] = None,
        static_embedding_dim: Optional[List[int]] = None,
        dynamic_embedding_dim: Optional[List[int]] = None,
        past_dynamic_embedding_dim: Optional[List[int]] = None,
        time_features: Optional[List[str]] = None,
        loss_config: Optional[LossConfig] = None, 
        scaling: bool = True,
        model_type: str = "transformer",
        d_model: int = 64,
        hidden_dropout_prob: float = 0.1,
        max_position_embeddings: int = 4096,
        architecture: Optional[TransformerArchitectureConfig] = None,
        value_embedding_config: Optional[EmbeddingConfig] = None,
        positional_embedding_config: Optional[EmbeddingConfig] = None,
        encoder_blocks: Optional[List[TransformerBlockConfig]] = None,
        decoder_blocks: Optional[List[TransformerBlockConfig]] = None,
        output_head_config: Optional[OutputHeadConfig] = None,
        norm_config: Optional[NormalizationConfig] = None,
        head_agg_config: Optional[HeadAggregationConfig] = None,
        quantizer_config: Optional[QuantizerConfig] = None,
        vocab_size: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        num_quantiles: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_teacher_forcing: bool = True,
        attention_blocks: Optional[Any] = None, # Deprecated
        feedforward_config: Optional[Any] = None, # Deprecated
        # feature_size is removed from here, will be handled via input_dim and kwargs
        autoregressive: Optional[bool] = None,
        is_decoder: Optional[bool] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ):
        if attention_blocks is not None:
            print("Warning: `attention_blocks` config key is deprecated. Configure attention within `encoder_blocks`/`decoder_blocks`.")
        if feedforward_config is not None:
            print("Warning: `feedforward_config` key is deprecated. Configure FFN within `encoder_blocks`/`decoder_blocks`.")

        # Pop feature_size and target_dim from kwargs if they exist, to prevent multiple values error
        # for BaseTimeSeriesConfig.__init__
        feature_size_from_kwargs = kwargs.pop('feature_size', None)
        target_dim_from_kwargs = kwargs.pop('target_dim', None)

        # Determine the feature_size to be passed to the superclass.
        # TransformerTimeSeriesConfig uses 'input_dim' for this concept.
        _input_dim_resolved = input_dim
        if feature_size_from_kwargs is not None and feature_size_from_kwargs != _input_dim_resolved:
            print(
                f"Warning: 'feature_size' ({feature_size_from_kwargs}) found in kwargs "
                f"differs from 'input_dim' ({_input_dim_resolved}). "
                f"Using 'input_dim' ({_input_dim_resolved}) for BaseTimeSeriesConfig's 'feature_size'."
            )
        
        # Determine the target_dim to be passed to the superclass.
        # TransformerTimeSeriesConfig derives this from 'output_dim' or 'input_dim'.
        _target_dim_resolved = output_dim if output_dim is not None else input_dim
        if target_dim_from_kwargs is not None and target_dim_from_kwargs != _target_dim_resolved:
            print(
                f"Warning: 'target_dim' ({target_dim_from_kwargs}) found in kwargs "
                f"differs from the derived target dimension ({_target_dim_resolved}). "
                f"Using the derived value ({_target_dim_resolved}) for BaseTimeSeriesConfig's 'target_dim'."
            )

        self.model_type = model_type
        self.d_model = d_model
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.architecture = architecture or TransformerArchitectureConfig()
        # Ensure value_embedding_config uses the resolved input_dim if it depends on feature_size
        self.value_embedding_config = value_embedding_config or EmbeddingConfig(type="value", kwargs={"feature_size": _input_dim_resolved, "d_model": d_model})
        self.positional_embedding_config = positional_embedding_config or EmbeddingConfig(type="sinusoidal", kwargs={"max_seq_len": max_position_embeddings, "d_model": d_model})
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        # Ensure output_head_config uses the resolved target_dim for its output_size
        self.output_head_config = output_head_config or OutputHeadConfig(type="linear", output_size=_target_dim_resolved)
        self.norm_config = norm_config or NormalizationConfig()
        self.head_agg_config = head_agg_config or HeadAggregationConfig()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_teacher_forcing = use_teacher_forcing
        self.use_cache          = use_cache

        self.quantizer_config = quantizer_config
        self.vocab_size = vocab_size
        self.decoder_start_token_id = decoder_start_token_id
        
        self.loss_config = loss_config or LossConfig()

        _autoregressive = autoregressive if autoregressive is not None else (self.architecture.layout != "encoder-only") # Check original logic: != "encoder" or != "encoder-only"
        _is_decoder = is_decoder if is_decoder is not None else (self.architecture.layout != "encoder-only") # Check original logic

        _quantiles = quantiles
        if num_quantiles is not None and quantiles is None:
            _quantiles = np.linspace(0.5 / num_quantiles, 1 - 0.5 / num_quantiles, num_quantiles).tolist()
        elif quantiles is not None:
             if num_quantiles is not None and num_quantiles != len(quantiles):
                 raise ValueError(f"num_quantiles ({num_quantiles}) does not match len(quantiles) ({len(quantiles)}). Set one or the other.")
             num_quantiles = len(quantiles)
        self.num_quantiles = num_quantiles
        
        # Pass resolved values to superclass
        super().__init__(
            feature_size=_input_dim_resolved, # Use the resolved input_dim
            target_dim=_target_dim_resolved,   # Use the resolved target_dim
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
            loss_config=self.loss_config.to_dict(),
            scaling=scaling,
            quantiles=_quantiles,
            autoregressive=_autoregressive,
            is_decoder=_is_decoder,
            **kwargs, # Pass remaining kwargs
        )

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert the entire configuration to a flat dictionary suitable for some integrations.
        """
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary, serializing nested config objects.
        """
        base = super().to_dict()
        hf_base_fields = {k: base.get(k) for k in ['_name_or_path', 'model_type', 'architectures'] if k in base}
        own = {
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
            "loss_config":                 self.loss_config, # MODIFIED
            "output_attentions":           self.output_attentions,
            "output_hidden_states":        self.output_hidden_states,
            "use_teacher_forcing":         self.use_teacher_forcing,
            "quantizer_config":            self.quantizer_config.to_dict() if self.quantizer_config else None,
            "vocab_size":                  self.vocab_size,
            "decoder_start_token_id":      self.decoder_start_token_id,
            "num_quantiles":               self.num_quantiles,
        }
        base_filtered = {k: v for k, v in base.items() if k not in own}
        # Ensure loss_config from own takes precedence if it was originally a dict in base
        if 'loss_config' in base_filtered and isinstance(base_filtered['loss_config'], dict) and isinstance(own['loss_config'], dict):
             pass # own['loss_config'] will overwrite
        
        final_dict = {**base_filtered, **own, **hf_base_fields}
        return final_dict

    def validate_config(self):
        """
        Validate the overall transformer time series configuration.
        """
        super().validate_config()
        all_blocks = (self.encoder_blocks or []) + (self.decoder_blocks or [])
        for i, block_config in enumerate(all_blocks):
             # Validate self-attention config
             if block_config.attention_config:
                  heads = block_config.attention_config.num_heads
                  assert self.d_model % heads == 0, \
                      f"d_model ({self.d_model}) must be divisible by num_heads ({heads}) in self-attention of block {i}"
             
             # Validate cross-attention config if present (and if it's a decoder block, implicitly)
             if hasattr(block_config, 'cross_attention_config') and block_config.cross_attention_config:
                  cross_heads = block_config.cross_attention_config.num_heads
                  assert self.d_model % cross_heads == 0, \
                      f"d_model ({self.d_model}) must be divisible by num_heads ({cross_heads}) in cross-attention of block {i}"

        if self.quantizer_config and not self.vocab_size:
             print("Warning: Quantizer configured but vocab_size not set.")
        if self.architecture.layout != "encoder-only" and self.decoder_start_token_id is None and self.vocab_size is not None:
             print("Warning: Decoder model needs decoder_start_token_id if tokenized.")
        if self.num_quantiles is not None and self.quantiles is not None and len(self.quantiles) != self.num_quantiles:
             raise ValueError(f"Length of quantiles ({len(self.quantiles)}) must match num_quantiles ({self.num_quantiles}).")


    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TransformerTimeSeriesConfig':
        """
        Create a TransformerTimeSeriesConfig from a dictionary, handling nested configs.
        """
        config_map = {
            "architecture": TransformerArchitectureConfig,
            "value_embedding_config": EmbeddingConfig,
            "positional_embedding_config": EmbeddingConfig,
            "output_head_config": OutputHeadConfig,
            "norm_config": NormalizationConfig,
            "head_agg_config": HeadAggregationConfig,
            "quantizer_config": QuantizerConfig,
            "loss_config": LossConfig,
        }
        for key, config_cls in config_map.items():
            config_dict = d.get(key)
            if isinstance(config_dict, dict):
                if key == "loss_config" and isinstance(config_dict, LossConfig):
                    d[key] = config_dict
                else:
                    d[key] = config_cls.from_dict(config_dict)
            elif config_dict is None and key == "loss_config": 
                 d[key] = LossConfig()
            # No 'else: d[key]=None' needed as .get already returns None if not found

        for block_key in ["encoder_blocks", "decoder_blocks"]:
            block_list = d.get(block_key)
            if isinstance(block_list, list):
                d[block_key] = [TransformerBlockConfig.from_dict(item) if isinstance(item, dict) else item for item in block_list]

        if "hidden_size" in d and "d_model" not in d:
            d["d_model"] = d.pop("hidden_size")
        
        # Handle feature_size/input_dim carefully for backward compatibility
        # If input_dim is present, it takes precedence.
        # If only feature_size is present (from old config), use it for input_dim.
        if "input_dim" not in d and "feature_size" in d:
             d["input_dim"] = d.pop("feature_size")
        elif "feature_size" in d and "input_dim" in d and d["feature_size"] != d["input_dim"]:
            print(f"Warning: Both 'input_dim' ({d['input_dim']}) and 'feature_size' ({d['feature_size']}) found in config dict. "
                  f"Preferring 'input_dim'. 'feature_size' will be ignored.")
            d.pop("feature_size") # Remove to avoid confusion
        elif "feature_size" in d: # If they were equal or only feature_size existed before mapping
            d.pop("feature_size")


        if "loss_type" in d and "loss_config" not in d:
             d["loss_config"] = LossConfig(type=d.pop("loss_type"))
        elif "loss_type" in d and "loss_config" in d and isinstance(d["loss_config"], LossConfig) and d["loss_config"].type is None: # If LossConfig obj exists with no type
            d["loss_config"].type = d.pop("loss_type")
        elif "loss_type" in d and "loss_config" in d and isinstance(d["loss_config"], dict) and "type" not in d["loss_config"]:
            d["loss_config"]["type"] = d.pop("loss_type")
        elif "loss_type" in d: 
            d.pop("loss_type")


        d.pop("attention_blocks", None) 
        d.pop("feedforward_config", None)

        if "input_dim" not in d:
             # This case should ideally not be hit if feature_size was correctly mapped
             raise ValueError("Missing required argument: input_dim")

        return cls(**d)