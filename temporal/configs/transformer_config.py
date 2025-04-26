from temporal.configs.basetimeseriesconfig import BaseTimeSeriesConfig
from transformers import PretrainedConfig
from typing import Optional, Union, List


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
        attention_type (str): Type of attention (e.g., 'full', 'local', 'flash', 'diffwist', 'hybrid').
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability for attention weights.
        kwargs (dict): Additional keyword arguments for attention implementation.
    """
    def __init__(self, attention_type="full", num_heads=4, dropout=0.1, kwargs=None):
        assert isinstance(dropout, (float, int)) and 0.0 <= dropout <= 1.0, \
            f"dropout must be in [0, 1], got {dropout}"
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.kwargs = kwargs or {}

    def to_dict(self):
        """
        Convert attention configuration to a dictionary.

        Returns:
            dict: Dictionary of attention settings.
        """
        return {
            "attention_type": self.attention_type,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create an AttentionConfig from a dictionary.

        Args:
            d (dict): Dictionary with keys matching __init__ parameters.

        Returns:
            AttentionConfig: New instance.
        """
        return cls(
            attention_type=d.get("attention_type", "full"),
            num_heads=d.get("num_heads", 4),
            dropout=d.get("dropout", 0.1),
            kwargs=d.get("kwargs", {})
        )

    def validate(self):
        """
        Validate the attention configuration for correctness.

        Raises:
            AssertionError: If any configuration value is invalid.
        """
        assert self.attention_type in {"full", "local", "flash", "diffwist", "hybrid"}, \
            f"Unknown attention_type: {self.attention_type}"
        assert self.num_heads > 0, "num_heads must be > 0"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0, 1]"


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


class TransformerAttentionBlockConfig:
    """
    Configuration grouping attention configs for encoder and decoder blocks.

    Args:
        encoder_attention (AttentionConfig): Config for encoder self-attention.
        decoder_attention (AttentionConfig): Config for decoder self-attention.
        decoder_cross_attention (AttentionConfig): Config for decoder cross-attention.
    """
    def __init__(self, encoder_attention=None, decoder_attention=None, decoder_cross_attention=None):
        default = AttentionConfig()
        self.encoder_attention = encoder_attention or default
        self.decoder_attention = decoder_attention or default
        self.decoder_cross_attention = decoder_cross_attention or default

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


class FeedForwardConfig:
    """
    Configuration for the feed-forward network sublayer in transformer blocks.

    Args:
        type (str): Type of feed-forward ('standard', 'glu', 'linear').
        intermediate_size (int): Size of the intermediate layer.
        activation (str): Activation function ('gelu', 'relu', 'swish', 'sigmoid').
        dropout (float): Dropout probability.
        kwargs (dict): Additional arguments for the feed-forward implementation.
    """
    def __init__(self, type="standard", intermediate_size=256, activation="gelu", dropout=0.1, kwargs=None):
        self.type = type
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.dropout = dropout
        self.kwargs = kwargs or {}

    def to_dict(self):
        """
        Convert feed-forward configuration to a dictionary.

        Returns:
            dict: Dictionary of feed-forward settings.
        """
        return {
            "type": self.type,
            "intermediate_size": self.intermediate_size,
            "activation": self.activation,
            "dropout": self.dropout,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create a FeedForwardConfig from a dictionary.

        Args:
            d (dict): Dictionary with keys matching __init__ parameters.

        Returns:
            FeedForwardConfig: New instance.
        """
        return cls(
            type=d.get("type", "standard"),
            intermediate_size=d.get("intermediate_size", 256),
            activation=d.get("activation", "gelu"),
            dropout=d.get("dropout", 0.1),
            kwargs=d.get("kwargs", {})
        )

    def validate(self):
        """
        Validate the feed-forward configuration.

        Raises:
            AssertionError: If any configuration value is invalid.
        """
        assert self.type in {"standard", "glu", "linear"}, f"Invalid FFN type: {self.type}"
        assert self.intermediate_size > 0, "intermediate_size must be > 0"
        assert self.activation in {"gelu", "relu", "swish", "sigmoid"}, f"Unsupported activation: {self.activation}"


class EmbeddingConfig:
    """
    Configuration for embedding layers in the transformer.

    Args:
        type (str): Type of embedding ('value', 'patch', 'positional_sinusoidal', 'learned').
        dropout (float): Dropout probability applied to embeddings.
        embedding_dim (int): Dimensionality of positional embeddings (if applicable).
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

        Returns:
            dict: Dictionary of embedding settings.
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

        Args:
            d (dict): Dictionary with keys matching __init__ parameters.

        Returns:
            EmbeddingConfig: New instance.
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

        Raises:
            AssertionError: If any configuration value is invalid.
        """
        assert self.type in {"value", "patch", "positional_sinusoidal", "learned"}, \
            f"Invalid embedding type: {self.type}"
        if "positional" in self.type:
            assert self.embedding_dim is not None, "Positional embedding must set embedding_dim"


class HeadAggregationConfig:
    """
    Configuration for combining multiple output heads.

    Args:
        type (str): Aggregation strategy (e.g., 'mean').
        kwargs (dict): Additional arguments for aggregation.
    """
    def __init__(self, type="mean", kwargs=None):
        self.type = type
        self.kwargs = kwargs or {}

    def to_dict(self):
        """
        Convert head aggregation configuration to a dictionary.

        Returns:
            dict: Dictionary of aggregation settings.
        """
        return {
            "type": self.type,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create a HeadAggregationConfig from a dictionary.

        Args:
            d (dict): Dictionary with keys matching __init__ parameters.

        Returns:
            HeadAggregationConfig: New instance.
        """
        return cls(
            type=d.get("type", "mean"),
            kwargs=d.get("kwargs", {})
        )


class NormalizationConfig:
    """
    Configuration for normalization layers.

    Args:
        norm_type (str): Type of normalization ('layer').
        eps (float): Small epsilon value for numerical stability.
    """
    def __init__(self, norm_type="layer", eps=1e-5):
        self.norm_type = norm_type
        self.eps = eps

    def to_dict(self):
        """
        Convert normalization configuration to a dictionary.

        Returns:
            dict: Dictionary of normalization settings.
        """
        return {
            "norm_type": self.norm_type,
            "eps": self.eps
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create a NormalizationConfig from a dictionary.

        Args:
            d (dict): Dictionary with keys matching __init__ parameters.

        Returns:
            NormalizationConfig: New instance.
        """
        return cls(
            norm_type=d.get("norm_type", "layer"),
            eps=d.get("eps", 1e-5)
        )


class TransformerBlockConfig:
    """
    Configuration for a generic transformer block.

    Args:
        block_type (str): Identifier for block type (e.g., 'standard').
        attention_config (AttentionConfig): Configuration for attention sublayer.
        ffn_config (FeedForwardConfig): Configuration for feed-forward sublayer.
        kwargs (dict): Additional arguments for the block.
    """
    def __init__(self, block_type="standard", attention_config=None, ffn_config=None, kwargs=None):
        self.block_type = block_type
        self.attention_config = attention_config
        self.ffn_config = ffn_config
        self.kwargs = kwargs or {}

    def to_dict(self):
        """
        Convert transformer block configuration to a dictionary.

        Returns:
            dict: Dictionary of block settings.
        """
        return {
            "block_type": self.block_type,
            "attention_config": self.attention_config.to_dict() if self.attention_config else None,
            "ffn_config": self.ffn_config.to_dict() if self.ffn_config else None,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create a TransformerBlockConfig from a dictionary.

        Args:
            d (dict): Dictionary with keys matching __init__ parameters.

        Returns:
            TransformerBlockConfig: New instance.
        """
        return cls(
            block_type=d.get("block_type", "standard"),
            attention_config=AttentionConfig.from_dict(d["attention_config"]) if d.get("attention_config") else None,
            ffn_config=FeedForwardConfig.from_dict(d["ffn_config"]) if d.get("ffn_config") else None,
            kwargs=d.get("kwargs", {})
        )

class TransformerTimeSeriesConfig(BaseTimeSeriesConfig):
    """
    Configuration for a transformer-based time series forecasting model.

    Extends BaseTimeSeriesConfig with transformer-specific options.

    Args:
        # BaseTimeSeriesConfig args:
        feature_size (int): Number of input features per time step.
        context_length (int): Length of the historical context window.
        prediction_length (int): Forecast horizon length.
        quantiles (List[float]): Quantile levels for probabilistic forecasting.
        output_token_lengths (int): Number of output tokens per prediction step.
        loss_type (str): Loss function type; one of PROBABILISTIC_LOSSES or a point-wise loss.
        use_dynamic_features (bool): Whether to include dynamic covariates.
        use_static_features (bool): Whether to include static covariates.
        autoregressive (bool): If True, model predicts autoregressively.
        is_decoder (bool): If True, configures model for decoder-only use.

        # Transformer-specific args:
        architecture: Optional[TransformerArchitectureConfig]       = None,
        attention_blocks: Optional[TransformerAttentionBlockConfig] = None,
        value_embedding_config: Optional[EmbeddingConfig]           = None,
        positional_embedding_config: Optional[EmbeddingConfig]      = None,
        feedforward_config: Optional[FeedForwardConfig]             = None,
        output_head_config: Optional[OutputHeadConfig]              = None,
        encoder_blocks: Optional[List[TransformerBlockConfig]] = None,
        decoder_blocks: Optional[List[TransformerBlockConfig]] = None,
        norm_config: Optional[NormalizationConfig]                  = None,
        head_agg_config: Optional[HeadAggregationConfig]            = None,
        hidden_size: int                  = 64,
        num_quantiles: int                = 3,
        output_attentions: bool           = False,
        output_hidden_states: bool        = False,
        use_teacher_forcing: bool         = True,
        hidden_dropout_prob: float        = 0.1,

        **kwargs
    ):
        """
        Initialize a TransformerTimeSeriesConfig.

        All base fields are initialized via super(), then transformer-specific
        sub-configs and hyperparameters are set.
        """
        # Assign transformer-specific attributes *before* calling super().__init__
        self.architecture               = architecture or TransformerArchitectureConfig()
        self.attention_blocks           = attention_blocks or TransformerAttentionBlockConfig()
        self.value_embedding_config     = value_embedding_config or EmbeddingConfig(type="value")
        self.positional_embedding_config = positional_embedding_config or EmbeddingConfig(type="positional_sinusoidal")
        self.feedforward_config         = feedforward_config or FeedForwardConfig()
        self.output_head_config         = output_head_config or OutputHeadConfig()
        # Handle encoder_blocks and decoder_blocks
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.norm_config                = norm_config or NormalizationConfig()
        self.head_agg_config            = head_agg_config or HeadAggregationConfig()

        self.hidden_size               = hidden_size
        self.num_quantiles             = num_quantiles
        self.output_attentions         = output_attentions
        self.output_hidden_states      = output_hidden_states
        self.use_teacher_forcing       = use_teacher_forcing
        self.hidden_dropout_prob = hidden_dropout_prob

        # Initialize BaseTimeSeriesConfig fields
        super().__init__(
            feature_size=feature_size,
            context_length=context_length,
            prediction_length=prediction_length,
            quantiles=quantiles,
            output_token_lengths=output_token_lengths,
            loss_type=loss_type,
            use_dynamic_features=use_dynamic_features,
            use_static_features=use_static_features,
            autoregressive=autoregressive,
            is_decoder=is_decoder,
            **kwargs
        )

        # final consistency check
        self.validate_config()

    def to_dict(self) -> dict:
        """
        Convert the full config (base + transformer-specific) to a dict.

        Returns:
            dict: A mapping of all config fields for serialization.
        """
        base = super().to_dict()
        own = {
            "architecture":                self.architecture.to_dict(),
            "attention_blocks":            self.attention_blocks.to_dict(),
            "value_embedding_config":      self.value_embedding_config.to_dict(),
            "positional_embedding_config": self.positional_embedding_config.to_dict(),
            "feedforward_config":          self.feedforward_config.to_dict(),
            "output_head_config":          self.output_head_config.to_dict(),
            "encoder_blocks":              [b.to_dict() for b in self.encoder_blocks] if self.encoder_blocks is not None else None,
            "decoder_blocks":              [b.to_dict() for b in self.decoder_blocks] if self.decoder_blocks is not None else None,
            "norm_config":                 self.norm_config.to_dict(),
            "head_agg_config":             self.head_agg_config.to_dict(),
            "hidden_size":                 self.hidden_size,
            "num_quantiles":               self.num_quantiles,
            "output_attentions":           self.output_attentions,
            "output_hidden_states":        self.output_hidden_states,
            "use_teacher_forcing":         self.use_teacher_forcing,
            "hidden_dropout_prob":         self.hidden_dropout_prob,
        }
        # Ensure all keys from the explicit map are included if they exist as attributes
        mapped_keys = {k: getattr(self, k).to_dict() if hasattr(getattr(self, k, None), 'to_dict') else getattr(self, k, None)
                       for k in self.attribute_map if hasattr(self, k) and k not in base}

        return {**base, **own, **mapped_keys}

    def validate_config(self):
        """
        Validate transformer-specific and base configuration for correctness.

        Raises:
            AssertionError: If any setting is invalid or inconsistent.
        """
        # base checks (e.g. context_length, quantiles)  
        super().validate_config()

        # hidden-size / heads divisibility
        if self.attention_blocks and self.attention_blocks.encoder_attention:
            enc_heads = self.attention_blocks.encoder_attention.num_heads
            assert self.hidden_size % enc_heads == 0, \
                "hidden_size must be divisible by encoder_attention.num_heads"
        if self.attention_blocks and self.attention_blocks.decoder_attention:
            dec_heads = self.attention_blocks.decoder_attention.num_heads
            assert self.hidden_size % dec_heads == 0, \
                "hidden_size must be divisible by decoder_attention.num_heads"

        # quantiles length vs declared
        assert self.num_quantiles > 0, "num_quantiles must be > 0"
        assert len(self.quantiles) == self.num_quantiles, \
            "len(quantiles) must equal num_quantiles"

        # output settings
        assert self.output_token_lengths > 0, "output_token_lengths must be > 0"


    @classmethod
    def from_dict(cls, d: dict):
        """
        Instantiate from a dictionary (e.g. loaded JSON).

        Args:
            d (dict): Dict matching to_dict() output.

        Returns:
            TransformerTimeSeriesConfig
        """
        # Ensure nested dictionaries are converted to config objects
        for key, config_cls in [
            ("architecture", TransformerArchitectureConfig),
            ("attention_blocks", TransformerAttentionBlockConfig),
            ("value_embedding_config", EmbeddingConfig),
            ("positional_embedding_config", EmbeddingConfig),
            ("feedforward_config", FeedForwardConfig),
            ("output_head_config", OutputHeadConfig),
            ("norm_config", NormalizationConfig),
            ("head_agg_config", HeadAggregationConfig),
        ]:
            if key in d and isinstance(d[key], dict):
                d[key] = config_cls.from_dict(d[key])

        # Handle block_configs, encoder_blocks, decoder_blocks separately
        if "block_configs" in d and isinstance(d["block_configs"], dict):
             d["block_configs"] = TransformerBlockConfig.from_dict(d["block_configs"])
        elif "block_configs" in d and isinstance(d["block_configs"], list):
            d["block_configs"] = [TransformerBlockConfig.from_dict(item) for item in d["block_configs"]]

        if "encoder_blocks" in d and isinstance(d["encoder_blocks"], list):
             d["encoder_blocks"] = [TransformerBlockConfig.from_dict(item) for item in d["encoder_blocks"]]

        if "decoder_blocks" in d and isinstance(d["decoder_blocks"], list):
             d["decoder_blocks"] = [TransformerBlockConfig.from_dict(item) for item in d["decoder_blocks"]]

        return cls(**d)


# ----------------------------------------------------------------
# COMMENTED OUT: AUTO-REGISTER transformer-specific keys in attribute_map
# ----------------------------------------------------------------
# _dummy = TransformerTimeSeriesConfig(
#     feature_size=1,
#     context_length=1,
#     prediction_length=1,
#     _name_or_path="transformer_time_series"
# )
# TransformerTimeSeriesConfig.attribute_map.update({k: k for k in _dummy.to_dict().keys()})
# del _dummy
