from temporal.configs.basetimeseriesconfig import BaseTimeSeriesConfig


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
        architecture (TransformerArchitectureConfig): Transformer architecture settings.
        attention_blocks (TransformerAttentionBlockConfig): Nested attention configurations.
        value_embedding_config (EmbeddingConfig): Config for value embeddings.
        positional_embedding_config (EmbeddingConfig): Config for positional embeddings.
        feedforward_config (FeedForwardConfig): Config for feed-forward networks.
        output_head_config (OutputHeadConfig): Config for output prediction head.
        block_configs (TransformerBlockConfig or list): Config(s) for transformer blocks.
        norm_config (NormalizationConfig): Config for normalization layers.
        head_agg_config (HeadAggregationConfig): Config for combining multiple heads.
        hidden_size (int): Dimensionality of hidden representations.
        output_token_lengths (int): Number of tokens produced by output head.
        num_quantiles (int): Number of quantiles for probabilistic forecasts.
        output_attentions (bool): If True, return attention weights in outputs.
        output_hidden_states (bool): If True, return all hidden states in outputs.
        use_teacher_forcing (bool): If True, apply teacher forcing during training.
        **kwargs: Additional arguments passed to BaseTimeSeriesConfig.
    """
    model_type = "transformer_time_series"

    def __init__(
        self,
        architecture=None,
        attention_blocks=None,
        value_embedding_config=None,
        positional_embedding_config=None,
        feedforward_config=None,
        output_head_config=None,
        block_configs=None,
        norm_config=None,
        head_agg_config=None,
        hidden_size=64,
        output_token_lengths=1,
        num_quantiles=3,
        output_attentions=False,
        output_hidden_states=False,
        use_teacher_forcing: bool = True,

        **kwargs
    ):
        super().__init__(**kwargs)

        self.architecture = architecture or TransformerArchitectureConfig()
        self.attention_blocks = attention_blocks or TransformerAttentionBlockConfig()
        self.feedforward_config = feedforward_config or FeedForwardConfig()
        self.norm_config = norm_config or NormalizationConfig()
        self.head_agg_config = head_agg_config or HeadAggregationConfig()
        self.block_configs = block_configs or TransformerBlockConfig()
        self.output_head_config = output_head_config or OutputHeadConfig()

        self.hidden_size = hidden_size
        self.output_token_lengths = output_token_lengths
        self.num_quantiles = num_quantiles
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_teacher_forcing = use_teacher_forcing

        self.value_embedding_config = value_embedding_config or EmbeddingConfig(type="value")
        self.positional_embedding_config = positional_embedding_config or EmbeddingConfig(type="positional_sinusoidal")

        self.value_embedding_type = self.value_embedding_config.type
        self.pos_embedding_type = self.positional_embedding_config.type

    def save_json(self, path):
        """
        Save the configuration to a JSON file.

        Args:
            path (str): File path to write JSON.
        """
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path):
        """
        Load configuration from a JSON file.

        Args:
            path (str): File path of JSON to read.

        Returns:
            TransformerTimeSeriesConfig: Loaded configuration instance.
        """
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_flat_dict(self):
        """
        Return a flattened dictionary of core configuration values.

        Returns:
            dict: Flat dictionary of key hyperparameters.
        """
        return {
            "hidden_size": self.hidden_size,
            "prediction_length": self.prediction_length,
            "context_length": self.context_length,
            "attention_type": self.attention_blocks.encoder_attention.attention_type,
            "num_attention_heads": self.attention_blocks.encoder_attention.num_heads,
        }

    @classmethod
    def from_flat_dict(cls, d):
        """
        Create configuration from a flattened dictionary.

        Args:
            d (dict): Dictionary containing keys 'hidden_size', 'prediction_length', 'context_length',
                      'attention_type', and 'num_attention_heads'.

        Returns:
            TransformerTimeSeriesConfig: New config instance.
        """
        return cls(
            hidden_size=d.get("hidden_size", 64),
            prediction_length=d.get("prediction_length", 12),
            context_length=d.get("context_length", 128),
            attention_blocks=TransformerAttentionBlockConfig(
                encoder_attention=AttentionConfig(
                    attention_type=d.get("attention_type", "full"),
                    num_heads=d.get("num_attention_heads", 4)
                )
            )
        )

    def to_dict(self):
        """
        Convert entire transformer time series configuration to a dictionary.

        Returns:
            dict: Dictionary representation of the full config.
        """
        base = super().to_dict()
        return {
            "architecture": self.architecture.to_dict(),
            "attention_blocks": self.attention_blocks.to_dict(),
            "value_embedding_config": self.value_embedding_config.to_dict(),
            "positional_embedding_config": self.positional_embedding_config.to_dict(),
            "feedforward_config": self.feedforward_config.to_dict(),
            "norm_config": self.norm_config.to_dict(),
            "head_agg_config": self.head_agg_config.to_dict(),
            "block_configs": self.block_configs.to_dict() if hasattr(self.block_configs, 'to_dict') else [b.to_dict() for b in self.block_configs],
            "output_head_config": self.output_head_config.to_dict(),
            "hidden_size": self.hidden_size,
            "output_token_lengths": self.output_token_lengths,
            "num_quantiles": self.num_quantiles,
            "output_attentions": self.output_attentions,
            "output_hidden_states": self.output_hidden_states,
            "use_teacher_forcing": self.use_teacher_forcing,
            **base
        }

    @classmethod
    def from_dict(cls, d):
        """
        Instantiate configuration from a dictionary.

        Args:
            d (dict): Dictionary matching the output of to_dict().

        Returns:
            TransformerTimeSeriesConfig: New config instance.
        """
        return cls(
            architecture=TransformerArchitectureConfig.from_dict(d.get("architecture", {})),
            attention_blocks=TransformerAttentionBlockConfig.from_dict(d.get("attention_blocks", {})),
            value_embedding_config=EmbeddingConfig.from_dict(d.get("value_embedding_config", {})),
            positional_embedding_config=EmbeddingConfig.from_dict(d.get("positional_embedding_config", {})),
            feedforward_config=FeedForwardConfig.from_dict(d.get("feedforward_config", {})),
            norm_config=NormalizationConfig.from_dict(d.get("norm_config", {})),
            head_agg_config=HeadAggregationConfig.from_dict(d.get("head_agg_config", {})),
            output_head_config=OutputHeadConfig.from_dict(d.get("output_head_config", {})),
            block_configs=[TransformerBlockConfig.from_dict(b) for b in d.get("block_configs", [])] if isinstance(d.get("block_configs"), list) else TransformerBlockConfig.from_dict(d.get("block_configs", {})),
            hidden_size=d.get("hidden_size", 64),
            output_token_lengths=d.get("output_token_lengths", 1),
            num_quantiles=d.get("num_quantiles", 3),
            output_attentions=d.get("output_attentions", False),
            output_hidden_states=d.get("output_hidden_states", False),
            use_teacher_forcing=d.get("use_teacher_forcing", True),
            **{k: v for k, v in d.items() if k not in {
                "architecture", "attention_blocks", "value_embedding_config", "positional_embedding_config",
                "feedforward_config", "norm_config", "head_agg_config", "output_head_config",
                "block_configs", "hidden_size", "output_token_lengths", "num_quantiles",
                "output_attentions", "output_hidden_states", "use_teacher_forcing"
            }}
        )

    def validate_config(self):
        """
        Validate transformer time series configuration for consistency and correctness.

        Raises:
            AssertionError: If any setting is invalid or inconsistent.
        """
        # Basic hyperparameter checks
        assert self.hidden_size > 0, "hidden_size must be > 0"
        assert self.output_token_lengths > 0, "output_token_lengths must be > 0"
        assert self.num_quantiles > 0, "num_quantiles must be > 0"
        assert self.output_head_config.output_size is not None, "output_head_config.output_size must be set"

        # Validate embedded configs
        if hasattr(self.value_embedding_config, 'validate'):
            self.value_embedding_config.validate()
        if hasattr(self.positional_embedding_config, 'validate'):
            self.positional_embedding_config.validate()

        # Validate feed-forward and attention blocks
        if hasattr(self.feedforward_config, 'validate'):
            self.feedforward_config.validate()
        for attn in [
            self.attention_blocks.encoder_attention,
            self.attention_blocks.decoder_attention,
            self.attention_blocks.decoder_cross_attention
        ]:
            if hasattr(attn, 'validate'):
                attn.validate()

        # Dimensional consistency
        enc_heads = self.attention_blocks.encoder_attention.num_heads
        assert self.hidden_size % enc_heads == 0, \
            "hidden_size must be divisible by encoder_attention.num_heads"
        dec_heads = self.attention_blocks.decoder_attention.num_heads
        assert self.hidden_size % dec_heads == 0, \
            "hidden_size must be divisible by decoder_attention.num_heads"

        # Normalization and aggregation
        if hasattr(self.norm_config, 'validate'):
            self.norm_config.validate()
        if self.output_token_lengths > 1 and hasattr(self.head_agg_config, 'validate'):
            self.head_agg_config.validate()

        # Layout and block count
        layout = self.architecture.layout
        assert layout in {"encoder", "decoder", "encoder-decoder"}, \
            f"Invalid architecture layout: {layout}"
        if layout in {"encoder", "encoder-decoder"}:
            assert self.architecture.num_encoder_layers > 0, "Must have at least one encoder layer"
        if layout in {"decoder", "encoder-decoder"}:
            assert self.architecture.num_decoder_layers > 0, "Must have at least one decoder layer"

        # Quantiles validation if probabilistic
        if getattr(self, 'quantiles', None):
            assert all(0 < q < 1 for q in self.quantiles), "All quantiles must be in (0, 1)"
