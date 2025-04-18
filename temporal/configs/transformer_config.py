from temporal.configs.basetimeseriesconfig import BaseTimeSeriesConfig
class TransformerArchitectureConfig:
    def __init__(self, layout="encoder-decoder", num_encoder_layers=4, num_decoder_layers=2, share_weights=False):
        assert layout in ("encoder", "decoder", "encoder-decoder")
        self.layout = layout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.share_weights = share_weights

    def to_dict(self):
        return {
            "layout": self.layout,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "share_weights": self.share_weights
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class AttentionConfig:
    def __init__(self, attention_type="full", num_heads=4, dropout=0.1, kwargs=None):
        assert isinstance(dropout, (float, int)) and 0.0 <= dropout <= 1.0
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.kwargs = kwargs or {}

    def to_dict(self):
        return {
            "attention_type": self.attention_type,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            attention_type=d.get("attention_type", "full"),
            num_heads=d.get("num_heads", 4),
            dropout=d.get("dropout", 0.1),
            kwargs=d.get("kwargs", {})
        )
    def validate(self):
        assert self.attention_type in {"full", "local", "flash", "diffwist", "hybrid"}, f"Unknown attention_type: {self.attention_type}"
        assert self.num_heads > 0, "num_heads must be > 0"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0, 1]"

class OutputHeadConfig:
    def __init__(self, type="linear", output_size=None, kwargs=None):
        self.type = type
        self.output_size = output_size  # ✅ Now configurable
        self.kwargs = kwargs or {}

    def to_dict(self):
        return {
            "type": self.type,
            "output_size": self.output_size,
            "kwargs": self.kwargs,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            type=d.get("type", "linear"),
            output_size=d.get("output_size"),
            kwargs=d.get("kwargs", {}),
        )

class TransformerAttentionBlockConfig:
    def __init__(self, encoder_attention=None, decoder_attention=None, decoder_cross_attention=None):
        default = AttentionConfig()
        self.encoder_attention = encoder_attention or default
        self.decoder_attention = decoder_attention or default
        self.decoder_cross_attention = decoder_cross_attention or default

    def to_dict(self):
        return {
            "encoder_attention": self.encoder_attention.to_dict(),
            "decoder_attention": self.decoder_attention.to_dict(),
            "decoder_cross_attention": self.decoder_cross_attention.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            encoder_attention=AttentionConfig.from_dict(d.get("encoder_attention", {})),
            decoder_attention=AttentionConfig.from_dict(d.get("decoder_attention", {})),
            decoder_cross_attention=AttentionConfig.from_dict(d.get("decoder_cross_attention", {})),
        )


class FeedForwardConfig:
    def __init__(self, type="standard", intermediate_size=256, activation="gelu", dropout=0.1, kwargs=None):
        self.type = type
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.dropout = dropout
        self.kwargs = kwargs or {}

    def to_dict(self):
        return {
            "type": self.type,
            "intermediate_size": self.intermediate_size,
            "activation": self.activation,
            "dropout": self.dropout,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def validate(self):
        assert self.type in {"standard", "glu", "linear"}, f"Invalid FFN type: {self.type}"
        assert self.intermediate_size > 0, "intermediate_size must be > 0"
        assert self.activation in {"gelu", "relu", "swish", "sigmoid"}, f"Unsupported activation: {self.activation}"


class EmbeddingConfig:
    def __init__(self, type="value", dropout=0.1, embedding_dim=None, kwargs=None):
        self.type = type
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.kwargs = kwargs or {}

    def to_dict(self):
        return {
            "type": self.type,
            "dropout": self.dropout,
            "embedding_dim": self.embedding_dim,
            "kwargs": self.kwargs,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            type=d.get("type", "value"),
            dropout=d.get("dropout", 0.1),
            embedding_dim=d.get("embedding_dim", None),
            kwargs=d.get("kwargs", {}),
        )
    def validate(self):
        assert self.type in {"value", "patch", "positional_sinusoidal", "learned"}, f"Invalid embedding type: {self.type}"
        if "positional" in self.type:
            assert self.embedding_dim is not None, "Positional embedding must set embedding_dim"


class HeadAggregationConfig:
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


class NormalizationConfig:
    def __init__(self, norm_type="layer", eps=1e-5):
        self.norm_type = norm_type
        self.eps = eps

    def to_dict(self):
        return {
            "norm_type": self.norm_type,
            "eps": self.eps
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

class TransformerBlockConfig:
    def __init__(self, block_type="standard", attention_config=None, ffn_config=None, kwargs=None):
        self.block_type = block_type
        self.attention_config = attention_config  # nested AttentionConfig
        self.ffn_config = ffn_config              # optional
        self.kwargs = kwargs or {}

    def to_dict(self):
        return {
            "block_type": self.block_type,
            "attention_config": self.attention_config.to_dict() if self.attention_config else None,
            "ffn_config": self.ffn_config.to_dict() if self.ffn_config else None,
            "kwargs": self.kwargs
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            block_type=d["block_type"],
            attention_config=AttentionConfig.from_dict(d["attention_config"]) if d.get("attention_config") else None,
            ffn_config=FeedForwardConfig.from_dict(d["ffn_config"]) if d.get("ffn_config") else None,
            kwargs=d.get("kwargs", {})
        )


class TransformerTimeSeriesConfig(BaseTimeSeriesConfig):
    model_type = "transformer_time_series"

    def __init__(
        self,
        architecture=None,
        attention_blocks=None,
        value_embedding_config=None,
        positional_embedding_config=None,
        feedforward_config=None,
        outputhead_config=None,
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
        #self.embedding_config = embedding_config or EmbeddingConfig()
        self.feedforward_config = feedforward_config or FeedForwardConfig()
        self.norm_config = norm_config or NormalizationConfig()
        self.head_agg_config = head_agg_config or HeadAggregationConfig()
        self.block_configs=  block_configs or TransformerBlockConfig()
        self.output_head_config = output_head_config or OutputHeadConfig()
        self.hidden_size = hidden_size
        self.output_token_lengths = output_token_lengths
        self.num_quantiles = num_quantiles
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.value_embedding_config = value_embedding_config or EmbeddingConfig(type="value")
        self.positional_embedding_config = pos_embedding_config or EmbeddingConfig(type="positional_sinusoidal")

        self.value_embedding_type = self.value_embedding_config.type
        self.pos_embedding_type = self.positional_embedding_config.type



    def save_json(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path):
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_flat_dict(self):
        return {
            "hidden_size": self.hidden_size,
            "prediction_length": self.prediction_length,
            "context_length": self.context_length,
            "attention_type": self.attention_blocks.encoder_attention.attention_type,
            "num_attention_heads": self.attention_blocks.encoder_attention.num_heads,
        }

    @classmethod
    def from_flat_dict(cls, d):
        return cls(
            hidden_size=d["hidden_size"],
            prediction_length=d["prediction_length"],
            context_length=d["context_length"],
            attention_blocks=TransformerAttentionBlockConfig(
                encoder_attention=AttentionConfig(
                    attention_type=d["attention_type"],
                    num_heads=d["num_attention_heads"]
                )
            )
        )

    def to_dict(self):
    return {
        "architecture": self.architecture.to_dict(),
        "attention_blocks": self.attention_blocks.to_dict(),
        "value_embedding_config": self.value_embedding_config.to_dict(),
        "positional_embedding_config": self.positional_embedding_config.to_dict(),
        "feedforward_config": self.feedforward_config.to_dict(),
        "norm_config": self.norm_config.to_dict(),
        "head_agg_config": self.head_agg_config.to_dict(),
        "block_configs": self.block_configs.to_dict() if isinstance(self.block_configs, TransformerBlockConfig) else [b.to_dict() for b in self.block_configs],
        "output_head_config": self.output_head_config.to_dict(),
        "hidden_size": self.hidden_size,
        "output_token_lengths": self.output_token_lengths,
        "num_quantiles": self.num_quantiles,
        "output_attentions": self.output_attentions,
        "output_hidden_states": self.output_hidden_states,
        "use_teacher_forcing": self.use_teacher_forcing,
        **super().to_dict()  # if BaseTimeSeriesConfig defines anything
    }

    @classmethod
    def from_dict(cls, d):
        return cls(
            architecture=TransformerArchitectureConfig.from_dict(d["architecture"]),
            attention_blocks=TransformerAttentionBlockConfig.from_dict(d["attention_blocks"]),
            value_embedding_config=EmbeddingConfig.from_dict(d.get("value_embedding_config", {})),
            positional_embedding_config=EmbeddingConfig.from_dict(d.get("positional_embedding_config", {})),
            feedforward_config=FeedForwardConfig.from_dict(d["feedforward_config"]),
            norm_config=NormalizationConfig.from_dict(d["norm_config"]),
            head_agg_config=HeadAggregationConfig.from_dict(d["head_agg_config"]),
            output_head_config=OutputHeadConfig.from_dict(d["output_head_config"]),
            block_configs=[
                TransformerBlockConfig.from_dict(b) for b in d["block_configs"]
            ] if isinstance(d.get("block_configs"), list) else TransformerBlockConfig.from_dict(d["block_configs"]),
            hidden_size=d["hidden_size"],
            output_token_lengths=d["output_token_lengths"],
            num_quantiles=d["num_quantiles"],
            output_attentions=d["output_attentions"],
            output_hidden_states=d["output_hidden_states"],
            use_teacher_forcing=d.get("use_teacher_forcing", True),
            **{k: v for k, v in d.items() if k not in {
                "architecture", "attention_blocks", "embedding_config", "feedforward_config",
                "norm_config", "head_agg_config", "output_head_config", "block_configs",
                "hidden_size", "output_token_lengths", "num_quantiles",
                "output_attentions", "output_hidden_states", "use_teacher_forcing"
            }}
        )

def validate_config(self):
    # === Basic required model parameters ===
    assert self.hidden_size > 0, "hidden_size must be > 0"
    assert self.output_token_lengths > 0, "output_token_lengths must be > 0"
    assert self.num_quantiles > 0, "num_quantiles must be > 0"
    assert self.output_head_config.output_size is not None, "output_head_config.output_size must be set"

    # === Validate embedding configs ===
    if hasattr(self.value_embedding_config, "validate"):
        self.value_embedding_config.validate()
    if hasattr(self.positional_embedding_config, "validate"):
        self.positional_embedding_config.validate()

    # === Validate feedforward config ===
    if hasattr(self.feedforward_config, "validate"):
        self.feedforward_config.validate()

    # === Validate attention configs ===
    if hasattr(self.attention_blocks.encoder_attention, "validate"):
        self.attention_blocks.encoder_attention.validate()
    if hasattr(self.attention_blocks.decoder_attention, "validate"):
        self.attention_blocks.decoder_attention.validate()
    if hasattr(self.attention_blocks.decoder_cross_attention, "validate"):
        self.attention_blocks.decoder_cross_attention.validate()

    # === Check attention dimensionality consistency ===
    if hasattr(self.attention_blocks.encoder_attention, "num_heads"):
        assert self.hidden_size % self.attention_blocks.encoder_attention.num_heads == 0, \
            "hidden_size must be divisible by encoder_attention.num_heads"
    if hasattr(self.attention_blocks.decoder_attention, "num_heads"):
        assert self.hidden_size % self.attention_blocks.decoder_attention.num_heads == 0, \
            "hidden_size must be divisible by decoder_attention.num_heads"

    # === Validate normalization config ===
    if hasattr(self.norm_config, "validate"):
        self.norm_config.validate()

    # === Validate head aggregator (optional if output_token_lengths > 1) ===
    if self.output_token_lengths > 1:
        assert hasattr(self, "head_agg_config"), "head_agg_config must be defined for multiple output heads"
        if hasattr(self.head_agg_config, "validate"):
            self.head_agg_config.validate()

    # === Validate output head config ===
    if hasattr(self.output_head_config, "validate"):
        self.output_head_config.validate()

    # === Validate layout and block alignment ===
    assert self.architecture.layout in {"encoder", "decoder", "encoder-decoder"}, \
        f"Invalid architecture layout: {self.architecture.layout}"

    if self.architecture.layout in {"encoder", "encoder-decoder"}:
        assert self.architecture.num_encoder_layers > 0, "Must have encoder layers"
        assert hasattr(self, "encoder_blocks"), "Missing encoder_blocks"
        assert len(self.encoder_blocks) == self.architecture.num_encoder_layers, \
            "encoder_blocks must match num_encoder_layers"

    if self.architecture.layout in {"decoder", "encoder-decoder"}:
        assert self.architecture.num_decoder_layers > 0, "Must have decoder layers"
        assert hasattr(self, "decoder_blocks"), "Missing decoder_blocks"
        assert len(self.decoder_blocks) == self.architecture.num_decoder_layers, \
            "decoder_blocks must match num_decoder_layers"

    # === Validate quantiles if used ===
    if "quantile" in self.output_head_config.type or getattr(self, "loss_type", "") in {"quantile", "mq", "wql"}:
        assert isinstance(self.quantiles, (list, tuple)), "quantiles must be a list"
        assert all(0 < q < 1 for q in self.quantiles), "All quantiles must be in (0, 1)"

    # === Validate all blocks ===
    block_list = []
    if hasattr(self, "encoder_blocks") and isinstance(self.encoder_blocks, list):
        block_list += self.encoder_blocks
    if hasattr(self, "decoder_blocks") and isinstance(self.decoder_blocks, list):
        block_list += self.decoder_blocks

    for block_cfg in block_list:
        if hasattr(block_cfg, "validate"):
            block_cfg.validate()

