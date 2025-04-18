from transformers import PretrainedConfig

PROBABILISTIC_LOSSES = ["quantile", "mq"]

class BaseTimeSeriesConfig(PretrainedConfig):
    model_type = "base_time_series"

    def __init__(
        self,
        feature_size: int = 1,
        context_length: int = 128,
        prediction_length: int = 12,
        quantiles: list = [0.1, 0.5, 0.9],

        output_token_lengths: int = 1,
        loss_type: str = "quantile",

        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        autoregressive: bool = True,
        is_decoder: bool = False,

        hidden_size: int = 128,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        ffn_hidden_size: int = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,

        use_layer_norm: bool = True,
        use_positional_encoding: bool = True,
        use_skip_connections: bool = True,
        value_embedding_type: str = "value",
        pos_embedding_type: str = "positional_sinusoidal",

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.feature_size = feature_size
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.output_token_lengths = output_token_lengths
        self.loss_type = loss_type
        self.use_dynamic_features = use_dynamic_features
        self.use_static_features = use_static_features
        self.autoregressive = autoregressive
        self.is_decoder = is_decoder

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_hidden_size = ffn_hidden_size or 4 * hidden_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        self.use_layer_norm = use_layer_norm
        self.use_positional_encoding = use_positional_encoding
        self.use_skip_connections = use_skip_connections
        self.value_embedding_type = value_embedding_type
        self.pos_embedding_type = pos_embedding_type

        self.validate_config()

    def validate_config(self):
        assert self.context_length > 0
        assert self.prediction_length > 0
        assert self.hidden_size % self.num_attention_heads == 0
        if self.loss_type in PROBABILISTIC_LOSSES:
            assert all(0 < q < 1 for q in self.quantiles)
