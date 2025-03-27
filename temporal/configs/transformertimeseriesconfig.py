from temporal.configs.basetimeseriesconfig import BaseTimeseriesConfig

class TransformerTimeSeriesConfig(BaseTimeSeriesConfig):
    """Configuration for Transformer based time series models."""
    model_type = "transformer_time_series"

    def __init__(
        self,
        n_layers: int = 2,
        d_model: int = 64,
        d_ff: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        hidden_size: int = 64,
        intermediate_size: int = 256,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "gelu",
        encoder_layerdrop: float = 0.1,
        layer_norm_eps: float = 1e-12,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_return_dict: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_hidden_layers = n_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.encoder_layerdrop = encoder_layerdrop
        self.layer_norm_eps = layer_norm_eps
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict