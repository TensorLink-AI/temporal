from temporal.configs.basetimeseriesconfig import BaseTimeSeriesConfig

class TransformerTimeSeriesConfig(BaseTimeSeriesConfig):
    """
    Configuration class for Transformer-based time-series models.

    This config extends BaseTimeSeriesConfig by adding Transformer-specific
    parameters such as the number of layers, hidden dimensions, attention heads,
    and dropout rates. It allows flexible creation of an encoder–decoder or
    single-stack Transformer for time-series tasks.

    Args:
        n_layers (int, optional, defaults to 2):
            Number of Transformer layers (stacked blocks) in the encoder
            or combined encoder–decoder, depending on your model usage.
            Typically referred to as `num_hidden_layers` in Hugging Face standards.
        d_model (int, optional, defaults to 64):
            [Deprecated or Alternative?] If you store a separate “d_model” name,
            note that your actual code sets `hidden_size`. If this is purely a
            reference, consider removing or ensuring it syncs with `hidden_size`.
        d_ff (int, optional, defaults to 256):
            [Potentially Deprecated?] Often replaced by `intermediate_size`.
            The dimension of the feed-forward sub-layer.
        n_heads (int, optional, defaults to 4):
            [Deprecated or Alternative?] Typically replaced by `num_attention_heads`.
        dropout (float, optional, defaults to 0.1):
            [Deprecated or Alternative?] If your code sets `attention_dropout`
            and `hidden_dropout_prob` separately, confirm if this field is
            needed or if it merges them.
        hidden_size (int, optional, defaults to 64):
            The main dimensionality of the model’s hidden states. Must be
            divisible by `num_attention_heads`.
        intermediate_size (int, optional, defaults to 256):
            Dimension of the feed-forward network's “intermediate” layer.
        num_attention_heads (int, optional, defaults to 4):
            Number of attention heads per Transformer layer.
        attention_dropout (float, optional, defaults to 0.1):
            Dropout rate within the self-attention mechanism.
        hidden_dropout_prob (float, optional, defaults to 0.1):
            Dropout rate applied after attention and feed-forward sub-layers
            (residual connections).
        hidden_act (str, optional, defaults to "gelu"):
            Activation function in the feed-forward layer. Choices typically
            include "gelu", "relu", etc.
        encoder_layerdrop (float, optional, defaults to 0.1):
            Probability of dropping an entire Transformer layer during training
            (for “layer-drop” or “stochastic depth”). If 0.0, layer dropping
            is disabled.
        layer_norm_eps (float, optional, defaults to 1e-12):
            Epsilon value for numerical stability in LayerNorm.
        output_attentions (bool, optional, defaults to False):
            If True, the model returns attention weights for each layer.
        output_hidden_states (bool, optional, defaults to False):
            If True, the model returns the hidden states at each layer.

    Attributes:
        num_hidden_layers (int):
            Internally set to `n_layers`, controlling the number of stacked
            Transformer blocks.
        hidden_size (int):
            Internally set to `hidden_size` (must match `d_model` if your
            code merges them).
        intermediate_size (int):
            Internally set from `d_ff` or the provided value.
        num_attention_heads (int):
            Internally set to `n_heads` or provided value, controlling
            multi-head attention.
        etc.

    Example:
        >>> from temporal.configs.basetimeseriesconfig import BaseTimeSeriesConfig
        >>> config = TransformerTimeSeriesConfig(
        ...     context_length=48,  # inherited from BaseTimeSeriesConfig
        ...     prediction_length=12,
        ...     n_layers=4,
        ...     hidden_size=128,
        ...     num_attention_heads=8
        ... )
        >>> # You can now pass this config to your TransformerTimeSeriesModel.

    Note:
        Some arguments (`d_model`, `d_ff`, `n_heads`, `dropout`) may be
        partially redundant if you rely exclusively on `hidden_size`,
        `intermediate_size`, `num_attention_heads`, `attention_dropout`,
        and `hidden_dropout_prob`. Check if your code merges them or
        if you plan to remove the older fields in future.
    """

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
