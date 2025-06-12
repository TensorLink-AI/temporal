from typing import Any, Optional
import torch
from transformers import PreTrainedModel
from temporal.configs.transformer_config import TransformerTimeSeriesConfig
from temporal.models.builder import build_time_series_transformer

class TimeSeriesTransformerModel(PreTrainedModel):
    """A Hugging Face-compatible wrapper for the custom TimeSeriesTransformer.

    This class acts as an adapter, allowing a `TimeSeriesTransformer` model to be
    used seamlessly within the Hugging Face ecosystem. By inheriting from
g    `transformers.PreTrainedModel` and defining the `config_class`, this wrapper
    enables standard Hugging Face functionalities like `.from_pretrained()`,
    `.save_pretrained()`, and integration with the `Trainer` and `pipeline` APIs.

    The core logic is delegated to the underlying `TimeSeriesTransformer` instance,
    which is created during initialization.

    Attributes:
        config_class: Specifies the configuration class to be used with this model.
        base_model_prefix: A name for the core model attribute, used by Hugging Face.
        temporal: The instance of the underlying `TimeSeriesTransformer`.
    """
    config_class = TransformerTimeSeriesConfig
    base_model_prefix = "time_series_transformer"

    def __init__(self, config: TransformerTimeSeriesConfig):
        """Initializes the TimeSeriesTransformerModel wrapper.

        Args:
            config (TransformerTimeSeriesConfig): The configuration object for the model.
        """
        super().__init__(config)
        # The core model is built and held as an attribute.
        self.temporal = build_time_series_transformer(config)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Any:
        """Delegates the forward pass to the underlying temporal model.

        Args:
            input_values (torch.Tensor): The input tensor for the model.
            attention_mask (Optional[torch.Tensor]): An optional attention mask.
            **kwargs: Any other keyword arguments to be passed to the underlying model.

        Returns:
            Any: The output from the underlying `TimeSeriesTransformer`'s forward method.
        """
        # All arguments are passed directly to the composed model.
        return self.temporal(
            input_values=input_values,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(
        self,
        input_values: torch.Tensor,
        prediction_length: Optional[int] = None,
        **kwargs: Any
    ) -> Any:
        """Delegates the generation task to the underlying temporal model.

        This allows the model to be used for autoregressive forecasting tasks.

        Args:
            input_values (torch.Tensor): The initial sequence to start generation from.
            prediction_length (Optional[int]): The number of time steps to forecast.
                If None, it defaults to the `prediction_length` in the model's config.
            **kwargs: Any other keyword arguments for the generation process.

        Returns:
            Any: The generated forecast sequence from the underlying model.
        """
        # The prediction_length from the config is used as a fallback.
        pred_len = prediction_length if prediction_length is not None else self.config.prediction_length
        return self.temporal.generate(
            input_values=input_values,
            prediction_length=pred_len,
            **kwargs
        )
