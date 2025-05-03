import torch
from transformers import PreTrainedModel
from temporal.configs.transformer_config import TransformerTimeSeriesConfig
from temporal.models.builder import build_time_series_transformer

class TimeSeriesTransformerModel(PreTrainedModel):
    """
    Hugging Face–compatible wrapper around our Temporal transformer.
    You can now do `.save_pretrained()` / `.from_pretrained()` / `Trainer` / `pipeline`.
    """
    config_class      = TransformerTimeSeriesConfig
    base_model_prefix = "time_series_transformer"

    def __init__(self, config: TransformerTimeSeriesConfig):
        super().__init__(config)
        # Composition: your core model stays untouched
        self.temporal = build_time_series_transformer(config)

    def forward(self, input_values, attention_mask=None, **kwargs):
        # Delegate to your core transformer
        return self.temporal(
            input_values,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(self, input_values, prediction_length=None, **kwargs):
        # Delegate to your core model’s generate method
        return self.temporal.generate(
            input_values,
            prediction_length=prediction_length or self.config.prediction_length,
            **kwargs
        )
