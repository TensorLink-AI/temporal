from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class TimeSeriesTransformerModel(PreTrainedModel):
    config_class = TimeSeriesConfig
    base_model_prefix = "time_series_transformer"

    def __init__(self, config):
        super().__init__(config)

        # Model Components
        self.encoder = TimeSeriesTransformerEncoder(config)
        self.decoder = TimeSeriesTransformerDecoder(config)
        self.loss_fn = TimeSeriesLoss(config, loss_type=config.loss_type)

        # Initialize weights
        self.post_init()

    def forward(self, input_ids, labels=None, **kwargs):
        """Forward pass with loss computation."""
        encoder_outputs = self.encoder(input_ids, **kwargs)
        decoder_outputs = self.decoder(encoder_outputs.last_hidden_state, **kwargs)

        predictions = decoder_outputs.last_hidden_state
        loss = None
        if labels is not None:
            loss = self.loss_fn(predictions, labels)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=predictions, loss=loss
        )
