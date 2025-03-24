import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

# Import developed modules
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from .encoders import TimeSeriesTransformerEncoder
from .decoders import TimeSeriesTransformerDecoder
from .embeddings import TimeSeriesValueEmbedding, TimeSeriesSinusoidalPositionalEmbedding
from .losses import TimeSeriesLoss
from .attention_head_agg import HeadAggregator
from .basemodel import  BaseTimeSeriesModel

class TimeSeriesTransformerModel( BaseTimeSeriesModel):
    """
    Base time series transformer model for encoder-decoder training.
    - Supports multiple loss functions (MSE, MAE, RMSE, Quantile, MQ)
    - Uses AR loss with a sliding window for training
    - Supports dynamic feature embeddings
    """

    config_class = TimeSeriesConfig
    base_model_prefix = "time_series_transformer"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Initialize encoder & decoder
        self.encoder = TimeSeriesTransformerEncoder(config)
        self.decoder = TimeSeriesTransformerDecoder(config)

        # Feature embeddings
        self.value_embedding = TimeSeriesValueEmbedding(config.feature_size, config.hidden_size)
        self.position_embedding = TimeSeriesSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.hidden_size
        )

        # Output heads (multi-output for quantile forecasting)
        self.num_quantiles = config.num_quantiles
        self.output_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.num_quantiles) for _ in range(config.output_token_lengths)
        ])

        # Head aggregation module
        self.head_aggregator = HeadAggregator(
            method=config.head_aggregation_method,  # "mean", "gated", "attention", "fusion", "stacked", "weighted_mean"
            hidden_size=config.hidden_size,
            num_heads=config.output_token_lengths,
            output_size=config.num_quantiles
        )

        # Loss function selection
        self.loss_fn = TimeSeriesLoss(config, loss_type=config.loss_type)

        # Initialize weights
        self.post_init()

    def _get_embeddings(
        self,
        input_ids: torch.FloatTensor,
        position_ids: Optional[torch.LongTensor] = None,
        dynamic_features: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Compute value embeddings, positional embeddings, and dynamic feature embeddings."""
        hidden_states = self.value_embedding(input_ids)

        # Add positional encoding
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)
        hidden_states = hidden_states + self.position_embedding(position_ids)

        # Add dynamic feature embeddings
        if self.config.use_dynamic_features and dynamic_features is not None:
            hidden_states = hidden_states + dynamic_features

        return hidden_states

    def forward(
        self,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        dynamic_features: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """
        Forward pass for autoregressive time series prediction.
        """
        # Compute encoder embeddings
        encoder_hidden_states = self._get_embeddings(input_ids, dynamic_features=dynamic_features)

        # Pass through encoder
        encoder_outputs = self.encoder(
            encoder_hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Compute decoder embeddings
        decoder_inputs = decoder_input_ids if decoder_input_ids is not None else input_ids[:, -1:]
        decoder_hidden_states = self._get_embeddings(decoder_inputs)

        # Pass through decoder
        decoder_outputs = self.decoder(
            decoder_hidden_states,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs.last_hidden_state

        # Generate predictions using output heads and aggregate
        head_outputs = [head(sequence_output) for head in self.output_heads]  # List of (B, T, Q)
        predictions = self.head_aggregator(head_outputs)  # Apply head aggregation

        # Compute loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(predictions, labels)

        if not return_dict:
            return (predictions,) + decoder_outputs[1:]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=predictions,
            loss=loss,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class TimeSeriesTransformerARPrediction(TimeSeriesTransformerModel):
    """
    AR (Autoregressive) version of TimeSeriesTransformer for rolling forecasts.
    Inherits from TimeSeriesTransformerModel but adds `generate()` for inference.
    """

    def generate(
        self,
        input_ids: torch.Tensor,
        prediction_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        dynamic_features: Optional[torch.Tensor] = None,
        static_cat_features: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        early_stopping: bool = False,
        decoder_start_token_value: Optional[float] = None,
        eos_token_value: Optional[float] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Generates autoregressive time series predictions using the head aggregation strategy.
        """
        batch_size, context_length = input_ids.shape
        device = input_ids.device

        generated_sequence = input_ids.clone()
        past_key_values = None

        # Encode once
        encoder_outputs = self.encoder(
            self._get_embeddings(input_ids, dynamic_features=dynamic_features),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True
        )

        # Start decoder with last context token or a special token
        if decoder_start_token_value is not None:
            decoder_input = torch.full(
                (batch_size, 1),
                decoder_start_token_value,
                dtype=torch.float32,
                device=device
            )
        else:
            decoder_input = input_ids[:, -1:].clone()

        predictions = []

        for step in range(prediction_length):
            decoder_embeddings = self._get_embeddings(decoder_input)

            decoder_outputs = self.decoder(
                decoder_embeddings,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=True
            )

            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]  # (B, 1, H)

            # Use head aggregator for predictions
            head_outputs = [head(last_hidden) for head in self.output_heads]  # List of (B, 1, Q)
            next_pred = self.head_aggregator(head_outputs)  # Aggregate across heads

            predictions.append(next_pred)
            decoder_input = next_pred[:, -1:, 0:1]  # Feed back one value

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            if early_stopping and eos_token_value is not None:
                if (decoder_input == eos_token_value).all():
                    break

        predictions = torch.cat(predictions, dim=1)  # (B, T, Q)
        return predictions
