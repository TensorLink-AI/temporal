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


class TimeSeriesTransformerModel(PreTrainedModel):
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

        # Generate predictions using output heads
        predictions = torch.stack([head(sequence_output) for head in self.output_heads], dim=-2)

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
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Generates autoregressive time series predictions.

        Args:
            input_ids (torch.Tensor): Initial context window (batch_size, context_length).
            prediction_length (int): Number of future steps to predict.
            attention_mask (Optional[torch.Tensor]): Attention mask (optional).
            dynamic_features (Optional[torch.Tensor]): Additional dynamic features.
            static_cat_features (Optional[torch.Tensor]): Static categorical features.
            use_cache (bool): Whether to use past key values for efficiency.
            output_attentions (bool): Whether to return attention weights.

        Returns:
            torch.Tensor: Generated sequence (batch_size, prediction_length).
        """

        batch_size, context_length = input_ids.shape
        device = input_ids.device

        # Start with initial context
        generated_sequence = input_ids.clone()

        for step in range(prediction_length):
            decoder_input = generated_sequence[:, -context_length:]  # Use last `context_length` timesteps

            # Forward pass through encoder and decoder
            encoder_outputs = self.encoder(
                decoder_input, 
                attention_mask=attention_mask,
                dynamic_features=dynamic_features,
                static_cat_features=static_cat_features,
                output_attentions=output_attentions
            )
            decoder_outputs = self.decoder(
                decoder_input,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                use_cache=use_cache
            )

            # Get the last predicted timestep
            next_step = decoder_outputs.last_hidden_state[:, -1, :]  # (batch_size, hidden_size)

            # Append to generated sequence
            next_step = next_step.unsqueeze(1)  # Reshape to (batch_size, 1, hidden_size)
            generated_sequence = torch.cat([generated_sequence, next_step], dim=1)

        return generated_sequence[:, -prediction_length:]
