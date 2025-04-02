import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

# Import developed modules
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from temporal.modules.encoders import TimeSeriesTransformerEncoder
from temporal.modules.decoders import TimeSeriesTransformerDecoder
from temporal.modules.losses import TimeSeriesLoss
from temporal.modules.attention_head_agg import HeadAggregator
from .basemodel import BaseTimeSeriesModel
from temporal.configs.basetimeseriesconfig import BaseTimeSeriesConfig
from temporal.modules.attention import TimeSeriesAttention

class TimeSeriesTransformerModel(BaseTimeSeriesModel):
    """
    Base time series transformer model for encoder-decoder training.
    - Supports multiple loss functions (MSE, MAE, RMSE, Quantile, MQ)
    - Uses AR loss with a sliding window for training
    - Supports dynamic feature embeddings
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Initialize encoder & decoder (which each handle embedding internally)
        self.encoder = TimeSeriesTransformerEncoder(config)
        self.decoder = TimeSeriesTransformerDecoder(config)

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

        self.post_init()

    def forward(
        self,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        dynamic_features: Optional[torch.FloatTensor] = None,  # optional if your encoder/decoder expect it
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """
        Forward pass for autoregressive time series prediction.
        We pass raw inputs to the encoder & decoder. Each submodule
        does its own value_embedding + position_embedding internally.
        """

        # ---------------------------------------------------
        # 1) Encoder: pass raw input (shape [B, S, features]) to encoder
        #    The encoder itself calls self.value_embedding(...).
        encoder_outputs = self.encoder(
            input_ids,  # raw shape [B, S, f]
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # ---------------------------------------------------
        # 2) Decoder: pass raw input for the decoder
        if decoder_input_ids is None:
            # e.g. last slice from input_ids if you want 1 step, or [-decoder_length:] if you prefer
            decoder_inputs = input_ids[:, -1:]  # shape [B, 1, f]
        else:
            decoder_inputs = decoder_input_ids

        decoder_outputs = self.decoder(
            decoder_inputs,  # raw shape => [B, T, f]
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs.last_hidden_state

        # ---------------------------------------------------
        # 3) Generate final predictions
        head_outputs = [head(sequence_output) for head in self.output_heads]  # List of (B, T, Q)
        predictions = self.head_aggregator(head_outputs)  # Apply head aggregation

        # ---------------------------------------------------
        # 4) Compute loss (optional)
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
        We pass raw inputs to the encoder & decoder. Each submodule
        does its own embedding.
        """
        batch_size, context_length = input_ids.shape[:2]  # e.g. [B, S, f]
        device = input_ids.device

        # 1) Encode once
        encoder_outputs = self.encoder(
            input_ids,                  # raw shape => [B, S, f]
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True
        )

        # 2) Initialize decoder input with last token or a special token
        if decoder_start_token_value is not None:
            decoder_input = torch.full(
                (batch_size, 1, input_ids.shape[-1]),  # shape => [B, 1, f], if input_ids is [B, S, f]
                decoder_start_token_value,
                dtype=input_ids.dtype,
                device=device
            )
        else:
            # Last step from input_ids
            decoder_input = input_ids[:, -1:].clone()  # shape [B, 1, f]

        # 3) Autoregressive loop
        predictions = []
        past_key_values = None

        for step in range(prediction_length):
            # Pass raw shape => [B, 1, f] to decoder
            decoder_outputs = self.decoder(
                decoder_input,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=True
            )

            # last hidden => shape [B, 1, hidden_size]
            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]

            # pass last_hidden to heads => [B, 1, Q]
            head_outputs = [head(last_hidden) for head in self.output_heads]
            next_pred = self.head_aggregator(head_outputs)  # shape => [B, 1, Q]

            predictions.append(next_pred)

            # feed the last value (or last dimension if multi-step) back
            # e.g. if single-step univariate => [B, 1, 1]
            decoder_input = next_pred[:, -1:, 0:1]  # shape => [B, 1, 1]

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            if early_stopping and eos_token_value is not None:
                if (decoder_input == eos_token_value).all():
                    break

        # 4) Concatenate predictions => [B, pred_length, Q]
        predictions = torch.cat(predictions, dim=1)
        return predictions
