import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from temporal.modules.encoders import TimeSeriesTransformerEncoder
from temporal.modules.decoders import TimeSeriesTransformerDecoder
from temporal.modules.losses import TimeSeriesLoss
from temporal.modules.attention_head_agg import HeadAggregator
from .basemodel import BaseTimeSeriesModel
from temporal.configs.basetimeseriesconfig import BaseTimeSeriesConfig
from temporal.modules.attention import TimeSeriesAttention


def expand_encoder_mask_2d(mask_2d, seq_len, dtype):
    # mask_2d => [B,S]
    bsz, src_len = mask_2d.shape
    if src_len != seq_len:
        raise ValueError("Mismatch in seq_len")
    # shape => [B,1,src_len,src_len]
    expanded = mask_2d[:, None, None, :].expand(bsz, 1, seq_len, src_len)
    expanded = expanded.to(dtype=dtype)
    # if 1 => keep, 0 => mask, do (1 - expanded) * -1e9
    expanded = (1.0 - expanded) * -1e9
    return expanded


class TimeSeriesTransformerModel(BaseTimeSeriesModel):
    """
    Base time series transformer model for encoder-decoder training.
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

        # Head aggregator module
        self.head_aggregator = HeadAggregator(
            method=config.head_aggregation_method,
            hidden_size=config.hidden_size,
            num_heads=config.output_token_lengths,
            output_size=config.num_quantiles
        )

        # Loss function
        self.loss_fn = TimeSeriesLoss(config, loss_type=config.loss_type)

        self.post_init()

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
        We pass raw inputs to the encoder & decoder. Each submodule
        does its own value_embedding + position_embedding internally.
        """

        # 1) Encoder
        S = input_ids.size(1)

        # Expand to 4D => [B, 1, S, S]
        attention_mask_4d = expand_encoder_mask_2d(
            attention_mask,  # shape [B,S]
            seq_len=S,
            dtype=torch.float32
        )

        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask_4d,  # Now 4D
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 2) Decoder
        if decoder_input_ids is None:
            # e.g. last slice from input_ids if you want 1 step
            decoder_inputs = input_ids[:, -1:]  # shape [B, 1, f]
        else:
            decoder_inputs = decoder_input_ids  # shape [B, T_dec, f]

        decoder_outputs = self.decoder(
            decoder_inputs,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            attention_mask=None,             # For single-step or if no causal mask
            encoder_attention_mask=None,      # you can pass separate masks if needed
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs.last_hidden_state

        # 3) Predictions
        head_outputs = [head(sequence_output) for head in self.output_heads]  # List of [B, T_dec, Q]
        predictions = self.head_aggregator(head_outputs)

        # 4) Loss
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
    AR (Autoregressive) version for rolling forecasts.
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
        Single-step AR loop. You currently do one token at a time.
        """
        batch_size, context_length = input_ids.shape[:2]
        device = input_ids.device

        # 1) Encode once
        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True
        )

        # 2) Initialize decoder input
        if decoder_start_token_value is not None:
            decoder_input = torch.full(
                (batch_size, 1, input_ids.size(-1)),
                decoder_start_token_value,
                dtype=input_ids.dtype,
                device=device
            )
        else:
            decoder_input = input_ids[:, -1:].clone()

        predictions = []
        past_key_values = None

        for step in range(prediction_length):
            decoder_outputs = self.decoder(
                decoder_input,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                attention_mask=None,  # single-step => no self-attn mask needed
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=True
            )

            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]
            head_outputs = [head(last_hidden) for head in self.output_heads]  # => list of [B,1,Q]
            next_pred = self.head_aggregator(head_outputs)                    # => [B,1,Q]

            predictions.append(next_pred)
            decoder_input = next_pred[:, -1:, 0:1]  # e.g. [B,1,1]

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            if early_stopping and eos_token_value is not None:
                if (decoder_input == eos_token_value).all():
                    break

        predictions = torch.cat(predictions, dim=1)  # [B, pred_length, Q]
        return predictions

    def generate_multistep(
        self,
        input_ids: torch.Tensor,
        decoder_length: int,
        encoder_mask_2d: Optional[torch.Tensor] = None,   # shape [B, enc_len], 1 => keep, 0 => mask
        causal: bool = True,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Multi-step decoding in one forward pass.
        - input_ids: [B, enc_len, features]
        - decoder_length: how many future steps we want to decode
        - encoder_mask_2d: 2D mask for ignoring padded tokens in the encoder
        - causal: whether to apply a causal mask on the decoder input
        """

        device = input_ids.device
        batch_size, enc_len, feat_dim = input_ids.shape

        # 1) Encode
        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=encoder_mask_2d,  # if your code auto-expands 2D => 4D
            output_attentions=output_attentions,
            return_dict=True
        )

        # 2) Construct a "decoder input" of shape [B, decoder_length, features]
        #    Initialize it to zeros or some known placeholder if you want a 'warm start'.
        #    In many tasks, you might feed the last known steps or a start token. We'll do zeros here.
        decoder_input = torch.zeros(batch_size, decoder_length, feat_dim, device=device)

        # 3) If we want a causal self-attn mask => shape [decoder_length, decoder_length]
        #    1 => keep, 0 => block future.
        if causal:
            tri_mask = build_causal_mask(decoder_length, device=device)  # => shape [decoder_length, decoder_length]
            tri_mask_2d = tri_mask.unsqueeze(0).expand(batch_size, -1, -1)  # => [B, dec_len, dec_len]

            # Expand to 4D => [B,1,dec_len,dec_len] for self-attn
            decoder_self_mask = expand_mask_4d(tri_mask_2d, tgt_len=decoder_length, dtype=torch.float32)
        else:
            decoder_self_mask = None

        # 4) Cross-attn mask from `encoder_mask_2d` => shape [B, enc_len] => expand => [B,1,dec_len,enc_len]
        if encoder_mask_2d is not None:
            cross_mask_4d = expand_mask_4d(
                encoder_mask_2d, tgt_len=decoder_length, dtype=torch.float32
            )
        else:
            cross_mask_4d = None

        # 5) Decode in one pass
        decoder_outputs = self.decoder(
            decoder_input,  # shape [B, dec_len, features]
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            attention_mask=decoder_self_mask,          # for self-attn
            encoder_attention_mask=cross_mask_4d,      # for cross-attn
            output_attentions=output_attentions,
            return_dict=True
        )

        # 6) Final hidden => [B, dec_len, hidden_size]
        sequence_output = decoder_outputs.last_hidden_state

        # 7) Pass to heads => [B, dec_len, Q]
        head_outputs = [head(sequence_output) for head in self.output_heads]
        predictions = self.head_aggregator(head_outputs)  # => [B, dec_len, Q]

        return predictions
