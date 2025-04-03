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

# We assume you already have these defined in your code.
def expand_encoder_mask_2d(mask_2d, seq_len, dtype):
    bsz, src_len = mask_2d.shape
    if src_len != seq_len:
        raise ValueError("Mismatch in seq_len")
    expanded = mask_2d[:, None, None, :].expand(bsz, 1, seq_len, src_len)
    expanded = expanded.to(dtype=dtype)
    expanded = (1.0 - expanded) * -1e9
    return expanded

def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    shape => [seq_len, seq_len], with 1 => keep, 0 => block future
    """
    mat = torch.ones(seq_len, seq_len, device=device)
    mat = torch.tril(mat)
    return mat  # [seq_len, seq_len]

def expand_mask_4d(
    mask_2d: torch.Tensor,
    tgt_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    shape => [B, seq_len, seq_len] => expand to [B, 1, seq_len, seq_len]
    or if mask_2d => [B, seq_len], e.g. build your logic accordingly.
    """
    bsz, src_len = mask_2d.shape[:2]
    expanded = mask_2d[:, None].expand(bsz, 1, src_len, src_len)
    expanded = expanded.to(dtype=dtype)
    expanded = (1.0 - expanded) * -1e9
    return expanded


class TimeSeriesTransformerModel(BaseTimeSeriesModel):
    """
    Base time series transformer model for encoder-decoder training,
    now supporting multi-step teacher forcing in a single forward.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Initialize encoder & decoder
        self.encoder = TimeSeriesTransformerEncoder(config)
        self.decoder = TimeSeriesTransformerDecoder(config)

        # Output heads (e.g., for quantile forecasting)
        self.num_quantiles = config.num_quantiles
        self.output_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.num_quantiles) 
            for _ in range(config.output_token_lengths)
        ])

        # Head aggregator
        self.head_aggregator = HeadAggregator(
            method=config.head_aggregation_method,
            hidden_size=config.hidden_size,
            num_heads=config.output_token_lengths,
            output_size=config.num_quantiles
        )

        self.loss_fn = TimeSeriesLoss(config, loss_type=config.loss_type)

        self.post_init()

    def forward(
        self,
        input_ids: torch.FloatTensor,              # [B, S, features]
        attention_mask: Optional[torch.FloatTensor] = None,  # [B, S]
        decoder_input_ids: Optional[torch.FloatTensor] = None, # [B, T_dec, features]
        labels: Optional[torch.FloatTensor] = None,           # [B, T_dec, Q] for multi-step
        dynamic_features: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        multi_step: bool = True,            # <-- new param: if True, do multi-step teacher forcing
        use_causal_mask: bool = True,        # <-- if multi_step, do we apply a causal mask?
    ) -> BaseModelOutputWithPastAndCrossAttentions:

        # 1) Encoder
        B, S, _ = input_ids.shape
        if attention_mask is not None:
            # expand encoder mask to 4D [B,1,S,S]
            encoder_mask_4d = expand_encoder_mask_2d(
                attention_mask, seq_len=S, dtype=input_ids.dtype
            )
        else:
            encoder_mask_4d = None

        encoder_outputs = self.encoder(
            inputs_embeds=input_ids,           # or pass input_ids if your code expects that
            attention_mask=encoder_mask_4d,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 2) If multi_step = False, fallback to single-step. If no decoder_input_ids, we do last slice
        if not multi_step:
            if decoder_input_ids is None:
                decoder_input_ids = input_ids[:, -1:]  # shape => [B,1,features]
            
            # we do no causal mask in single-step mode
            decoder_mask = None

        else:
            # multi_step training: we pass the entire future horizon as decoder_input_ids
            if decoder_input_ids is None:
                raise ValueError("For multi_step=True, must provide decoder_input_ids with shape [B, T_dec, features]")

            T_dec = decoder_input_ids.size(1)
            if use_causal_mask and T_dec > 1:
                # Build a [T_dec, T_dec] lower-tri mask
                tri = build_causal_mask(T_dec, device=input_ids.device)  # [T_dec, T_dec] => 1 keep, 0 block
                # expand to [B, T_dec, T_dec]
                tri_3d = tri.unsqueeze(0).expand(B, T_dec, T_dec)  # shape => [B,T_dec,T_dec]
                # expand to [B,1,T_dec,T_dec]
                decoder_mask = expand_mask_4d(tri_3d, tgt_len=T_dec, dtype=input_ids.dtype)
            else:
                decoder_mask = None

        decoder_outputs = self.decoder(
            decoder_input_ids,                 # shape [B, T_dec, features]
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            attention_mask=decoder_mask,       # for self-attn
            encoder_attention_mask=None,       # if you want cross-attn mask for encoder
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # [B, T_dec, hidden_size]
        sequence_output = decoder_outputs.last_hidden_state

        # 3) Predictions => [B, T_dec, Q]
        head_outputs = [head(sequence_output) for head in self.output_heads]
        predictions = self.head_aggregator(head_outputs)

        # 4) Compute Loss
        loss = None
        if labels is not None:
            # expects labels => shape [B, T_dec, Q]
            # model => shape [B, T_dec, Q]
            loss = self.loss_fn(predictions, labels)

        if not return_dict:
            return (predictions, loss, decoder_outputs.hidden_states, decoder_outputs.attentions)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=predictions,
            # Typically BaseModelOutputWithPastAndCrossAttentions does not include 'loss',
            # so we return it as a separate item or define a custom output. For simplicity:
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # or store the encoder outputs as well:
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
