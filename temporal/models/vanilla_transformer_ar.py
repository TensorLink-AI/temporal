import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from transformers import PreTrainedModel
from transformers.modeling_outputs import  Seq2SeqLMOutput
from temporal.modules.encoders import TimeSeriesTransformerEncoder
from temporal.modules.decoders import TimeSeriesTransformerDecoder
from temporal.modules.losses import TimeSeriesLoss
from temporal.modules.attention_head_agg import HeadAggregator
from .basemodel import BaseTimeSeriesModel
from temporal.configs.basetimeseriesconfig import BaseTimeSeriesConfig
from temporal.modules.attention import TimeSeriesAttention

# We assume you already have these defined in your code.
def expand_encoder_mask_2d(mask_2d: torch.Tensor, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert a 2D encoder attention mask of shape [batch_size, seq_len]
    into a 4D attention mask [batch_size, 1, seq_len, seq_len] with
    -inf where tokens should be masked.

    Args:
        mask_2d (torch.Tensor):
            2D mask of shape [B, seq_len], where 1.0 => keep and 0.0 => mask.
        seq_len (int):
            Expected sequence length; used to check shape consistency.
        dtype (torch.dtype):
            The dtype to which the expanded mask is cast (e.g. float32).

    Returns:
        torch.Tensor of shape [B, 1, seq_len, seq_len], with values 0 for
        “keep” and -1e9 for “mask”. This can be added to attention logits
        inside the Transformer to block padded tokens.
    """
    bsz, src_len = mask_2d.shape
    if src_len != seq_len:
        raise ValueError(f"Mismatch in seq_len: got {src_len}, expected {seq_len}.")

    # Expand to [B, 1, seq_len, seq_len]
    expanded = mask_2d[:, None, None, :].expand(bsz, 1, seq_len, src_len)
    expanded = expanded.to(dtype=dtype)

    # Convert 1 => 0.0 keep, 0 => -1e9 mask
    expanded = (1.0 - expanded) * -1e9
    return expanded


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build a lower-triangular (causal) mask for self-attention in the decoder,
    blocking access to future tokens.

    Args:
        seq_len (int):
            The length of the target sequence for which we build the mask.
        device (torch.device):
            The device on which the returned mask resides.

    Returns:
        A 2D tensor of shape [seq_len, seq_len], where the upper-right
        triangle is 0 (blocked) and the lower-left triangle (including diag)
        is 1 (keep).
    """
    mat = torch.ones(seq_len, seq_len, device=device)
    mat = torch.tril(mat)  # zero out upper triangle
    return mat  # shape [seq_len, seq_len]


def expand_mask_4d(mask_2d: torch.Tensor, tgt_len: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Expand a 2D or 3D mask into a 4D mask for self-attention or cross-attention.

    Typically used for building a causal mask or cross-attn mask:
      - shape => [B, seq_len, seq_len] or [B, seq_len]
      - expand => [B, 1, seq_len, seq_len]
    then convert 1 => 0.0 keep, 0 => -1e9 mask.

    Args:
        mask_2d (torch.Tensor):
            Mask of shape [B, seq_len, seq_len] or [B, seq_len].
        tgt_len (int):
            The target sequence length (if needed to shape the final mask).
        dtype (torch.dtype):
            The dtype for the returned mask.

    Returns:
        torch.Tensor of shape [B, 1, seq_len, seq_len].
        Positions to block become -1e9, positions to keep are 0.0.
    """
    bsz, src_len = mask_2d.shape[:2]
    expanded = mask_2d[:, None].expand(bsz, 1, src_len, src_len)
    expanded = expanded.to(dtype=dtype)
    expanded = (1.0 - expanded) * -1e9
    return expanded



class TimeSeriesTransformerModel(BaseTimeSeriesModel):
    """
    A base time-series Transformer model for encoder–decoder tasks,
    supporting multi-step teacher forcing in a single forward pass.

    This model:
      1. Runs an encoder on [B, S, features] of input time-series data.
      2. Runs a decoder on either:
         - single-step input if multi_step=False
         - full future horizon if multi_step=True
      3. Produces final predictions via multiple “output heads,” possibly
         aggregated by a `HeadAggregator`.

    Args:
        config (BaseTimeSeriesConfig):
            Configuration object specifying hidden sizes, loss type, number
            of quantiles, etc.

    Forward Args:
        input_ids (torch.FloatTensor):
            The encoder “input” of shape [B, S, features].
        attention_mask (torch.FloatTensor, optional):
            A 2D or 4D mask specifying which tokens to attend to
            in the encoder. If shape [B, S], it is expanded to [B,1,S,S].
        decoder_input_ids (torch.FloatTensor, optional):
            The decoder “input” of shape [B, T_dec, features].
            If `multi_step=True`, expects the entire future horizon.
            If not provided, and multi_step=False, defaults to the last step
            of `input_ids`.
        labels (torch.FloatTensor, optional):
            The ground-truth future time-series of shape [B, T_dec, Q].
            If provided, a loss is computed via `self.loss_fn`.
        dynamic_features (torch.FloatTensor, optional):
            Additional per-timestep features used inside the model if needed.
            Not used directly in this snippet, but shown for future expansion.
        output_attentions (bool, optional):
            If True, returns attention weights in the decoder and possibly
            the encoder.
        output_hidden_states (bool, optional):
            If True, returns hidden states of each encoder/decoder layer.
        return_dict (bool, optional):
            If True, returns a `Seq2SeqLMOutput` dataclass. Otherwise returns
            a tuple.
        multi_step (bool, optional, defaults to True):
            Whether the model is run in “multi-step teacher forcing,” passing
            the entire horizon to the decoder in one go.
        use_causal_mask (bool, optional, defaults to True):
            If multi_step=True, whether to create a lower-triangular mask in
            the decoder to block future positions.

    Returns:
        Seq2SeqLMOutput or tuple:
            - If return_dict=True: 
              A Seq2SeqLMOutput with fields:
                * loss (optional): The computed loss if `labels` was provided.
                * logits: The final predictions of shape [B, T_dec, Q].
                * (optionally) decoder_hidden_states, attentions, cross_attentions,
                  and encoder states.
            - If return_dict=False:
              A tuple of (predictions, loss, hidden_states, attentions).

    Example Usage:
        >>> config = BaseTimeSeriesConfig(context_length=48, prediction_length=12)
        >>> model = TimeSeriesTransformerModel(config)
        >>> input_ids = torch.randn(16, 48, config.feature_size)
        >>> decoder_input_ids = torch.randn(16, 12, config.feature_size)
        >>> labels = torch.randn(16, 12, config.num_quantiles)  # e.g. quantile outputs
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> preds = outputs.logits
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
    ) -> Seq2SeqLMOutput:

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

        return Seq2SeqLMOutput(
            loss=loss,
            logits=predictions,                          # rename from last_hidden_state => logits
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class TimeSeriesTransformerARPrediction(TimeSeriesTransformerModel):
    """
    AR (Autoregressive) extension of TimeSeriesTransformerModel for
    rolling forecasts in an auto-regressive loop.

    This class reuses the same encoder–decoder architecture but
    provides a `generate(...)` method that iteratively decodes
    one step at a time, feeding the model’s previous predictions
    back into the next time-step.

    Args:
        config (BaseTimeSeriesConfig):
            Configuration object, as in the parent class.

    generate Args:
        input_ids (torch.Tensor):
            Shape [B, context_len, features]. The initial historical
            data for the encoder.
        prediction_length (int):
            How many steps to roll out auto-regressively.
        attention_mask (torch.Tensor, optional):
            Encoder mask for ignoring padded tokens, shape [B, context_len].
        dynamic_features (torch.Tensor, optional):
            Additional features. Not directly used unless your model
            merges them into the decoder input somehow.
        static_cat_features (torch.Tensor, optional):
            Unused in this snippet, but placeholders for future expansions.
        use_cache (bool, optional, defaults to True):
            Whether to use past key-values to speed up decoding in
            an AR loop (if your decoder supports it).
        early_stopping (bool, optional, defaults to False):
            If True, and an `eos_token_value` is set, generation stops
            early if all tokens match `eos_token_value`.
        decoder_start_token_value (float, optional):
            If given, starts the decoder with a special token instead
            of the last input value.
        eos_token_value (float, optional):
            Value that triggers early stopping if encountered in
            all predicted steps.
        output_attentions (bool, optional):
            If True, returns attentions from each decode step.

    Returns:
        torch.Tensor of shape [B, prediction_length, Q]
        containing the rolling forecasts. (Q might be 1 or
        `num_quantiles`, depending on your config.)

    Example:
        >>> model = TimeSeriesTransformerARPrediction(config)
        >>> input_ids = torch.randn(16, 48, config.feature_size)  # 48-step context
        >>> # Generate a 12-step forecast in an AR loop:
        >>> preds = model.generate(
        ...     input_ids=input_ids,
        ...     prediction_length=12
        ... )
        >>> preds.shape  # [16, 12, Q]
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
