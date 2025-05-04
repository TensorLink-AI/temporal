\
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List # Added List
from dataclasses import dataclass # Added dataclass

from temporal.models.base_model import BaseTemporalModel
from temporal.registry.generate import register_generate
# Import the mixins
from temporal.models.mixin.autoregressive import AutoregressiveMixin
from temporal.models.mixin.multistep import MultiStepMixin # Assuming this is the correct name

# --- Output Dataclass --- 
@dataclass
class TransformerOutput:
    """ Base class for model specific outputs. Allows attribute access and acts like a dict. """
    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None # Keeping for potential use
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Allow dictionary-like access
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def keys(self):
         return [k for k, v in self.__dict__.items() if not k.startswith('_')]

# --- End Output Dataclass --- 

# HF-like attention mask processing functions
def _prepare_decoder_attention_mask(
    attention_mask: Optional[torch.Tensor], input_shape: Tuple[int, int], inputs_embeds: torch.Tensor, past_key_values_length: int
) -> Optional[torch.Tensor]:
    """
    Creates causal attention mask for decoding, optionally combining with padding mask.
    Adapted from Hugging Face Transformers library.
    Args:
        attention_mask: 2D padding mask [B, L_dec].
        input_shape: Shape of the decoder input_ids (B, L_dec).
        inputs_embeds: Embedded decoder inputs, used for dtype and device.
        past_key_values_length: Length of past key values (for caching).
    Returns:
        Optional[torch.Tensor]: The combined 4D causal and padding mask, or None.
    """
    combined_attention_mask = None
    bsz, seq_len = input_shape
    if seq_len > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape, inputs_embeds.dtype, device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=seq_len).to(inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
        combined_attention_mask = torch.nan_to_num(combined_attention_mask)
    return combined_attention_mask

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
) -> torch.Tensor:
    """
    Make causal mask used for bi-directional self-attention.
    Adapted from Hugging Face Transformers library.
    Args:
        input_ids_shape: Shape of input IDs (B, T).
        dtype: Data type for the mask.
        device: Device for the mask.
        past_key_values_length: Length of past key values.
    Returns:
        torch.Tensor: A 4D causal mask tensor.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float('-inf'), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`
    and converts it for additive masking (0 for attend, -inf for mask).
    Adapted from Hugging Face Transformers library.
    Args:
        mask: The 2D attention mask (1 for attend, 0 for mask).
        dtype: Target data type for the mask.
        tgt_len: Target length for the third dimension. Defaults to source length.
    Returns:
        The expanded 4D mask ready for additive attention.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), float('-inf'))


@register_generate(name="transformer")
class TransformerTemporalModel(AutoregressiveMixin, MultiStepMixin, BaseTemporalModel):
    """Concrete implementation of a transformer-based temporal model,
    providing both autoregressive and direct multi-step generation methods."""

    def __init__(self,
        config, encoder=None, decoder=None,
        output_heads=None, head_aggregator=None, loss_fn=None
    ):
        """Initialize the TransformerTemporalModel."""
        super().__init__(
            config=config, encoder=encoder, decoder=decoder,
            output_heads=output_heads, head_aggregator=head_aggregator, loss_fn=loss_fn
        )
        self._encoder_dtype = getattr(encoder, 'dtype', torch.float32) if encoder else torch.float32
        self._decoder_dtype = getattr(decoder, 'dtype', torch.float32) if decoder else torch.float32

    def forward(
        self,
        encoder_inputs: Optional[torch.Tensor] = None,
        decoder_inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TransformerOutput: # <<< Changed return type hint
        """Perform a forward pass through the transformer model.

        Args:
            encoder_inputs: Inputs fed to the encoder [B, L_enc, F_enc].
            decoder_inputs: Inputs fed to the decoder [B, L_dec, F_dec].
            attention_mask: 2D mask for encoder_inputs [B, L_enc].
            decoder_attention_mask: 2D mask for decoder_inputs [B, L_dec].
            targets: Target values for loss calculation [B, L_dec, F_out].
            past_key_values: Cached key/values for faster decoding.
            use_cache: Whether to return key/values for caching.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states for all layers.

        Returns:
            TransformerOutput: An object containing model outputs like logits, loss, etc.
        """
        # Prepare kwargs for sub-modules
        encoder_call_kwargs = {
            "output_attentions": output_attentions, "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        decoder_call_kwargs = {
            "use_cache": use_cache, "past_key_values": past_key_values,
            "output_attentions": output_attentions, "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }

        encoder_output_obj = None
        decoder_output_obj = None
        processed_encoder_mask = None
        processed_cross_mask = None

        # 1. Process Encoder Mask & Run Encoder
        if self.encoder:
            if encoder_inputs is None: raise ValueError("Encoder exists but encoder_inputs are None.")
            if attention_mask is not None:
                 processed_encoder_mask = _expand_mask(attention_mask, dtype=self._encoder_dtype)
            encoder_output_obj = self.encoder(
                input_values=encoder_inputs,
                attention_mask=processed_encoder_mask,
                **encoder_call_kwargs
            )

        # 2. Prepare Decoder Masks, Inputs & Run Decoder
        input_to_heads = None
        encoder_hidden_states = None
        if self.decoder:
            if encoder_output_obj is not None:
                 encoder_hidden_states = encoder_output_obj.get('last_hidden_state') # Use .get for dict-like access
            if decoder_inputs is None: raise ValueError("Decoder exists but decoder_inputs are None.")

            decoder_input_shape = decoder_inputs.shape[:-1]
            past_kv_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            processed_decoder_mask = _prepare_decoder_attention_mask(
                 decoder_attention_mask, decoder_input_shape, decoder_inputs, past_kv_length
            )
            if encoder_hidden_states is not None and attention_mask is not None:
                processed_cross_mask = _expand_mask(
                    attention_mask, dtype=self._decoder_dtype, tgt_len=decoder_input_shape[-1]
                )

            decoder_output_obj = self.decoder(
                input_ids=decoder_inputs,
                attention_mask=processed_decoder_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=processed_cross_mask,
                **decoder_call_kwargs
            )
            input_to_heads = decoder_output_obj.get('last_hidden_state')
        elif encoder_output_obj is not None: # Encoder-only case
             input_to_heads = encoder_output_obj.get('last_hidden_state')

        # 3. Check for Head Input & Run Heads
        if input_to_heads is None: raise ValueError("No output generated for heads.")
        logits = self.output_heads(input_to_heads)
        if self.head_aggregator is not None: logits = self.head_aggregator(logits)

        # 4. Calculate Loss
        loss = None
        if targets is not None:
            if self.loss_fn is None: raise ValueError("Loss calculation requires a loss_fn.")
            loss = self.loss_fn(logits, targets)

        # 5. Construct and return TransformerOutput object
        return TransformerOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_output_obj.get("past_key_values") if use_cache and decoder_output_obj else None,
            decoder_hidden_states=decoder_output_obj.get("hidden_states") if output_hidden_states and decoder_output_obj else None,
            decoder_attentions=decoder_output_obj.get("attentions") if output_attentions and decoder_output_obj else None,
            decoder_cross_attentions=decoder_output_obj.get("cross_attentions") if output_attentions and decoder_output_obj else None,
            encoder_last_hidden_state=encoder_output_obj.get("last_hidden_state") if encoder_output_obj else None,
            encoder_hidden_states=encoder_output_obj.get("hidden_states") if output_hidden_states and encoder_output_obj else None,
            encoder_attentions=encoder_output_obj.get("attentions") if output_attentions and encoder_output_obj else None,
        )

    # Mixin methods (generate, generate_multistep) are inherited
