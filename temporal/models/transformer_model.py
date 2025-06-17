
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, asdict

from temporal.models.base_model import BaseTemporalModel
from temporal.registry.generate import register_generate
from temporal.models.mixin.autoregressive import AutoregressiveMixin
from temporal.models.mixin.multistep import MultiStepMixin

@dataclass
class TransformerOutput:
    """
    A structured output class for the transformer model.

    This dataclass holds all the potential outputs of the `TransformerTemporalModel`,
    making them accessible by attribute. It provides a consistent and predictable
    output format, similar to the output objects in the Hugging Face
    Transformers library.

    Attributes:
        logits (torch.FloatTensor): The final model predictions.
        loss (Optional[torch.FloatTensor]): The loss, computed if targets are provided.
        past_key_values (Optional[Tuple[Tuple[torch.Tensor]]]): The KV cache for
            accelerated decoding.
        decoder_hidden_states (Optional[Tuple[torch.FloatTensor]]): Hidden states
            of the decoder.
        decoder_attentions (Optional[Tuple[torch.FloatTensor]]): Attention weights
            from the decoder's self-attention layers.
        decoder_cross_attentions (Optional[Tuple[torch.FloatTensor]]): Attention
            weights from the decoder's cross-attention layers.
        encoder_last_hidden_state (Optional[torch.FloatTensor]): The last hidden
            state of the encoder.
        encoder_hidden_states (Optional[Tuple[torch.FloatTensor]]): Hidden states
            of the encoder.
        encoder_attentions (Optional[Tuple[torch.FloatTensor]]): Attention weights
            from the encoder's self-attention layers.
    """
    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    def __getitem__(self, key: str) -> Any:
        """Allows dictionary-style access to attributes."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        """Allows dictionary-style setting of attributes."""
        setattr(self, key, value)

    def keys(self) -> List[str]:
        """Returns the names of the attributes."""
        return [f.name for f in self.__dataclass_fields__.values()]

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass to a dictionary."""
        return asdict(self)


def _prepare_decoder_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Tuple[int, int],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int
) -> Optional[torch.Tensor]:
    """Creates a 4D causal attention mask for a decoder.

    This helper function combines a 2D padding mask with a causal mask to
    ensure that the decoder only attends to past positions and non-padded tokens.
    """
    bsz, seq_len = input_shape
    combined_attention_mask = None

    if seq_len > 0:
        causal_mask = _make_causal_mask(
            (bsz, seq_len),
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )
        combined_attention_mask = causal_mask

    if attention_mask is not None:
        expanded_padding_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=seq_len
        ).to(inputs_embeds.device)
        if combined_attention_mask is not None:
            combined_attention_mask = expanded_padding_mask + combined_attention_mask
        else:
            combined_attention_mask = expanded_padding_mask

    return combined_attention_mask

def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0
) -> torch.Tensor:
    """Creates a causal mask for ensuring unidirectional attention."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
) -> torch.Tensor:
    """Expands a 2D padding mask to a 4D attention mask."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


@register_generate(name="transformer")
class TransformerTemporalModel(AutoregressiveMixin, MultiStepMixin, BaseTemporalModel):
    """A concrete implementation of a transformer-based temporal model.

    This class assembles the encoder, decoder, and output heads into a cohesive
    model. It defines the main `forward` pass, handling the flow of data through
    the different components. It also inherits generation capabilities from the
    `AutoregressiveMixin` and `MultiStepMixin` classes.
    """

    def __init__(
        self,
        config,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        output_heads: Optional[nn.Module] = None,
        head_aggregator: Optional[nn.Module] = None,
        loss_fn: Optional[callable] = None
    ):
        """Initializes the TransformerTemporalModel."""
        super().__init__(
            config=config,
            encoder=encoder,
            decoder=decoder,
            output_heads=output_heads,
            head_aggregator=head_aggregator,
            loss_fn=loss_fn
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
    ) -> TransformerOutput:
        """
        Performs a forward pass through the entire transformer model.

        Args:
            encoder_inputs (Optional[torch.Tensor]): Inputs for the encoder,
                shape `[B, L_enc, F_enc]`.
            decoder_inputs (Optional[torch.Tensor]): Inputs for the decoder,
                shape `[B, L_dec, F_dec]`.
            attention_mask (Optional[torch.Tensor]): A 2D padding mask for the
                encoder inputs, shape `[B, L_enc]`.
            decoder_attention_mask (Optional[torch.Tensor]): A 2D padding mask for
                the decoder inputs, shape `[B, L_dec]`.
            targets (Optional[torch.Tensor]): The ground truth values for loss
                calculation, shape `[B, L_dec, F_out]`.
            past_key_values (Optional[Tuple]): A cache of key-value states for
                efficient autoregressive decoding.
            use_cache (Optional[bool]): If True, the model will return the
                updated `past_key_values`.
            output_attentions (Optional[bool]): If True, returns attention weights.
            output_hidden_states (Optional[bool]): If True, returns hidden states
                from all layers.

        Returns:
            TransformerOutput: A structured object containing the model's outputs.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Step 1: Run the encoder if it exists.
        encoder_outputs = None
        if self.encoder:
            if encoder_inputs is None:
                raise ValueError("The model's encoder requires 'encoder_inputs'.")
            encoder_outputs = self.encoder(
                input_values=encoder_inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        # Step 2: Run the decoder.
        encoder_hidden_states = encoder_outputs.last_hidden_state if encoder_outputs else None
        
        if self.decoder:
            if decoder_inputs is None:
                raise ValueError("The model's decoder requires 'decoder_inputs'.")

            # Prepare the decoder attention mask (causal + padding).
            past_kv_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            decoder_mask = _prepare_decoder_attention_mask(
                decoder_attention_mask, decoder_inputs.shape[:-1], decoder_inputs, past_kv_length
            )
            # Prepare the cross-attention mask.
            cross_attention_mask = _expand_mask(attention_mask, self._decoder_dtype, tgt_len=decoder_inputs.shape[1]) if attention_mask is not None else None
            
            decoder_outputs = self.decoder(
                input_ids=decoder_inputs,
                attention_mask=decoder_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=cross_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            input_to_heads = decoder_outputs.last_hidden_state
        else:
            # If there's no decoder, the encoder output is fed to the head.
            if encoder_hidden_states is None:
                raise ValueError("Model requires an encoder or decoder to produce output for the head.")
            input_to_heads = encoder_hidden_states
            decoder_outputs = None # No decoder outputs to return

        # Step 3: Align head input with targets for loss calculation if needed.
        if targets is not None and self.config.architecture.layout == "decoder":
            num_target_steps = targets.size(1)
            # Take the last `num_target_steps` from the head input to align with targets.
            input_to_heads = input_to_heads[:, -num_target_steps:, :]
        
        # Step 4: Project the final hidden states through the output head(s).
        logits = self.output_heads(input_to_heads)
        if self.head_aggregator is not None:
            logits = self.head_aggregator(logits)

        # Step 5: Calculate the loss if targets are provided.
        loss = None
        if targets is not None:
            if self.loss_fn is None:
                raise ValueError("Loss calculation requires a 'loss_fn' to be set on the model.")
            loss = self.loss_fn(logits, targets)

        # Step 6: Construct and return the final output object.
        return TransformerOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values if decoder_outputs else None,
            decoder_hidden_states=decoder_outputs.hidden_states if decoder_outputs else None,
            decoder_attentions=decoder_outputs.attentions if decoder_outputs else None,
            decoder_cross_attentions=decoder_outputs.cross_attentions if decoder_outputs else None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if encoder_outputs else None,
            encoder_hidden_states=encoder_outputs.hidden_states if encoder_outputs else None,
            encoder_attentions=encoder_outputs.attentions if encoder_outputs else None,
        )
