\
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from temporal.models.base_model import BaseTemporalModel
from temporal.registry.generate import register_generate
# Import the mixins
from temporal.models.mixin.autoregressive import AutoregressiveMixin
from temporal.models.mixin.multistep import MultiStepMixin # Assuming this is the correct name

# HF-like attention mask processing functions (can be placed in utils later)
def _prepare_decoder_attention_mask(
    attention_mask: Optional[torch.Tensor], input_shape: Tuple[int, int], inputs_embeds: torch.Tensor, past_key_values_length: int
) -> Optional[torch.Tensor]:
    """
    Creates causal attention mask for decoding, optionally combining with padding mask.
    Adapted from Hugging Face Transformers library.

    Args:
        attention_mask (torch.Tensor, optional): 2D padding mask [B, L_dec].
        input_shape (Tuple[int, int]): Shape of the decoder input_ids (B, L_dec).
        inputs_embeds (torch.Tensor): Embedded decoder inputs, used for dtype and device.
        past_key_values_length (int): Length of past key values (for caching).

    Returns:
        Optional[torch.Tensor]: The combined 4D causal and padding mask, or None.
    """
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    bsz, seq_len = input_shape
    # Only create causal mask if seq_len > 1 to avoid issues with single token generation
    if seq_len > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # Expand the passed padding mask (e.g., from decoder_inputs)
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=seq_len).to(
            inputs_embeds.device
        )
        # Combine masks: 0 means attend, -inf means mask. Additive combination works.
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )
        # Ensure no NaNs from adding large negative numbers
        combined_attention_mask = torch.nan_to_num(combined_attention_mask)


    return combined_attention_mask

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
) -> torch.Tensor:
    """
    Make causal mask used for bi-directional self-attention.
    Adapted from Hugging Face Transformers library.

    Args:
        input_ids_shape (torch.Size): Shape of input IDs (B, T).
        dtype (torch.dtype): Data type for the mask.
        device (torch.device): Device for the mask.
        past_key_values_length (int): Length of past key values.

    Returns:
        torch.Tensor: A 4D causal mask tensor.
    """
    bsz, tgt_len = input_ids_shape
    # Use float('-inf') directly for additive masking compatibility
    mask = torch.full((tgt_len, tgt_len), float('-inf'), device=device)
    # Create upper triangular matrix (including diagonal) filled with 0
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        # If using KV caching, expand mask to include past keys
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    # Expand to 4D: [B, 1, T, Total_T]
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`
    and converts it for additive masking (0 for attend, -inf for mask).
    Adapted from Hugging Face Transformers library.

    Args:
        mask (torch.Tensor): The 2D attention mask (1 for attend, 0 for mask).
        dtype (torch.dtype): Target data type for the mask.
        tgt_len (Optional[int]): Target length for the third dimension. Defaults to source length.

    Returns:
        torch.Tensor: The expanded 4D mask ready for additive attention.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    # Expand to [B, 1, T, S]
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    # Invert the mask: 1 becomes 0, 0 becomes 1
    inverted_mask = 1.0 - expanded_mask

    # Fill inverted mask with float('-inf') where the inverted value is 1 (original 0)
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), float('-inf'))


@register_generate(name="transformer")
class TransformerTemporalModel(AutoregressiveMixin, MultiStepMixin, BaseTemporalModel):
    """Concrete implementation of a transformer-based temporal model,
    providing both autoregressive and direct multi-step generation methods."""

    def __init__(
        self,
        config,
        encoder=None,
        decoder=None,
        output_heads=None,
        head_aggregator=None,
        loss_fn=None,
    ):
        """Initialize the TransformerTemporalModel.

        Args:
            config: Configuration instance for the model.
            encoder (nn.Module, optional): Encoder network.
            decoder (nn.Module, optional): Decoder network.
            output_heads (nn.Module, optional): One or more output head modules.
            head_aggregator (callable, optional): Function/module to combine output heads.
            loss_fn (callable, optional): Loss function to use during training.
        """
        super().__init__(
            config=config,
            encoder=encoder,
            decoder=decoder,
            output_heads=output_heads,
            head_aggregator=head_aggregator,
            loss_fn=loss_fn,
        )
        # Store encoder/decoder dtype if possible for mask creation
        self._encoder_dtype = getattr(encoder, 'dtype', torch.float32) if encoder else torch.float32
        self._decoder_dtype = getattr(decoder, 'dtype', torch.float32) if decoder else torch.float32


    def forward(
        self,
        encoder_inputs: Optional[torch.Tensor] = None,
        decoder_inputs: Optional[torch.Tensor] = None,
        # --- Accept 2D masks as input ---
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        # --- End mask input ---
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, # Added for generation/caching
        use_cache: Optional[bool] = None, # Added for generation/caching
        output_attentions: Optional[bool] = None, # Added for introspection
        output_hidden_states: Optional[bool] = None, # Added for introspection
        # **kwargs removed to promote explicit argument passing **
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Perform a forward pass through the transformer model.

        Handles encoder-decoder and decoder-only architectures.
        Processes attention masks internally from 2D input masks.

        Args:
            encoder_inputs (torch.Tensor, optional): Inputs fed to the encoder (e.g., past_values). Shape [B, L_enc, F_enc].
            decoder_inputs (torch.Tensor, optional): Inputs fed to the decoder (e.g., future_values shifted). Shape [B, L_dec, F_dec]. Mandatory if decoder exists.
            attention_mask (torch.Tensor, optional): 2D mask for encoder_inputs [B, L_enc], 1 indicates valid token, 0 padding.
            decoder_attention_mask (torch.Tensor, optional): 2D mask for decoder_inputs [B, L_dec], 1 indicates valid token, 0 padding.
            targets (torch.Tensor, optional): Target values for loss calculation. Shape [B, L_dec, F_out].
            past_key_values (Tuple[Tuple[torch.Tensor]], optional): Cached key/values for faster decoding.
            use_cache (bool, optional): Whether to return key/values for caching.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return hidden states for all layers.

        Returns:
            Dict[str, Optional[torch.Tensor]]: A dictionary containing outputs like 'logits', 'loss', 'past_key_values', etc.
        """
        # --- Prepare specific arguments for internal modules ---
        # Filter arguments relevant ONLY for the encoder
        encoder_call_kwargs = {
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        # Filter arguments relevant ONLY for the decoder
        decoder_call_kwargs = {
            "use_cache": use_cache,
            "past_key_values": past_key_values,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        # --- End argument prep ---

        encoder_output = None
        processed_encoder_mask = None
        processed_decoder_mask = None
        processed_cross_mask = None

        # 1. Process Encoder Mask (if encoder exists and mask provided)
        if self.encoder and attention_mask is not None:
            if encoder_inputs is None:
                raise ValueError("attention_mask provided but encoder_inputs is None.")
            # Expand encoder mask to 4D: [B, 1, L_enc, L_enc]
            processed_encoder_mask = _expand_mask(attention_mask, dtype=self._encoder_dtype)

        # 2. Run Encoder if it exists
        if self.encoder:
            if encoder_inputs is None:
                 raise ValueError("Encoder exists but encoder_inputs are None.")
            # *** Corrected: Pass ONLY relevant arguments to encoder ***
            encoder_output = self.encoder(
                input_values=encoder_inputs, # Assuming internal encoder uses 'input_values'
                attention_mask=processed_encoder_mask,
                **encoder_call_kwargs # Pass filtered kwargs
            )

        # 3. Prepare Decoder Masks and Inputs
        decoder_output = None
        input_to_heads = None
        encoder_hidden_states = None # For cross-attention

        if self.decoder:
            # Extract encoder hidden states if encoder ran
            if encoder_output is not None:
                 encoder_hidden_states = encoder_output.get('last_hidden_state') if isinstance(encoder_output, dict) else encoder_output

            # Decoder requires decoder_inputs
            if decoder_inputs is None:
                raise ValueError("Decoder exists but decoder_inputs are None.")

            # --- Process Decoder Masks ---
            decoder_input_shape = decoder_inputs.shape[:-1] # Get [B, L_dec]
            past_kv_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

            # Create 4D causal mask combined with padding mask for decoder self-attention
            processed_decoder_mask = _prepare_decoder_attention_mask(
                 decoder_attention_mask, decoder_input_shape, decoder_inputs, past_kv_length
            )

            # Create 4D mask for cross-attention (using encoder's padding mask)
            if encoder_hidden_states is not None and attention_mask is not None:
                processed_cross_mask = _expand_mask(
                    attention_mask, dtype=self._decoder_dtype, tgt_len=decoder_input_shape[-1]
                )
            # --- End Mask Processing ---

            # 4. Run Decoder
            # *** Corrected: Pass ONLY relevant arguments to decoder ***
            decoder_output = self.decoder(
                input_ids=decoder_inputs, # Assuming internal decoder uses 'input_ids'
                attention_mask=processed_decoder_mask, # Self-attention mask
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=processed_cross_mask, # Cross-attention mask
                **decoder_call_kwargs # Pass filtered kwargs
            )
            input_to_heads = decoder_output.get('last_hidden_state') if isinstance(decoder_output, dict) else decoder_output

        else: # No Decoder case
            if encoder_output is not None:
                 input_to_heads = encoder_output.get('last_hidden_state') if isinstance(encoder_output, dict) else encoder_output

        # 5. Ensure there's some output to feed to the heads
        if input_to_heads is None:
             raise ValueError("Model configuration seems incomplete or invalid. No output generated for heads.")

        # 6. Pass through Output Heads
        # Heads generally only need the final hidden states
        logits = self.output_heads(input_to_heads)

        # 7. Use Head Aggregator if it exists
        if self.head_aggregator is not None:
             logits = self.head_aggregator(logits) # Apply aggregator

        # 8. Calculate Loss
        loss = None
        if targets is not None:
            if self.loss_fn is None:
                raise ValueError("Loss calculation requires a loss_fn, but it's None.")
            loss = self.loss_fn(logits, targets)

        # 9. Prepare final output dictionary
        final_output = {
            "logits": logits,
            "loss": loss,
        }
        # Add optional outputs
        if output_hidden_states:
            if encoder_output and isinstance(encoder_output, dict):
                final_output["encoder_hidden_states"] = encoder_output.get("hidden_states")
            if decoder_output and isinstance(decoder_output, dict):
                final_output["decoder_hidden_states"] = decoder_output.get("hidden_states")
        if output_attentions:
             if encoder_output and isinstance(encoder_output, dict):
                 final_output["encoder_attentions"] = encoder_output.get("attentions")
             if decoder_output and isinstance(decoder_output, dict):
                 final_output["decoder_attentions"] = decoder_output.get("attentions") # Assuming self-attentions
                 final_output["decoder_cross_attentions"] = decoder_output.get("cross_attentions") # If decoder output includes these
        if use_cache:
             if decoder_output and isinstance(decoder_output, dict):
                 final_output["past_key_values"] = decoder_output.get("past_key_values")

        return final_output

    # Mixin methods (generate, generate_multistep) are inherited

