import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from temporal.models.base_model import BaseTemporalModel
from temporal.registry.generate import register_generate
# Import the mixins
from temporal.models.mixin.autoregressive import AutoregressiveMixin
from temporal.models.mixin.multistep import MultiStepMixin # Assuming this is the correct name

@register_generate(name="transformer")  # Registering with the name "transformer"
# Inherit from both mixins and BaseTemporalModel
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
        # Call the BaseTemporalModel's __init__ first.
        # Ensure mixins don't have conflicting __init__ requirements or call them appropriately if they do.
        super().__init__(
            config=config,
            encoder=encoder,
            decoder=decoder,
            output_heads=output_heads,
            head_aggregator=head_aggregator,
            loss_fn=loss_fn,
        )

    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_inputs: Optional[torch.Tensor] = None, # Often named past_values for decoder-only
        targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Perform a forward pass through the transformer model.

        Handles encoder-decoder and decoder-only architectures.

        Args:
            encoder_inputs (torch.Tensor): Inputs fed to the encoder (used if self.encoder exists).
                                          For decoder-only, this might be None or unused.
            decoder_inputs (torch.Tensor, optional): Inputs fed to the decoder.
                                                  For encoder-decoder, this is the target sequence (shifted).
                                                  For decoder-only, this is the primary input sequence.
            targets (torch.Tensor, optional): Target values for loss calculation.
            **kwargs: Additional arguments passed to encoder/decoder/heads.

        Returns:
            Dict[str, Optional[torch.Tensor]]: A dictionary containing outputs like 'logits' and 'loss'.
        """
        encoder_output = None
        # 1. Run Encoder if it exists
        if self.encoder:
            # Ensure encoder_inputs are provided if encoder exists
            if encoder_inputs is None:
                 raise ValueError("Encoder exists but encoder_inputs are None.")
            encoder_output = self.encoder(encoder_inputs, **kwargs)
            # encoder_output might be the raw output or a BaseModelOutput containing last_hidden_state

        # 2. Prepare Decoder Input and Run Decoder if it exists
        decoder_output = None
        input_to_heads = None

        if self.decoder:
            # Determine the actual hidden states from the encoder output if applicable
            encoder_hidden_states = None
            if encoder_output is not None:
                 # Check if encoder_output is an object with last_hidden_state (like HF models)
                 if hasattr(encoder_output, 'last_hidden_state'):
                      encoder_hidden_states = encoder_output.last_hidden_state
                 else:
                      encoder_hidden_states = encoder_output # Assume raw tensor output

            # Ensure decoder_inputs are provided
            if decoder_inputs is None:
                raise ValueError("Decoder exists but decoder_inputs are None.")

            # Call the decoder, passing encoder_hidden_states (which is None for decoder-only)
            decoder_output = self.decoder(
                # Pass appropriate inputs to the decoder
                # For Hugging Face compatibility, decoder input is often just 'input_ids'
                input_ids=decoder_inputs,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs # Pass other relevant kwargs like attention_mask, etc.
            )
            # Extract the hidden states to pass to the heads
            if hasattr(decoder_output, 'last_hidden_state'):
                 input_to_heads = decoder_output.last_hidden_state
            else:
                 input_to_heads = decoder_output # Assume raw tensor output
        else:
            # If no decoder, the encoder's output goes to the heads
            if encoder_output is not None:
                 if hasattr(encoder_output, 'last_hidden_state'):
                      input_to_heads = encoder_output.last_hidden_state
                 else:
                      input_to_heads = encoder_output
            # If neither encoder nor decoder exist, input_to_heads remains None

        # 3. Ensure there's some output to feed to the heads
        if input_to_heads is None:
             raise ValueError("Neither encoder nor decoder produced output for the heads.")

        # 4. Pass through Output Heads
        logits = self.output_heads(input_to_heads, **kwargs)

        # 5. Use Head Aggregator if it exists
        if self.head_aggregator is not None:
             logits = self.head_aggregator(logits) # Apply aggregator

        # 6. Calculate Loss
        loss = None
        if targets is not None:
            if self.loss_fn is None:
                raise ValueError("Loss calculation requires a loss_fn, but it's None.")
            loss = self.loss_fn(logits, targets)

        # 7. Return results (include encoder_output even if None)
        return {
            "logits": logits,
            "loss": loss,
            "encoder_output": encoder_output, # Could be None, raw tensor, or BaseModelOutput
            "decoder_output": decoder_output  # Could be None, raw tensor, or BaseModelOutput
        }

    # No need to define generate or generate_multistep here,
    # they are inherited from the mixins.

