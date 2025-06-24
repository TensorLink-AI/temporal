import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, asdict

from temporal.models.base_model import BaseTemporalModel
from temporal.registry.generate import register_generate
from temporal.models.mixin.autoregressive import AutoregressiveMixin
from temporal.models.mixin.multistep import MultiStepMixin
from temporal.models.preprocessor import InputPreprocessor
from temporal.models.module_builder_helper import ModuleBuilder

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
        aux_loss (Optional[torch.FloatTensor]): The auxiliary loss, e.g. from MoE layers.
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
    aux_loss: Optional[torch.FloatTensor] = None
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
        loss_fn: Optional[callable] = None,
        builder: Optional[ModuleBuilder] = None,
    ):
        """
        Initializes the TransformerTemporalModel in an order that matches the
        forward pass for clearer model summaries.
        """
        # We manually initialize the base class and then assign modules in the
        # desired order for printing.
        super(BaseTemporalModel, self).__init__()
        self.config = config

        if builder is None:
            builder = ModuleBuilder(config)

        # 1. Preprocessor is the first step in the forward pass.
        self.preprocessor = InputPreprocessor(config, builder)

        # 2. Encoder runs second.
        self.encoder = encoder

        # 3. Decoder runs third.
        self.decoder = decoder

        # 4. Output heads run last.
        self.output_heads = output_heads
        self.head_aggregator = head_aggregator
        
        # 5. Loss function is not a module, but we assign it here.
        self.loss_fn = loss_fn

        # Store dtypes for casting if necessary
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
        validate_shapes: bool = False,
        verbose: bool = False,
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
            validate_shapes (bool): If True, performs assertions inside the
                preprocessor to check for shape consistency.
            verbose (bool): If True, prints detailed shape information from
                the preprocessor for debugging.

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
            
            processed_encoder = self.preprocessor.process(
                input_values=encoder_inputs,
                attention_mask=attention_mask,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )
            encoder_outputs = self.encoder(
                hidden_states=processed_encoder["hidden_states"],
                attention_mask=processed_encoder["attention_mask"],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        # Step 2: Run the decoder.
        encoder_hidden_states = encoder_outputs.last_hidden_state if encoder_outputs else None
        
        if self.decoder:
            if decoder_inputs is None:
                raise ValueError("The model's decoder requires 'decoder_inputs'.")
            
            past_kv_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

            processed_decoder = self.preprocessor.process(
                input_values=decoder_inputs,
                past_key_values_length=past_kv_length,
                attention_mask=decoder_attention_mask,
                validate_shapes=validate_shapes,
                verbose=verbose,
            )

            decoder_outputs = self.decoder(
                hidden_states=processed_decoder["hidden_states"],
                attention_mask=processed_decoder["attention_mask"],
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            input_to_heads = decoder_outputs.last_hidden_state
        else:
            if encoder_hidden_states is None:
                raise ValueError("Model requires an encoder or decoder to produce output for the head.")
            input_to_heads = encoder_hidden_states
            decoder_outputs = None

        # Step 3: Align head input with targets for loss calculation if needed.
        if targets is not None and self.config.architecture.layout == "decoder":
            num_target_steps = targets.size(1)
            input_to_heads = input_to_heads[:, -num_target_steps:, :]
        
        # Step 4: Project the final hidden states through the output head(s).
        logits = self.output_heads(input_to_heads)
        if self.head_aggregator is not None:
            logits = self.head_aggregator(logits)

        # Step 5: Calculate the loss if targets are provided.
        loss = None
        total_aux_loss = None
        if encoder_outputs and hasattr(encoder_outputs, 'aux_loss'):
            total_aux_loss = encoder_outputs.aux_loss
        if decoder_outputs and hasattr(decoder_outputs, 'aux_loss'):
            if total_aux_loss is None:
                total_aux_loss = decoder_outputs.aux_loss
            else:
                total_aux_loss += decoder_outputs.aux_loss

        if targets is not None:
            if self.loss_fn is None:
                raise ValueError("Loss calculation requires a 'loss_fn' to be set on the model.")
            loss = self.loss_fn(logits, targets)
            if total_aux_loss is not None:
                loss += self.config.aux_loss_weight * total_aux_loss

        # Step 6: Construct and return the final output object.
        return TransformerOutput(
            loss=loss,
            logits=logits,
            aux_loss=total_aux_loss,
            past_key_values=decoder_outputs.past_key_values if decoder_outputs else None,
            decoder_hidden_states=decoder_outputs.hidden_states if decoder_outputs else None,
            decoder_attentions=decoder_outputs.attentions if decoder_outputs else None,
            decoder_cross_attentions=decoder_outputs.cross_attentions if decoder_outputs else None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if encoder_outputs else None,
            encoder_hidden_states=encoder_outputs.hidden_states if encoder_outputs else None,
            encoder_attentions=encoder_outputs.attentions if encoder_outputs else None,
        )
