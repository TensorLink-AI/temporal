import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, asdict

from temporal.models.base_model import BaseTemporalModel
from temporal.registry.generate import register_generate
from temporal.models.mixin.autoregressive import AutoregressiveDispatchMixin
from temporal.models.mixin.autoregressive_patch import AutoregressivePatchMixin
from temporal.models.mixin.autoregressive_stepwise import AutoregressiveStepwiseMixin
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
class TransformerTemporalModel(AutoregressiveDispatchMixin,AutoregressivePatchMixin,AutoregressiveStepwiseMixin, MultiStepMixin, BaseTemporalModel):
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

        self.patch_merger = None # Start as None
        #  partion this out eventually to make it cleaner
        if self.preprocessor.is_patched:
            patch_size = self.preprocessor.patch_size
            use_mlp        =  getattr(self.preprocessor.value_embedding, 'use_mlp', False) 
            num_patches = config.context_length // patch_size
            mlp_hidden_size    = getattr(self.preprocessor.value_embedding, 'mlp_hidden_size',None) or (patch_size  * 2)
            d_model = self.config.d_model
            feature_size = self.config.feature_size
            output_projection_size = patch_size * feature_size

            if use_mlp:
                print(f"INFO: Building MLP patch_merger (d_model -> {mlp_hidden_size} -> {output_projection_size}).")
                self.patch_merger = nn.Sequential(
                    nn.Linear(d_model, mlp_hidden_size),
                    nn.ReLU(),
                    nn.Linear(mlp_hidden_size, output_projection_size)
                )
            else:
                print(f"INFO: Building Linear patch_merger (d_model -> {output_projection_size}).")
                self.patch_merger = nn.Linear(d_model, output_projection_size)
        
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
        loss_mask: Optional[torch.Tensor] = None,
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
            loss_mask (Optional[torch.Tensor]): An optional mask to apply to the loss
                calculation. Its shape should be broadcastable to the shape of `logits`.
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
                is_causal=False,  # Encoders are never causal
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
                is_causal=True, # Decoders are always causal
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


        # CORRECTED Step 4: Apply patch merger BEFORE the output head.
        # ─── Step 4: Merge or Expand Patch Tokens ───
        if self.preprocessor.is_patched:
            

            projected_patches = self.patch_merger(input_to_heads)
            B, P, _ = projected_patches.shape
            p_sz = self.preprocessor.patch_size
            f_sz = self.config.feature_size

            # The .view() operation flattens the patch dimension back into a time sequence.
            # [B, P, patch_size * F] -> [B, P * patch_size, F]
            input_to_heads = projected_patches.view(B, P * p_sz, f_sz)

        # Step 3: Align head input with targets for loss calculation if needed.
        if (
            targets is not None and 
            self.config.architecture.layout == "decoder" 
        ):
            num_target_steps = targets.size(1)

            # This check is crucial for catching data pipeline issues
            if input_to_heads.shape[1] < num_target_steps:
                 raise ValueError(
                     f"Input to heads ({input_to_heads.shape[1]} steps) is shorter than targets ({num_target_steps} steps). "
                     f"Cannot align for loss calculation. Ensure your decoder_inputs or model's effective output length in 'forward' covers your targets."
                 )
            input_to_heads = input_to_heads[:, -num_target_steps:, :]
        
        # Step 5: Project the final hidden states through the output head(s).
        logits = self.output_heads(input_to_heads)
        if self.head_aggregator is not None:
            logits = self.head_aggregator(logits)

        # Step 6: Calculate the loss if targets are provided.
        loss = None
        total_aux_loss = None
        if encoder_outputs and hasattr(encoder_outputs, 'aux_loss') and encoder_outputs.aux_loss is not None:
            total_aux_loss = encoder_outputs.aux_loss
        if decoder_outputs and hasattr(decoder_outputs, 'aux_loss') and decoder_outputs.aux_loss is not None:
            if total_aux_loss is None:
                total_aux_loss = decoder_outputs.aux_loss
            else:
                total_aux_loss += decoder_outputs.aux_loss

        if targets is not None:
            if self.loss_fn is None:
                raise ValueError("Loss calculation requires a 'loss_fn' to be set on the model.")
            
            loss = self.loss_fn(preds=logits, targets=targets, loss_mask=loss_mask)

            if total_aux_loss is not None:
                # Ensure aux loss is a scalar before adding
                loss += self.config.aux_loss_weight * total_aux_loss.mean()

        # Step 7: Construct and return the final output object.
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
