import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Any, Tuple, Callable, Dict, List

logger = logging.getLogger(__name__)

class AutoregressivePatchMixin:
    """
    A mixin class for autoregressive generation capabilities in neural network models.

    The primary `generate` method is designed for modern, patch-based models
    that use a "generate-then-merge" paradigm. It assumes the model's preprocessor
    has a dedicated method for handling latent embeddings during generation.

    The class also retains legacy helper methods for traditional, step-by-step
    autoregressive generation.
    """

    def enable_dropout(self):
        """Enable dropout for MC sampling during autoregressive generation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    # ... [ _get_scalar_value, _get_head_output, and legacy helpers remain unchanged ] ...

    def _get_scalar_value(self, value: Union[torch.Tensor, float, int, Any], name: str) -> Optional[float]:
        """
        Safely converts a potential tensor value to a float scalar.
        """
        if value is None:
            return None
        if torch.is_tensor(value):
            temp_value = value
            while temp_value.numel() > 1:
                logger.warning(f"Tensor for '{name}' had {temp_value.numel()} elements. Taking the first element.")
                temp_value = temp_value[0]
            if temp_value.numel() == 1:
                return float(temp_value.item())
            else:
                raise ValueError(f"Could not reduce '{name}' tensor (original shape {value.shape}) to a scalar.")
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Could not convert '{name}'={value} (type {type(value)}) to float scalar. Error: {e}")

    def _get_head_output(self, head_input: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Applies final output head(s) to the processed model output.
        """
        if not hasattr(self, 'output_heads'):
            raise AttributeError("Model is missing output_heads, which is required for generation.")

        if isinstance(self.output_heads, nn.ModuleList):
            return [head(head_input) for head in self.output_heads]
        else:
            return self.output_heads(head_input)

    @torch.no_grad()
    def generate(
        self,
        encoder_inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        num_patches_to_generate: int = 0,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, List[Dict[str, Union[torch.Tensor, List[str]]]]]:
        """
        Autoregressively generates a sequence of features, correctly designed for patch-based models.

        This process assumes the model's preprocessor has two distinct functions:
        1. `process()`: For handling the initial raw `encoder_inputs`.
        2. `_prepare_decoder_inputs_for_generation()`: For handling latent patch embeddings
           inside the autoregressive loop (adds positional encoding and creates causal masks).

        Args:
            encoder_inputs (torch.Tensor): The initial context sequence of raw time-series data.
            attention_mask (Optional[torch.Tensor]): Mask for encoder attention.
            decoder_attention_mask (Optional[torch.Tensor]): Pre-computed mask for the decoder's
                self-attention. Typically not needed when using `use_cache=True`.
            num_patches_to_generate (int): The number of future *patches* to generate.
            use_cache (bool): Whether to use past key values for faster decoding.
            output_attentions (bool): Whether to return attentions from the model.
            output_hidden_states (bool): Whether to return hidden states from the model.
            **kwargs: Additional arguments passed to the encoder/decoder.

        Returns:
            The generated sequence of predicted features.
        """
        self.eval()

        # --- Attribute validation ---
        required_attrs = ['preprocessor', 'encoder', 'decoder', 'patch_merger', 'output_heads']
        if not all(hasattr(self, attr) for attr in required_attrs):
             raise AttributeError(f"Model must have {required_attrs} attributes for patch-based generation.")
        if not hasattr(self.preprocessor, '_prepare_decoder_inputs_for_generation'):
            raise AttributeError("Preprocessor must have a '_prepare_decoder_inputs_for_generation' method.")

        # --- Step 1: Process initial context and run the encoder ---
        processed_encoder = self.preprocessor.process(
            input_values=encoder_inputs,
            attention_mask=attention_mask,
            is_causal=False,
        )
        context_patches = processed_encoder["hidden_states"]
        
        encoder_outputs = self.encoder(
            hidden_states=context_patches,
            attention_mask=processed_encoder["attention_mask"],
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # --- Step 2: Initialize the autoregressive loop ---
        decoder_sequence_patches = context_patches[:, -1:, :]
        generated_patches = []
        past_key_values = None

        # --- Step 3: Autoregressively generate patch embeddings ---
        for _ in range(num_patches_to_generate):
            input_patches_for_step = decoder_sequence_patches[:, -1:, :] if use_cache and past_key_values else decoder_sequence_patches
            past_kv_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            
            # Use the dedicated method for processing latent embeddings in the decoder.
            # This correctly adds positional information and creates causal masks without re-patching.
            processed_decoder = self.preprocessor._prepare_decoder_inputs_for_generation(
                patch_embeds=input_patches_for_step,
                attention_mask=decoder_attention_mask if not (use_cache and past_key_values) else None,
                past_key_values_length=past_kv_length,
                is_causal=True,
            )
            
            decoder_outputs = self.decoder(
                hidden_states=processed_decoder["hidden_states"],
                attention_mask=processed_decoder["attention_mask"],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=processed_encoder["attention_mask"],
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            next_patch_embedding = decoder_outputs.last_hidden_state[:, -1:, :]
            generated_patches.append(next_patch_embedding)
            decoder_sequence_patches = torch.cat([decoder_sequence_patches, next_patch_embedding], dim=1)
            
            if use_cache:
                past_key_values = decoder_outputs.past_key_values

        if not generated_patches:
            return torch.empty((encoder_inputs.shape[0], 0, encoder_inputs.shape[-1]), device=encoder_inputs.device)

        # --- Step 4: Merge generated patches into final predictions ---
        all_generated_patches = torch.cat(generated_patches, dim=1)
        x = all_generated_patches.transpose(1, 2)
        x = self.patch_merger(x)
        point_predictions = x.transpose(1, 2)

        # --- Step 5: Apply final output heads (e.g., for probabilistic forecasts) ---
        final_output = self._get_head_output(point_predictions)
        
        return final_output