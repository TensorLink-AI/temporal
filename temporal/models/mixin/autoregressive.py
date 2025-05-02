import torch
import torch.nn as nn
from typing import Optional


class AutoregressiveMixin:
    def enable_dropout(self):
        """Enable dropout for MC sampling during autoregressive generation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def generate_autoregressive(
        self,
        input_ids: torch.Tensor,
        prediction_length: int,
        attention_mask: Optional[torch.Tensor] = None, # Mask for encoder input
        decoder_attention_mask: Optional[torch.Tensor] = None, # Optional separate mask for decoder start
        use_cache: bool = True,
        decoder_start_token_id: Optional[float] = None, # Changed type hint to float
        eos_token_id: Optional[float] = None,           # Changed type hint to float
        early_stopping: bool = False,
        output_attentions: bool = False,
        # feedback_strategy: str = "raw",  # Options: "raw", "mean", "first", "sample"
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive generation, supporting encoder-decoder and decoder-only.

        Args:
            input_ids: Inputs. Shape depends on architecture:
                       - Encoder-Decoder: Encoder input sequence [B, Seq_Enc, Feat_Enc].
                       - Decoder-Only: Initial decoder input sequence [B, Seq_Dec, Feat_Dec].
            prediction_length: Number of steps to generate.
            attention_mask: Mask for encoder inputs (if encoder exists).
            decoder_attention_mask: Mask for initial decoder inputs (if needed).
            use_cache: Whether to use KV caching for the decoder.
            decoder_start_token_id: Value to initialize the first decoder step (optional).
            eos_token_id: Value indicating end of sequence (optional, for early stopping).
            early_stopping: Stop generation if eos_token is predicted.
            output_attentions: Whether to output attention weights.
            **kwargs: Additional arguments passed to encoder/decoder.

        Returns:
            Tensor of generated sequence, shape [B, prediction_length, Feat_Dec].
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1) Prepare Encoder Output (if encoder exists)
        encoder_hidden_states = None
        if hasattr(self, 'encoder') and self.encoder is not None:
            if input_ids is None:
                 raise ValueError("Encoder exists but input_ids are None.")
            encoder_outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                return_dict=True,
                **kwargs.get("encoder_kwargs", {}), # Pass specific encoder args
            )
            if hasattr(encoder_outputs, 'last_hidden_state'):
                encoder_hidden_states = encoder_outputs.last_hidden_state
            else:
                encoder_hidden_states = encoder_outputs # Assume raw tensor
        else:
            # If decoder-only, input_ids are actually the initial decoder inputs
            pass # encoder_hidden_states remains None

        # 2) Initialize Decoder Input Sequence
        #    For encoder-decoder, usually start with a start token.
        #    For decoder-only, start with the provided input_ids.

        # Get the effective start token ID, either from args or config
        effective_start_token_id = decoder_start_token_id
        if hasattr(self, 'encoder') and self.encoder is not None and effective_start_token_id is None:
             # Check config only if encoder exists and start token wasn't passed in args
             if hasattr(self.config, "decoder_start_token_id") and self.config.decoder_start_token_id is not None:
                  effective_start_token_id = self.config.decoder_start_token_id
             else:
                  # If encoder exists but no start token provided via args or config, raise error
                  raise ValueError("Encoder-decoder generation requires a decoder_start_token_id via argument or config.")


        if hasattr(self, 'encoder') and self.encoder is not None:
             # Encoder-Decoder Path
             if effective_start_token_id is None:
                 # This condition should now be unreachable due to the check above, but keep for safety
                 raise ValueError("Encoder-decoder effective_start_token_id is None unexpectedly.")
             
             # Get model dimension, preferring config.hidden_size
             model_dim = getattr(self.config, 'hidden_size', None) # Use hidden_size first
             if model_dim is None:
                 # Fallback to d_model if hidden_size isn't present (legacy)
                 model_dim = getattr(self.config, 'd_model', None)
             if model_dim is None and hasattr(self, 'decoder') and self.decoder and hasattr(self.decoder, 'config'):
                  # Fallback to decoder's config if main config fails
                  model_dim = getattr(self.decoder.config, 'hidden_size', getattr(self.decoder.config, 'd_model', None))
             if model_dim is None:
                  # Last resort if dimension cannot be inferred
                  raise AttributeError("Could not determine model dimension (hidden_size/d_model) from config or decoder.")

             # Start with a single token/step
             # The feature dimension of the *input* to the decoder should match the model dimension
             decoder_input_ids = torch.full(
                 (batch_size, 1, model_dim), # Shape [B, 1, ModelDim]
                 float(effective_start_token_id), # Use float for potential feature values
                 dtype=torch.float32 if encoder_hidden_states is None else encoder_hidden_states.dtype,
                 device=device,
             )
             current_seq_len = 1
        
        elif not (hasattr(self, 'encoder') and self.encoder is not None):
             # Decoder-only: the input_ids are the initial sequence
             decoder_input_ids = input_ids
             current_seq_len = decoder_input_ids.shape[1]
        # else: # This case is now handled by the check for effective_start_token_id
        #     pass 

        predictions = []
        past_key_values = None
        internal_decoder_attention_mask = decoder_attention_mask # Start with provided mask

        # 3) Autoregressive Loop
        for step in range(prediction_length):
            # Prepare decoder inputs for this step
            # Use only the last token/step for input if using cache
            step_input_ids = decoder_input_ids[:, -1:, :] if use_cache and past_key_values is not None else decoder_input_ids

            # Prepare attention mask for decoder self-attention
            # This mask needs to grow with the sequence length unless using cache
            if not use_cache or past_key_values is None:
                 step_attention_mask = internal_decoder_attention_mask
            else:
                 step_attention_mask = None # Relying on cache mechanism or decoder internal handling

            if not hasattr(self, 'decoder') or self.decoder is None:
                raise AttributeError("Model must have a 'decoder' attribute for autoregressive generation.")

            decoder_outputs = self.decoder(
                input_ids=step_input_ids, # Use last token if cache enabled
                encoder_hidden_states=encoder_hidden_states, # Might be None
                encoder_attention_mask=attention_mask,
                attention_mask=step_attention_mask, # Decoder self-attention mask
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=True,
                **kwargs.get("decoder_kwargs", {}), # Pass specific decoder args
            )

            # Extract hidden state for the *last* generated token
            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]

            # Pass through output heads
            if not hasattr(self, 'output_heads'):
                 raise AttributeError("Model must have 'output_heads' for generation.")

            if isinstance(self.output_heads, nn.ModuleList):
                head_outputs = [head(last_hidden) for head in self.output_heads]
                if hasattr(self, 'head_aggregator') and self.head_aggregator:
                    next_pred_features = self.head_aggregator(head_outputs)
                elif head_outputs:
                    next_pred_features = head_outputs[0] # Fallback
                else:
                    raise ValueError("Output heads list is empty.")
            else:
                next_pred_features = self.output_heads(last_hidden)
            
            # next_pred_features shape: [B, 1, OutputFeatures]
            predictions.append(next_pred_features)

            # --- Prepare input for the next step --- 
            # The input to the *next* decoder step depends on whether the output head's feature size
            # matches the decoder's expected input dimension (model_dim/hidden_size).
            # If they match, we can feed the prediction directly.
            # If they don't match (common case, output head projects to final feature count),
            # we need to re-embed the prediction or use some other strategy.

            # Get model dimension again (needed for comparison)
            model_dim = getattr(self.config, 'hidden_size', getattr(self.config, 'd_model', None))
            if model_dim is None and hasattr(self, 'decoder') and self.decoder and hasattr(self.decoder, 'config'):
                  model_dim = getattr(self.decoder.config, 'hidden_size', getattr(self.decoder.config, 'd_model', None))
            if model_dim is None:
                  raise AttributeError("Could not determine model dimension (hidden_size/d_model) for next step input prep.")

            output_feature_size = next_pred_features.shape[-1]

            if output_feature_size == model_dim:
                # Output dimension matches model dimension, feed directly
                next_decoder_input_step = next_pred_features
            else:
                 # Output dimension differs. Need to re-embed or project.
                 # Simple strategy: Use the model's value embedding layer if it exists.
                 if hasattr(self.decoder, 'value_embedding'):
                     # Requires value_embedding input dim to match output_feature_size
                     try:
                         next_decoder_input_step = self.decoder.value_embedding(next_pred_features)
                     except RuntimeError as e:
                         # Catch dimension mismatch errors from linear layer
                         raise RuntimeError(f"Output feature size ({output_feature_size}) likely mismatch with \
                                             decoder value_embedding input size. Error: {e}")
                 else:
                     # Fallback: Create a dummy input or raise error
                     # This part is model-specific and might need adjustment.
                     # For now, raise error as we don't know how to get back to model_dim.
                     raise NotImplementedError(
                         f"Output feature size ({output_feature_size}) differs from model dimension ({model_dim}), "
                         f"and no decoder.value_embedding found to re-embed. Autoregressive loop cannot proceed." 
                         f" Implement a re-embedding strategy."
                     )

            # Concatenate the prepared next input step
            decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_step], dim=1)
            current_seq_len += 1
            # --- End next step input prep --- 

            # Update attention mask if necessary (only relevant if not fully relying on cache)
            if internal_decoder_attention_mask is not None and not use_cache:
                 try:
                     # This assumes mask shape [B, Seq] - needs adjustment for [B, 1, T, S]
                     new_mask_column = torch.ones((batch_size, 1), dtype=internal_decoder_attention_mask.dtype, device=device)
                     internal_decoder_attention_mask = torch.cat([internal_decoder_attention_mask, new_mask_column], dim=1)
                 except Exception as e:
                     print(f"Warning: Could not update internal_decoder_attention_mask: {e}")
                     pass

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            # Check for early stopping
            if early_stopping and eos_token_id is not None:
                try:
                    # Compare the first feature of the prediction tensor
                    if torch.isclose(next_pred_features[:, :, 0], torch.tensor(float(eos_token_id), device=device)).all():
                        break
                except IndexError:
                     print("Warning: Could not check eos_token_id, prediction tensor shape might be unexpected.")

        if not predictions:
             # Handle case where prediction_length was 0 or loop exited early
             output_feature_size = 1 # Default
             if hasattr(self, 'output_heads'):
                 try:
                     out_head = self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads
                     # Attempt to infer from head's output size if available
                     if hasattr(out_head, 'output_size'): output_feature_size = out_head.output_size
                     elif hasattr(out_head, 'out_features'): output_feature_size = out_head.out_features
                     elif hasattr(out_head, 'decoder') and hasattr(out_head.decoder, 'out_features'): output_feature_size = out_head.decoder.out_features
                     # Add more checks if needed based on head structure
                 except (AttributeError, IndexError, TypeError):
                     pass 
             return torch.empty((batch_size, 0, output_feature_size), device=device)

        return torch.cat(predictions, dim=1) # Shape [B, prediction_length, OutputFeatures]
