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
        decoder_start_token_id: Optional[int] = None, # Typically used for text, maybe feature index/value?
        eos_token_id: Optional[int] = None,
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
        if hasattr(self, 'encoder') and self.encoder is not None and hasattr(self.config, "decoder_start_token_id") and decoder_start_token_id is None:
            decoder_start_token_id = self.config.decoder_start_token_id

        if hasattr(self, 'encoder') and self.encoder is not None and decoder_start_token_id is not None:
            # Assume feature size is accessible via config
            feature_size = getattr(self.config, 'd_model', self.decoder.config.d_model if hasattr(self, 'decoder') and self.decoder else 1)
            # Start with a single token/step
            decoder_input_ids = torch.full(
                (batch_size, 1, feature_size), # Shape [B, 1, Features]
                float(decoder_start_token_id), # Use float for potential feature values
                dtype=torch.float32 if encoder_hidden_states is None else encoder_hidden_states.dtype,
                device=device,
            )
            current_seq_len = 1
        elif not (hasattr(self, 'encoder') and self.encoder is not None):
             # Decoder-only: the input_ids are the initial sequence
             decoder_input_ids = input_ids
             current_seq_len = decoder_input_ids.shape[1]
        else: # Encoder-decoder but no explicit start token -> use last encoder input as first decoder input?
             # This part might need refinement based on specific model needs
             # Let's assume for now we need a start token if encoder exists
             raise ValueError("Encoder-decoder generation requires a decoder_start_token_id or similar initialization.")

        predictions = []
        past_key_values = None
        internal_decoder_attention_mask = decoder_attention_mask # Start with provided mask

        # 3) Autoregressive Loop
        for step in range(prediction_length):
            # Prepare decoder inputs for this step
            # Use only the last token for input if using cache
            step_input_ids = decoder_input_ids[:, -1:, :] if use_cache and past_key_values is not None else decoder_input_ids

            # Prepare attention mask for decoder self-attention
            # This mask needs to grow with the sequence length unless using cache
            if not use_cache or past_key_values is None:
                 # If not using cache, the mask applies to the whole current sequence
                 # You might need a causal mask here depending on the decoder implementation
                 # For simplicity, assume decoder handles causal masking internally if needed
                 # or use the provided decoder_attention_mask if it covers the full target length
                 step_attention_mask = internal_decoder_attention_mask
            else:
                 # If using cache, mask only needs to cover the current step + cached steps
                 # Often handled implicitly by past_key_values mechanism or requires specific shape
                 step_attention_mask = None # Relying on cache mechanism or decoder internal handling
                 # If step_attention_mask is needed, it should be shaped [B, 1, current_seq_len + step] approx

            if not hasattr(self, 'decoder') or self.decoder is None:
                raise AttributeError("Model must have a 'decoder' attribute for autoregressive generation.")

            decoder_outputs = self.decoder(
                input_ids=step_input_ids, # Use last token if cache enabled
                encoder_hidden_states=encoder_hidden_states, # Might be None
                # encoder_attention_mask should be derived from the original *encoder's* attention_mask
                # Pass the original attention_mask, decoder needs to handle its shape/use for cross-attn
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
                    next_pred = self.head_aggregator(head_outputs)
                elif head_outputs:
                    next_pred = head_outputs[0] # Fallback
                else:
                    raise ValueError("Output heads list is empty.")
            else:
                next_pred = self.output_heads(last_hidden)

            predictions.append(next_pred)

            # Prepare input for the next step
            decoder_input_ids = torch.cat([decoder_input_ids, next_pred], dim=1)
            current_seq_len += 1

            # Update attention mask if necessary (only relevant if not fully relying on cache)
            if internal_decoder_attention_mask is not None and not use_cache:
                 # Example: append a '1' for the newly generated token
                 # Mask shape needs careful handling depending on implementation (e.g., [B, Seq] or [B, 1, Seq, Seq])
                 # This simplistic update assumes a [B, Seq] mask
                 try:
                     new_mask_column = torch.ones((batch_size, 1), dtype=internal_decoder_attention_mask.dtype, device=device)
                     internal_decoder_attention_mask = torch.cat([internal_decoder_attention_mask, new_mask_column], dim=1)
                 except Exception as e:
                     print(f"Warning: Could not update internal_decoder_attention_mask: {e}")
                     # Potentially fall back or raise error if mask is critical
                     pass

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            # Check for early stopping
            if early_stopping and eos_token_id is not None:
                # Compare the *last predicted feature* (or a specific one if needed)
                # Using float comparison might require tolerance
                try:
                    # Assuming eos_token_id applies to the first feature
                    if torch.isclose(next_pred[:, :, 0], torch.tensor(float(eos_token_id), device=device)).all():
                        break
                except IndexError:
                     print("Warning: Could not check eos_token_id, prediction tensor shape might be unexpected.")

        if not predictions:
            # Handle case where prediction_length was 0 or loop exited early
            # Need to return a tensor with correct shape [B, 0, Feat_Dec]
            # Infer feature size from somewhere (e.g., output head)
            output_feature_size = 1 # Default or placeholder
            if hasattr(self, 'output_heads'):
                try:
                    # Attempt to get output size from the first head or the single head
                    out_head = self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads
                    # This requires the head to have an identifiable output feature attribute (e.g., out_features)
                    # Or run a dummy forward pass if necessary (less ideal)
                    if hasattr(out_head, 'out_features'):
                         output_feature_size = out_head.out_features
                    elif hasattr(out_head, 'decoder') and hasattr(out_head.decoder, 'out_features'): # Common in HF heads
                         output_feature_size = out_head.decoder.out_features
                    # Add more checks if needed based on head structure
                except (AttributeError, IndexError):
                     pass # Keep default
            return torch.empty((batch_size, 0, output_feature_size), device=device)

        return torch.cat(predictions, dim=1) # Shape [B, prediction_length, Feat_Dec]
