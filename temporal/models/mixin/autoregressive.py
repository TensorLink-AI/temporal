import torch
import torch.nn as nn
from typing import Optional, Union, Any # Added Union, Any


class AutoregressiveMixin:
    def enable_dropout(self):
        """Enable dropout for MC sampling during autoregressive generation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _get_scalar_value(self, value: Union[torch.Tensor, float, int, Any], name: str) -> float:
        """ Safely converts a potential tensor value to a float scalar. """
        if torch.is_tensor(value):
            if value.numel() == 1:
                return float(value.item())
            else:
                # This case is usually unexpected for start/eos tokens
                print(f"Warning: {name} was a tensor with {value.numel()} elements. Using first element.")
                return float(value[0].item())
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Could not convert {name}={value} (type {type(value)}) to float scalar. Error: {e}")


    def generate_autoregressive(
        self,
        input_ids: torch.Tensor,
        prediction_length: int,
        attention_mask: Optional[torch.Tensor] = None, # Mask for encoder input
        decoder_attention_mask: Optional[torch.Tensor] = None, # Optional separate mask for decoder start
        use_cache: bool = True,
        decoder_start_token_id: Optional[Any] = None, # Allow Tensor or scalar
        eos_token_id: Optional[Any] = None,           # Allow Tensor or scalar
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
            decoder_start_token_id: Value (scalar or tensor) to initialize the first decoder step (optional).
            eos_token_id: Value (scalar or tensor) indicating end of sequence (optional, for early stopping).
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
            pass # encoder_hidden_states remains None

        # 2) Initialize Decoder Input Sequence
        effective_start_token_id = decoder_start_token_id
        if hasattr(self, 'encoder') and self.encoder is not None and effective_start_token_id is None:
             if hasattr(self.config, "decoder_start_token_id") and self.config.decoder_start_token_id is not None:
                  effective_start_token_id = self.config.decoder_start_token_id
             else:
                  raise ValueError("Encoder-decoder generation requires a decoder_start_token_id via argument or config.")

        if hasattr(self, 'encoder') and self.encoder is not None:
             # Encoder-Decoder Path
             if effective_start_token_id is None:
                 raise ValueError("Encoder-decoder effective_start_token_id is None unexpectedly.")
             
             model_dim = getattr(self.config, 'hidden_size', None)
             if model_dim is None: model_dim = getattr(self.config, 'd_model', None)
             if model_dim is None and hasattr(self, 'decoder') and self.decoder and hasattr(self.decoder, 'config'):
                  model_dim = getattr(self.decoder.config, 'hidden_size', getattr(self.decoder.config, 'd_model', None))
             if model_dim is None: raise AttributeError("Could not determine model dimension (hidden_size/d_model)")

             # Safely get scalar start value
             start_value_scalar = self._get_scalar_value(effective_start_token_id, "decoder_start_token_id")

             decoder_input_ids = torch.full(
                 (batch_size, 1, model_dim), 
                 start_value_scalar, # Use the ensured scalar value
                 dtype=torch.float32 if encoder_hidden_states is None else encoder_hidden_states.dtype,
                 device=device,
             )
             current_seq_len = 1
        
        elif not (hasattr(self, 'encoder') and self.encoder is not None):
             decoder_input_ids = input_ids
             current_seq_len = decoder_input_ids.shape[1]

        predictions = []
        past_key_values = None
        internal_decoder_attention_mask = decoder_attention_mask

        # Safely get scalar eos value if needed
        eos_value_scalar = None
        if early_stopping and eos_token_id is not None:
            eos_value_scalar = self._get_scalar_value(eos_token_id, "eos_token_id")

        # 3) Autoregressive Loop
        for step in range(prediction_length):
            step_input_ids = decoder_input_ids[:, -1:, :] if use_cache and past_key_values is not None else decoder_input_ids
            step_attention_mask = internal_decoder_attention_mask if not use_cache or past_key_values is None else None

            if not hasattr(self, 'decoder') or self.decoder is None: raise AttributeError("Model missing decoder")

            decoder_outputs = self.decoder(
                input_ids=step_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=True,
                **kwargs.get("decoder_kwargs", {}),
            )

            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]

            if not hasattr(self, 'output_heads'): raise AttributeError("Model missing output_heads")

            if isinstance(self.output_heads, nn.ModuleList):
                head_outputs = [head(last_hidden) for head in self.output_heads]
                if hasattr(self, 'head_aggregator') and self.head_aggregator:
                    next_pred_features = self.head_aggregator(head_outputs)
                elif head_outputs:
                    next_pred_features = head_outputs[0]
                else: raise ValueError("Output heads list empty")
            else:
                next_pred_features = self.output_heads(last_hidden)
            
            predictions.append(next_pred_features)

            # Prepare input for the next step (handle potential dim mismatch)
            model_dim = getattr(self.config, 'hidden_size', getattr(self.config, 'd_model', None))
            if model_dim is None and hasattr(self, 'decoder') and self.decoder and hasattr(self.decoder, 'config'):
                  model_dim = getattr(self.decoder.config, 'hidden_size', getattr(self.decoder.config, 'd_model', None))
            if model_dim is None: raise AttributeError("Could not determine model dim for next step")

            output_feature_size = next_pred_features.shape[-1]
            if output_feature_size == model_dim:
                next_decoder_input_step = next_pred_features
            elif hasattr(self.decoder, 'value_embedding'):
                 try: next_decoder_input_step = self.decoder.value_embedding(next_pred_features)
                 except RuntimeError as e: raise RuntimeError(f"Output feature/embedding mismatch? Error: {e}")
            else:
                 raise NotImplementedError(f"Output/model dim mismatch ({output_feature_size} vs {model_dim}), no value_embedding found.")

            decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_step], dim=1)
            current_seq_len += 1

            if internal_decoder_attention_mask is not None and not use_cache:
                 try:
                     new_mask_column = torch.ones((batch_size, 1), dtype=internal_decoder_attention_mask.dtype, device=device)
                     internal_decoder_attention_mask = torch.cat([internal_decoder_attention_mask, new_mask_column], dim=1)
                 except Exception as e: print(f"Warning: Mask update failed: {e}")

            if use_cache: past_key_values = decoder_outputs.past_key_values

            # Check for early stopping using the scalar eos value
            if early_stopping and eos_value_scalar is not None:
                try:
                    if torch.isclose(next_pred_features[:, :, 0], torch.tensor(eos_value_scalar, device=device)).all():
                        break
                except IndexError: print("Warning: EOS check failed (IndexError)")

        if not predictions:
             output_feature_size = 1
             if hasattr(self, 'output_heads'):
                 try:
                     out_head = self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads
                     if hasattr(out_head, 'output_size'): output_feature_size = out_head.output_size
                     elif hasattr(out_head, 'out_features'): output_feature_size = out_head.out_features
                     elif hasattr(out_head, 'decoder') and hasattr(out_head.decoder, 'out_features'): output_feature_size = out_head.decoder.out_features
                 except Exception: pass 
             return torch.empty((batch_size, 0, output_feature_size), device=device)

        return torch.cat(predictions, dim=1) # Shape [B, prediction_length, OutputFeatures]
