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
            temp_value = value
            while temp_value.numel() > 1:
                 print(f"Warning: {name} tensor had {temp_value.numel()} elements. Taking first element.")
                 temp_value = temp_value[0]
            if temp_value.numel() == 1:
                 return float(temp_value.item())
            else:
                 raise ValueError(f"Could not reduce {name} tensor (original shape {value.shape}) to a scalar.")
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

        # 1) Prepare Encoder Output
        encoder_hidden_states = None
        input_feature_size = input_ids.shape[-1] # Get feature size from input
        if hasattr(self, 'encoder') and self.encoder is not None:
            if input_ids is None: raise ValueError("Encoder exists but input_ids are None.")
            encoder_outputs = self.encoder(
                input_ids, attention_mask=attention_mask, output_attentions=output_attentions,
                return_dict=True, **kwargs.get("encoder_kwargs", {}),
            )
            encoder_hidden_states = getattr(encoder_outputs, 'last_hidden_state', encoder_outputs)

        # 2) Initialize Decoder Input Sequence
        effective_start_token_id = decoder_start_token_id
        if hasattr(self, 'encoder') and self.encoder is not None and effective_start_token_id is None:
             effective_start_token_id = getattr(self.config, "decoder_start_token_id", None)
             if effective_start_token_id is None: 
                  raise ValueError("Encoder-decoder requires decoder_start_token_id via arg or config.")

        if hasattr(self, 'encoder') and self.encoder is not None:
             # Encoder-Decoder Path
             if effective_start_token_id is None: raise ValueError("Start token ID is None unexpectedly.")
             
             # Use input_feature_size determined from the actual input tensor
             model_feature_size = input_feature_size 

             start_value_scalar = self._get_scalar_value(effective_start_token_id, "decoder_start_token_id")

             # Initialize with the correct feature dimension
             decoder_input_ids = torch.full(
                 (batch_size, 1, model_feature_size), # Shape [B, 1, Feat_Enc]
                 start_value_scalar, 
                 dtype=encoder_hidden_states.dtype if encoder_hidden_states is not None else torch.float32,
                 device=device,
             )
             current_seq_len = 1
        
        else: # Decoder-Only Path
             decoder_input_ids = input_ids
             current_seq_len = decoder_input_ids.shape[1]

        predictions = []
        past_key_values = None
        internal_decoder_attention_mask = decoder_attention_mask
        eos_value_scalar = self._get_scalar_value(eos_token_id, "eos_token_id") if early_stopping and eos_token_id is not None else None

        # 3) Autoregressive Loop
        for step in range(prediction_length):
            # Use last *feature* step as input if using cache, otherwise full history
            step_attention_mask = internal_decoder_attention_mask if not use_cache or past_key_values is None else None
            step_inputs = decoder_input_ids[:, -1:, :] \
                        if use_cache and past_key_values is not None \
                        else decoder_input_ids
            print(f"[debug] step_inputs.shape = {step_inputs.shape}, "
                f"feature_size = {self.config.feature_size}, "
                f"d_model = {self.config.d_model}")
            # decide how to call the decoder
            feat_dim = self.config.feature_size
            hidden_dim = self.config.d_model

            if step_inputs.shape[-1] == feat_dim:
                # raw features: let decoder apply its value_embedding
                dec_call = {"input_ids": step_inputs}
            elif step_inputs.shape[-1] == hidden_dim:
                # already embedded: skip value_embedding
                dec_call = {"inputs_embeds": step_inputs}
            else:
                raise ValueError(
                    f"Step input last dim={step_inputs.shape[-1]} "
                    f"but expected feature_size={feat_dim} or d_model={hidden_dim}"
                )
            if not hasattr(self, 'decoder') or self.decoder is None: raise AttributeError("Model missing decoder")

            decoder_outputs = self.decoder(
                **dec_call,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=True,
                **kwargs.get("decoder_kwargs", {}),
            )

            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :] # Shape [B, 1, HiddenSize]

            # Pass through Output Heads 
            if not hasattr(self, 'output_heads'): raise AttributeError("Model missing output_heads")
            last_hidden_contiguous = last_hidden.contiguous()
            if isinstance(self.output_heads, nn.ModuleList):
                head_outputs = [head(last_hidden_contiguous) for head in self.output_heads]
                if hasattr(self, 'head_aggregator') and self.head_aggregator:
                    next_pred_features = self.head_aggregator(head_outputs)
                elif head_outputs:
                    next_pred_features = head_outputs[0]
                else: raise ValueError("Output heads list empty")
            else:
                next_pred_features = self.output_heads(last_hidden_contiguous) # Shape [B, 1, OutputFeatures]
            
            predictions.append(next_pred_features)

            # Prepare input for the next step - Use the predicted features directly.
            # The decoder's value_embedding should handle projection from OutputFeatures to HiddenSize.
            # Use output head's internal logic to reduce to feedback input
            if hasattr(self.output_heads, "predict"):
                next_decoder_input_step = self.output_heads.predict(next_pred_features)
            else:
                expected_input_features = getattr(self.config, "feature_size", 1)
                next_decoder_input_step = next_pred_features[:, :, :expected_input_features].contiguous()

            # NOTE: The previous logic assuming the input to the next step should be model_dim was likely incorrect.
            # The standard flow is: predict features -> feed features back -> value_embedding projects features to model_dim.

            decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_step], dim=1)
            current_seq_len += 1

            if internal_decoder_attention_mask is not None and not use_cache:
                 try:
                     # Simple mask update assuming [B, Seq] - needs adjustment if mask is 4D
                     new_mask_column = torch.ones((batch_size, 1), dtype=internal_decoder_attention_mask.dtype, device=device)
                     internal_decoder_attention_mask = torch.cat([internal_decoder_attention_mask, new_mask_column], dim=1)
                 except Exception as e: print(f"Warning: Mask update failed: {e}")

            if use_cache: past_key_values = decoder_outputs.past_key_values

            if early_stopping and eos_value_scalar is not None:
                try:
                    if torch.isclose(next_pred_features[:, :, 0], torch.tensor(eos_value_scalar, device=device)).all(): break
                except IndexError: print("Warning: EOS check failed (IndexError)")

        if not predictions:
             output_feature_size = 1 # Default
             if hasattr(self, 'output_heads'):
                 try:
                     out_head = self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads
                     # Infer output feature size from head
                     if hasattr(out_head, 'output_size'): output_feature_size = out_head.output_size
                     elif hasattr(out_head, 'out_features'): output_feature_size = out_head.out_features
                     elif hasattr(out_head, 'decoder') and hasattr(out_head.decoder, 'out_features'): output_feature_size = out_head.decoder.out_features
                 except Exception: pass 
             return torch.empty((batch_size, 0, output_feature_size), device=device)

        return torch.cat(predictions, dim=1) # Shape [B, prediction_length, OutputFeatures]
