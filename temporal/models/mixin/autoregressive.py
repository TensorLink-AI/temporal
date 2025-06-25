import torch
import torch.nn as nn
from typing import Optional, Union, Any, Tuple

class AutoregressiveMixin:
    def enable_dropout(self):
        """Enable dropout for MC sampling during autoregressive generation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _get_scalar_value(self, value: Union[torch.Tensor, float, int, Any], name: str) -> Optional[float]:
        """ Safely converts a potential tensor value to a float scalar. """
        if value is None:
            return None
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

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        prediction_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        decoder_start_token_id: Optional[Any] = None,
        eos_token_id: Optional[Any] = None,
        early_stopping: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Autoregressively generates a sequence of features.

        This method supports both raw feature inputs and pre-processed embeddings.
        - If `input_ids` has 3 dimensions and the last dimension matches `config.feature_size`,
          it is treated as raw features and passed through the `preprocessor`.
        - Otherwise, it is assumed to be pre-processed embeddings.
        """
        self.eval()
        batch_size, device = input_ids.shape[0], input_ids.device

        # 1) Preprocess inputs if they are raw features
        if hasattr(self, "preprocessor") and input_ids.ndim == 3 and input_ids.shape[-1] == self.config.feature_size:
            proc = self.preprocessor.process(
                input_values=input_ids,
                attention_mask=attention_mask,
                is_causal=False,
                validate_shapes=False,
                verbose=False,
            )
            processed_inputs = proc["hidden_states"]
            attention_mask = proc["attention_mask"]
        else:
            processed_inputs = input_ids  # Assumed to be embeddings

        # 2) Prepare Encoder Output
        encoder_hidden_states = None
        if hasattr(self, 'encoder') and self.encoder is not None:
            encoder_outputs = self.encoder(
                hidden_states=processed_inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                return_dict=True,
                **kwargs.get("encoder_kwargs", {}),
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            hidden_size = encoder_hidden_states.shape[-1]
        else:
            # Decoder-only model
            hidden_size = processed_inputs.shape[-1]

        # 3) Initialize Decoder Input Sequence
        if decoder_start_token_id is None:
            decoder_start_token_id = getattr(self.config, "decoder_start_token_id", 0)
        
        start_val = self._get_scalar_value(decoder_start_token_id, "decoder_start_token_id")
        
        # Create a start tensor: zero for all features except the first.
        decoder_start_tensor = torch.zeros(hidden_size, device=device, dtype=processed_inputs.dtype)
        if hidden_size > 0:
            decoder_start_tensor[0] = start_val
        
        decoder_inputs = decoder_start_tensor.repeat(batch_size, 1, 1)

        # For decoder-only models, the generation starts from the provided inputs
        if not (hasattr(self, 'encoder') and self.encoder is not None):
            decoder_inputs = torch.cat([processed_inputs, decoder_inputs], dim=1)

        predictions = []
        past_key_values = None
        internal_decoder_attention_mask = decoder_attention_mask
        eos_value_scalar = self._get_scalar_value(eos_token_id, "eos_token_id")

        # 4) Autoregressive Loop
        for _ in range(prediction_length):
            step_attention_mask = internal_decoder_attention_mask if not use_cache or past_key_values is None else None
            
            # For caching, only use the last generated step as input
            step_inputs = decoder_inputs[:, -1:, :] if use_cache and past_key_values is not None else decoder_inputs

            if not hasattr(self, 'decoder'):
                raise AttributeError("Model is missing a decoder, which is required for autoregressive generation.")
            
            decoder_outputs = self.decoder(
                hidden_states=step_inputs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=True,
                **kwargs.get("decoder_kwargs", {}),
            )
            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :].contiguous()

            if not hasattr(self, 'output_heads'):
                raise AttributeError("Model is missing output_heads required to project decoder outputs to features.")
            
            # Project to feature space
            if isinstance(self.output_heads, nn.ModuleList):
                head_outputs = [head(last_hidden) for head in self.output_heads]
                if hasattr(self, 'head_aggregator') and self.head_aggregator:
                    next_pred_features = self.head_aggregator(head_outputs)
                else:
                    next_pred_features = head_outputs[0]
            else:
                next_pred_features = self.output_heads(last_hidden)

            predictions.append(next_pred_features)

            # Use the output head's dedicated `predict` method for feedback if it exists.
            if hasattr(self.output_heads, "predict"):
                next_decoder_input_step = self.output_heads.predict(next_pred_features)
            else:
                # Fallback: assume the prediction itself is the input for the next step.
                next_decoder_input_step = next_pred_features

            # The feedback loop requires inputs to be in the hidden dimension space, not feature space.
            # We must re-embed the generated features before feeding them back.
            if hasattr(self, "preprocessor"):
                 # We only need to embed the value, no complex processing needed here.
                 next_decoder_input_step = self.preprocessor.value_embed(next_decoder_input_step)

            decoder_inputs = torch.cat([decoder_inputs, next_decoder_input_step], dim=1)

            if internal_decoder_attention_mask is not None and not use_cache:
                 new_mask_column = torch.ones((batch_size, 1), dtype=internal_decoder_attention_mask.dtype, device=device)
                 internal_decoder_attention_mask = torch.cat([internal_decoder_attention_mask, new_mask_column], dim=1)

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            if early_stopping and eos_value_scalar is not None:
                # Check the first feature of the *predicted* value for the EOS token
                if torch.isclose(next_pred_features[:, :, 0], torch.tensor(eos_value_scalar, device=device)).all():
                    break
        
        if not predictions:
            output_feature_size = 1 
            if hasattr(self, 'output_heads'):
                try:
                    out_head = self.output_heads[0] if isinstance(self.output_heads, nn.ModuleList) else self.output_heads
                    if hasattr(out_head, 'output_size'): output_feature_size = out_head.output_size
                    elif hasattr(out_head, 'out_features'): output_feature_size = out_head.out_features
                except Exception: pass
            return torch.empty((batch_size, 0, output_feature_size), device=device)

        return torch.cat(predictions, dim=1)
