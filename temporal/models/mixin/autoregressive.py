import torch
import torch.nn as nn
from typing import Optional, Union, Any


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
        Autoregressive generation, supporting encoder-decoder and decoder-only.
        Assumes `input_ids` are pre-processed (i.e., embeddings / hidden states).
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1) Prepare Encoder Output
        encoder_hidden_states = None
        if hasattr(self, 'encoder') and self.encoder is not None:
            encoder_outputs = self.encoder(
                hidden_states=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                return_dict=True,
                **kwargs.get("encoder_kwargs", {}),
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state

        # 2) Initialize Decoder Input Sequence
        if hasattr(self, 'encoder') and self.encoder is not None:
            effective_start_token_id = decoder_start_token_id
            if effective_start_token_id is None:
                effective_start_token_id = getattr(self.config, "decoder_start_token_id", 0)
            start_value_scalar = self._get_scalar_value(effective_start_token_id, "decoder_start_token_id")
            decoder_input_ids = torch.full(
                 (batch_size, 1, encoder_hidden_states.shape[-1]),
                 start_value_scalar,
                 dtype=encoder_hidden_states.dtype,
                 device=device,
            )
        else:
             decoder_input_ids = input_ids

        predictions = []
        past_key_values = None
        internal_decoder_attention_mask = decoder_attention_mask
        eos_value_scalar = self._get_scalar_value(eos_token_id, "eos_token_id")

        # 3) Autoregressive Loop
        for _ in range(prediction_length):
            step_attention_mask = internal_decoder_attention_mask if not use_cache or past_key_values is None else None
            step_inputs = decoder_input_ids[:, -1:, :] if use_cache and past_key_values is not None else decoder_input_ids

            if not hasattr(self, 'decoder'): raise AttributeError("Model missing decoder")
            
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
            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]

            if not hasattr(self, 'output_heads'): raise AttributeError("Model missing output_heads")
            
            last_hidden_contiguous = last_hidden.contiguous()
            if isinstance(self.output_heads, nn.ModuleList):
                head_outputs = [head(last_hidden_contiguous) for head in self.output_heads]
                if hasattr(self, 'head_aggregator') and self.head_aggregator:
                    next_pred_features = self.head_aggregator(head_outputs)
                else:
                    next_pred_features = head_outputs[0]
            else:
                next_pred_features = self.output_heads(last_hidden_contiguous)

            predictions.append(next_pred_features)

            # Use the output head's dedicated `predict` method for feedback if it exists. This is the most robust approach.
            if hasattr(self.output_heads, "predict"):
                next_decoder_input_step = self.output_heads.predict(next_pred_features)
            else:
                # Fallback: Assume the first feature dimension of the prediction is the feedback.
                # This is a reasonable default for simple, single-variate forecasting heads.
                expected_input_features = getattr(self.config, "feature_size", 1)
                next_decoder_input_step = next_pred_features[:, :, :expected_input_features].contiguous()

            decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_step], dim=1)

            if internal_decoder_attention_mask is not None and not use_cache:
                 new_mask_column = torch.ones((batch_size, 1), dtype=internal_decoder_attention_mask.dtype, device=device)
                 internal_decoder_attention_mask = torch.cat([internal_decoder_attention_mask, new_mask_column], dim=1)

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            if early_stopping and eos_value_scalar is not None:
                if torch.isclose(next_decoder_input_step[:, :, 0], torch.tensor(eos_value_scalar, device=device)).all():
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
