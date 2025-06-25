import torch
import torch.nn as nn
from typing import Optional


class MultiStepMixin:
    """Mixin for models supporting multi-step generation (predicting the entire horizon at once)."""

    def enable_dropout(self):
        """Enable dropout for MC sampling during generation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        prediction_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Multi-step generation in a single forward pass.
        Assumes `input_ids` are pre-processed (i.e., embeddings / hidden states).
        """
        self.eval()
        batch_size, history_len, feature_size = input_ids.shape
        device = input_ids.device

        encoder_hidden_states: Optional[torch.Tensor] = None
        encoder_attention_mask_for_cross_attn: Optional[torch.Tensor] = None

        if hasattr(self, 'encoder') and self.encoder is not None:
            # Encoder-Decoder Path
            encoder_outputs = self.encoder(
                hidden_states=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                return_dict=True,
                **kwargs.get("encoder_kwargs", {}),
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_attention_mask_for_cross_attn = attention_mask
            last_value = input_ids[:, -1:, :]
            decoder_input_hidden_states = last_value.expand(batch_size, prediction_length, feature_size).clone()
        else:
            # Decoder-Only Path
            last_value = input_ids[:, -1:, :]
            future_placeholders = last_value.expand(batch_size, prediction_length, feature_size).clone()
            decoder_input_hidden_states = torch.cat([input_ids, future_placeholders], dim=1)
            combined_sequence_len = decoder_input_hidden_states.shape[1]
            
            # Create a combined causal and padding mask if a history mask is provided
            if attention_mask is not None:
                if attention_mask.dim() != 2 or attention_mask.shape[1] != history_len:
                    raise ValueError(f"For decoder-only models, attention_mask must have shape [B, Hist_Len], but got {attention_mask.shape}")
                
                # Expand history mask to the full sequence length
                future_mask = torch.ones((batch_size, prediction_length), dtype=attention_mask.dtype, device=device)
                combined_padding_mask = torch.cat([attention_mask, future_mask], dim=1)
                
                # Create causal mask and combine them
                causal_mask = torch.tril(torch.ones((combined_sequence_len, combined_sequence_len), device=device, dtype=torch.bool))
                decoder_attention_mask = causal_mask[None, :, :] & combined_padding_mask[:, None, :]
            else:
                 # If no history mask, just create a causal mask for the whole sequence
                 causal_mask = torch.tril(torch.ones((combined_sequence_len, combined_sequence_len), device=device, dtype=torch.bool))
                 decoder_attention_mask = causal_mask[None, :, :].expand(batch_size, -1, -1)


        if not hasattr(self, 'decoder'):
             raise AttributeError("Model must have a 'decoder' for generation.")

        decoder_outputs = self.decoder(
            hidden_states=decoder_input_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask_for_cross_attn,
            output_attentions=output_attentions,
            use_cache=False,
            return_dict=True,
            **kwargs.get("decoder_kwargs", {}),
        )
        hidden_states = decoder_outputs.last_hidden_state

        if not (hasattr(self, 'encoder') and self.encoder is not None):
            hidden_states_for_heads = hidden_states[:, -prediction_length:, :]
        else:
            hidden_states_for_heads = hidden_states

        if not hasattr(self, 'output_heads'):
            raise AttributeError("Model must have 'output_heads' for generation.")

        if isinstance(self.output_heads, nn.ModuleList):
            head_outputs = [head(hidden_states_for_heads) for head in self.output_heads]
            if hasattr(self, 'head_aggregator') and self.head_aggregator is not None:
                predictions = self.head_aggregator(head_outputs)
            else:
                predictions = head_outputs[0]
        else:
            predictions = self.output_heads(hidden_states_for_heads)

        return prediction