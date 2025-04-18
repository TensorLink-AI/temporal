import torch
import torch.nn as nn
from typing import Optional


class AutoregressiveGenerateMixin:
    def enable_dropout(self):
        """Enable dropout for MC sampling during autoregressive generation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def generate(
        self,
        input_ids: torch.Tensor,
        prediction_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        decoder_start_token_value: Optional[float] = None,
        eos_token_value: Optional[float] = None,
        early_stopping: bool = False,
        output_attentions: bool = False,
        feedback_strategy: str = "raw",  # Options: "raw", "mean", "first", "sample"
        **kwargs,
    ) -> torch.Tensor:
        """
        Autoregressive generation for forecasting.

        Returns:
            Tensor of shape [B, T, Q]
        """
        batch_size, context_len = input_ids.shape[:2]
        device = input_ids.device

        # 1) Encode
        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # 2) Init decoder input
        if decoder_start_token_value is not None:
            decoder_input = torch.full(
                (batch_size, 1, self.config.feature_size),
                decoder_start_token_value,
                dtype=input_ids.dtype,
                device=device,
            )
        else:
            decoder_input = input_ids[:, -1:, :]

        predictions = []
        past_key_values = None

        for step in range(prediction_length):
            decoder_outputs = self.decoder(
                decoder_input,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=True,
            )

            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]  # [B, 1, H]

            # Support both single and multiple output heads
            if isinstance(self.output_heads, nn.ModuleList):
                head_outputs = [head(last_hidden) for head in self.output_heads]  # List[B,1,Q]
                if self.head_aggregator:
                    next_pred = self.head_aggregator(head_outputs)  # [B,1,Q]
                else:
                    next_pred = head_outputs[0]  # fallback: use first head
            else:
                next_pred = self.output_heads(last_hidden)  # [B,1,Q]

            predictions.append(next_pred)

            # === FEEDBACK STRATEGY ===
            if feedback_strategy == "first":
                decoder_input = next_pred[:, -1:, :1]  # [B,1,1]
            elif feedback_strategy == "mean":
                decoder_input = next_pred.mean(dim=-1, keepdim=True)  # [B,1,1]
            elif feedback_strategy == "sample":
                # Sample from predicted distribution (if logits over bins)
                probs = torch.softmax(next_pred, dim=-1)
                sampled = torch.multinomial(probs.squeeze(1), 1).float() / probs.size(-1)
                decoder_input = sampled.unsqueeze(-1)  # [B,1,1]
            else:  # "raw" or unrecognized
                decoder_input = next_pred  # [B,1,Q]

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            if early_stopping and eos_token_value is not None:
                if (decoder_input == eos_token_value).all():
                    break

        return torch.cat(predictions, dim=1)  # [B, T, Q]
