import torch
import torch.nn.functional as F
from typing import Optional


class MultistepGenerateMixin:
    def generate_multistep(
        self,
        input_ids: torch.Tensor,
        decoder_length: int,
        encoder_mask_2d: Optional[torch.Tensor] = None,
        decoder_start_token_value: Optional[float] = None,
        causal: bool = True,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Multi-step generation in a single forward pass.

        Args:
            input_ids: [B, S, F] encoder input sequence
            decoder_length: number of future time steps to predict
            encoder_mask_2d: [B, S] mask over encoder sequence (1=keep, 0=mask)
            decoder_start_token_value: float to seed the decoder inputs (optional)
            causal: whether to use a lower-triangular self-attn mask
            output_attentions: return attention weights (passed to encoder/decoder)

        Returns:
            predictions: [B, T, Q]
        """
        device = input_ids.device
        B, S, F = input_ids.shape

        # === Encode input ===
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=encoder_mask_2d,
            output_attentions=output_attentions,
            return_dict=True
        )

        # === Prepare decoder input ===
        if decoder_start_token_value is not None:
            decoder_input = torch.full(
                (B, decoder_length, F),
                decoder_start_token_value,
                dtype=input_ids.dtype,
                device=device
            )
        else:
            # Use last input value repeated
            decoder_input = input_ids[:, -1:, :].expand(B, decoder_length, F).clone()

        # === Causal mask ===
        if causal:
            tri = torch.tril(torch.ones(decoder_length, decoder_length, device=device))
            decoder_mask = (1.0 - tri)[None, None, :, :] * -1e9
        else:
            decoder_mask = None

        # === Cross-attention mask ===
        if encoder_mask_2d is not None:
            cross_mask = (1.0 - encoder_mask_2d[:, None, None, :].float()) * -1e9
        else:
            cross_mask = None

        # === Decode full sequence ===
        decoder_outputs = self.decoder(
            decoder_input,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            attention_mask=decoder_mask,
            encoder_attention_mask=cross_mask,
            output_attentions=output_attentions,
            return_dict=True
        )

        hidden_states = decoder_outputs.last_hidden_state  # [B, T, H]

        # === Project output ===
        if isinstance(self.output_heads, torch.nn.ModuleList):
            head_outputs = [head(hidden_states) for head in self.output_heads]  # List of [B, T, Q]
            if self.head_aggregator is not None:
                predictions = self.head_aggregator(head_outputs)
            else:
                predictions = head_outputs[0]  # fallback to first
        else:
            predictions = self.output_heads(hidden_states)  # [B, T, Q]

        return predictions
