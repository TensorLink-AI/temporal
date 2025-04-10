import torch
import torch.nn.functional as F
from typing import Optional
from temporal.registry.generate import register_generate

@register_generate("multistep")
class MultistepGenerateMixin:
    """
    Mixin for multi-step generation in a single forward pass.

    Requires `self.encoder`, `self.decoder`, `self.head_aggregator`, and `self.output_heads`.
    """

    def generate_multistep(
        self,
        input_ids: torch.Tensor,
        decoder_length: int,
        encoder_mask_2d: Optional[torch.Tensor] = None,
        causal: bool = True,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Multi-step decoding in one forward pass.

        Args:
            input_ids: [B, S, F] encoder input
            decoder_length: int, length of prediction horizon
            encoder_mask_2d: optional [B, S] mask (1 = keep, 0 = pad)
            causal: whether to apply causal mask to decoder self-attn
            output_attentions: include attention outputs (not returned here but passed through)

        Returns:
            predictions: [B, decoder_length, Q]
        """
        device = input_ids.device
        B, S, F = input_ids.shape

        # === 1) Encode
        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=encoder_mask_2d,
            output_attentions=output_attentions,
            return_dict=True
        )

        # === 2) Prepare decoder input
        decoder_input = torch.zeros(B, decoder_length, F, device=device)

        # === 3) Causal mask [B, 1, T, T]
        if causal:
            tri_mask = torch.tril(torch.ones(decoder_length, decoder_length, device=device))
            decoder_mask = (1.0 - tri_mask)[None, None, :, :] * -1e9
            decoder_mask = decoder_mask.expand(B, 1, decoder_length, decoder_length)
        else:
            decoder_mask = None

        # === 4) Optional cross-attention mask [B, 1, T_dec, T_enc]
        if encoder_mask_2d is not None:
            cross_mask = (1.0 - encoder_mask_2d[:, None, None, :].float()) * -1e9
            cross_mask = cross_mask.expand(B, 1, decoder_length, S)
        else:
            cross_mask = None

        # === 5) Decode
        decoder_outputs = self.decoder(
            decoder_input,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            attention_mask=decoder_mask,
            encoder_attention_mask=cross_mask,
            output_attentions=output_attentions,
            return_dict=True
        )

        # === 6) Apply output heads + aggregator
        seq_output = decoder_outputs.last_hidden_state  # [B, T, H]
        head_outputs = [head(seq_output) for head in self.output_heads]  # List of [B, T, Q]
        predictions = self.head_aggregator(head_outputs)  # [B, T, Q]

        return predictions
