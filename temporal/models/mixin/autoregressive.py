from temporal.registry.generate import register_generate

@register_generate("ar")
class AutoregressiveGenerateMixin:
    def generate(
        self,
        input_ids: torch.Tensor,
        prediction_length: int,
        decoder_start_token_value: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None,
        early_stopping: bool = False,
        eos_token_value: Optional[float] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        B, T = input_ids.shape[:2]
        device = input_ids.device
        decoder_input = (
            torch.full((B, 1, input_ids.size(-1)), decoder_start_token_value, device=device)
            if decoder_start_token_value is not None
            else input_ids[:, -1:].clone()
        )

        predictions = []
        past_key_values = None

        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        for step in range(prediction_length):
            decoder_outputs = self.decoder(
                decoder_input,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True
            )
            last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]
            head_outputs = [head(last_hidden) for head in self.output_heads]
            next_pred = self.head_aggregator(head_outputs)

            predictions.append(next_pred)
            decoder_input = next_pred[:, -1:, :1]  # next input

            if use_cache:
                past_key_values = decoder_outputs.past_key_values

            if early_stopping and eos_token_value is not None:
                if (decoder_input == eos_token_value).all():
                    break

        return torch.cat(predictions, dim=1)
