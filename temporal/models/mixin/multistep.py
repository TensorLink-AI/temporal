import torch
import torch.nn as nn
from typing import Optional


class MultiStepMixin:
    """Mixin for models supporting multi-step generation (predicting the entire horizon at once)."""

    def generate(
        self,
        input_ids: torch.Tensor,
        prediction_length: int,
        attention_mask: Optional[torch.Tensor] = None, # Mask for encoder input OR history in decoder-only
        decoder_attention_mask: Optional[torch.Tensor] = None, # Explicit mask for decoder self-attention, if needed
        use_cache: bool = False, # Cache is typically not used in multi-step generation
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Multi-step generation in a single forward pass.

        Predicts the entire `prediction_length` horizon simultaneously.

        Args:
            input_ids: Inputs. Shape depends on architecture:
                       - Encoder-Decoder: Encoder input sequence [B, Seq_Enc, Feat_Enc].
                       - Decoder-Only: Known history sequence [B, Seq_Hist, Feat_Hist].
            prediction_length: Number of future time steps to predict.
            attention_mask: Mask for encoder inputs (if encoder exists) or history inputs (decoder-only).
                            Shape [B, Seq_Enc] or [B, Seq_Hist].
            decoder_attention_mask: Optional explicit mask for the decoder's self-attention.
                                    If None, a causal mask might be constructed or assumed internally.
            use_cache: Typically False for multi-step generation. Included for signature consistency.
            output_attentions: Whether to output attention weights.
            **kwargs: Additional arguments passed to encoder/decoder.

        Returns:
            Tensor of predicted sequence, shape [B, prediction_length, Feat_Dec].
        """
        if use_cache:
            # Caching is generally not beneficial for single-pass multi-step generation
            # as the entire output sequence is computed simultaneously.
            # You could potentially implement caching *within* the single forward pass
            # (e.g., if the decoder uses multiple layers), but the standard
            # step-by-step KV caching doesn't apply directly here.
            # We keep the argument for signature consistency but ignore it in this basic implementation.
            pass # Or raise NotImplementedError("use_cache=True not supported for MultiStepMixin")


        batch_size, history_len, feature_size = input_ids.shape
        device = input_ids.device

        encoder_hidden_states: Optional[torch.Tensor] = None
        encoder_attention_mask: Optional[torch.Tensor] = None # Mask for cross-attention
        decoder_input: torch.Tensor
        combined_sequence_len: int

        # 1) Prepare Encoder Output (if encoder exists)
        if hasattr(self, 'encoder') and self.encoder is not None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask, # Mask for encoder self-attention
                output_attentions=output_attentions,
                return_dict=True,
                **kwargs.get("encoder_kwargs", {}),
            )
            if hasattr(encoder_outputs, 'last_hidden_state'):
                encoder_hidden_states = encoder_outputs.last_hidden_state
            else:
                encoder_hidden_states = encoder_outputs # Assume raw tensor

            # Prepare cross-attention mask from encoder's input mask
            if attention_mask is not None:
                 # Expand mask for cross-attention: [B, 1, 1, Seq_Enc]
                 # The decoder implementation will typically handle the broadcasting
                 encoder_attention_mask = attention_mask[:, None, None, :].float()
                 encoder_attention_mask = (1.0 - encoder_attention_mask) * -1e9 # Invert and scale for attention scores
            else:
                 encoder_attention_mask = None # No masking

            # Prepare decoder input: Placeholder for the prediction steps
            # Use the last value of the input_ids to fill the decoder input steps
            last_value = input_ids[:, -1:, :] # [B, 1, F]
            decoder_input = last_value.expand(batch_size, prediction_length, feature_size).clone() # [B, Pred_Len, F]
            combined_sequence_len = prediction_length # Decoder only sees the future steps

            # Decoder self-attention mask (causal for prediction_length)
            if decoder_attention_mask is None:
                # Create a causal mask for the decoder sequence length (prediction_length)
                causal_mask = torch.tril(torch.ones((prediction_length, prediction_length), device=device))
                # Convert to additive mask format expected by many attention layers
                # Shape [1, 1, Pred_Len, Pred_Len] for broadcasting
                decoder_attention_mask = (1.0 - causal_mask)[None, None, :, :] * -1e9

        else:
            # Decoder-only architecture
            # Input_ids are the history. We need to create placeholders for the future.
            # Use the last known value as a simple placeholder for future inputs.
            last_value = input_ids[:, -1:, :] # [B, 1, F]
            future_placeholders = last_value.expand(batch_size, prediction_length, feature_size).clone() # [B, Pred_Len, F]

            # Concatenate history and future placeholders for decoder input
            decoder_input = torch.cat([input_ids, future_placeholders], dim=1) # [B, Hist_Len + Pred_Len, F]
            combined_sequence_len = history_len + prediction_length

            # Decoder self-attention mask (causal for combined length)
            if decoder_attention_mask is None:
                # Create a causal mask for the combined sequence length
                causal_mask = torch.tril(torch.ones((combined_sequence_len, combined_sequence_len), device=device))

                # If an attention_mask for the history was provided, integrate it.
                # Masked history parts should not be attended to by anyone.
                # Future parts should not attend to other future parts ahead of them.
                if attention_mask is not None:
                    # attention_mask is [B, Hist_Len]. Expand to match causal_mask [B, 1, Combined_Len, Combined_Len]
                    # We want rows corresponding to history to be masked based on attention_mask
                    # We want columns corresponding to history to be masked based on attention_mask for all rows
                    expanded_hist_mask = attention_mask[:, None, :, None].expand(batch_size, 1, history_len, combined_sequence_len).float()
                    # Also mask attending *to* masked history positions
                    expanded_hist_mask_T = attention_mask[:, None, None, :].expand(batch_size, 1, combined_sequence_len, history_len).float()

                    # Combine: Start with full causal mask
                    full_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1) # [B, 1, Combined, Combined]

                    # Apply history mask (where history attends to others, and others attend to history)
                    # Mask attending TO history:
                    full_mask[:, :, :, :history_len] = full_mask[:, :, :, :history_len] * expanded_hist_mask_T
                    # Mask attending FROM history (redundant with causal mask, but explicit):
                    # full_mask[:, :, :history_len, :] = full_mask[:, :, :history_len, :] * expanded_hist_mask

                    # Convert to additive mask
                    decoder_attention_mask = (1.0 - full_mask) * -1e9
                else:
                     # Simple causal mask if no history mask provided
                     decoder_attention_mask = (1.0 - causal_mask)[None, None, :, :] * -1e9


            # No encoder, so no cross-attention needed
            encoder_hidden_states = None
            encoder_attention_mask = None


        # 2) Decode the full sequence
        # Decoder attention mask is for self-attention.
        # Encoder attention mask is for cross-attention (if encoder exists).
        decoder_outputs = self.decoder(
            input_ids=decoder_input,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask, # Decoder self-attention mask
            encoder_attention_mask=encoder_attention_mask, # Decoder cross-attention mask
            output_attentions=output_attentions,
            use_cache=False, # Ensure cache is off for decoder call
            return_dict=True,
            **kwargs.get("decoder_kwargs", {}),
        )

        hidden_states = decoder_outputs.last_hidden_state # [B, Seq_Out, H]
                                                          # Seq_Out = Pred_Len (Encoder-Decoder)
                                                          # Seq_Out = Hist_Len + Pred_Len (Decoder-Only)

        # 3) Project output using heads
        # Select only the predicted part if decoder-only
        if not (hasattr(self, 'encoder') and self.encoder is not None):
            # Decoder-only: Take the last `prediction_length` hidden states
            hidden_states_for_heads = hidden_states[:, -prediction_length:, :] # [B, Pred_Len, H]
        else:
            # Encoder-Decoder: The output corresponds directly to prediction_length
            hidden_states_for_heads = hidden_states # [B, Pred_Len, H]


        if isinstance(self.output_heads, nn.ModuleList):
            head_outputs = [head(hidden_states_for_heads) for head in self.output_heads] # List of [B, Pred_Len, Q]
            if hasattr(self, 'head_aggregator') and self.head_aggregator is not None:
                predictions = self.head_aggregator(head_outputs)
            elif head_outputs:
                predictions = head_outputs[0] # fallback to first head
            else:
                 # This should not happen if a model is meant to generate predictions
                 raise ValueError("No output heads found to generate predictions.")
        elif hasattr(self, 'output_heads') and self.output_heads is not None:
            predictions = self.output_heads(hidden_states_for_heads) # [B, Pred_Len, Q]
        else:
             raise ValueError("Model does not have 'output_heads' attribute required for generation.")


        return predictions # Shape [B, prediction_length, Feat_Dec]

    def enable_dropout(self):
        """Enable dropout - usually not needed for deterministic multi-step generation."""
        # While dropout can be enabled, it might not be standard practice
        # for deterministic multi-step generation like it is for MC Dropout
        # in autoregressive sampling. Provide it for completeness if needed.
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

# Example Usage (Conceptual)
# class MyMultiStepModel(MultiStepMixin, BaseModel): # Inherit from mixin and base
#     def __init__(self, config):
#         super().__init__(config)
#         # ... initialize self.encoder (optional), self.decoder, self.output_heads ...
#
# model = MyMultiStepModel(config)
# history_data = torch.randn(4, 50, 8) # Batch=4, History=50, Features=8
# predictions = model.generate(history_data, prediction_length=10)
# assert predictions.shape == (4, 10, config.output_feature_size) # Check output shape

